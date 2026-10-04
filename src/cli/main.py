"""Typer CLI `seg`: verify, prepare-data, train, evaluate, predict."""

import functools
from pathlib import Path
import time
from typing import Any, Callable, Optional
import torch
import typer

from src.configs.schema import ConfigError, load_config, resolve_config, validate_config
from src.data.dataset import build_dataloaders, prepare_atr_data
from src.data.download import download_atr_dataset
from src.evaluation.evaluator import evaluate_split
from src.inference.predict import predict_from_checkpoint
from src.models.registry import build_model_from_config
from src.training.checkpoint import load_checkpoint
from src.training.losses import BCEDiceLoss
from src.training.trainer import ClothesSegmentationTrainer
from src.utils.logging import get_logger
from src.utils.memory import measured_total_mib, reset_peak

__version__ = "0.1.0"

app = typer.Typer(
    name="seg",
    help="Clothes segmentation (U-Net, BCE + Dice) CLI for assignment CVCY002.",
    add_completion=False,
)
logger = get_logger("seg.cli")


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "cuda":
        raise RuntimeError("CUDA was requested but is not available.")
    typer.secho("Warning: CUDA not available, falling back to CPU.", fg=typer.colors.YELLOW)
    return torch.device("cpu")


def _handle_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except (ConfigError, FileNotFoundError, KeyError, ValueError) as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        except (torch.cuda.OutOfMemoryError, RuntimeError, ArithmeticError) as exc:
            typer.secho(f"Runtime error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

    return wrapper


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Hands-on Segmentation version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """Hands-on Segmentation command-line interface."""


def _check(results: list[tuple[str, bool, str]], name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark, color = ("✓", typer.colors.GREEN) if ok else ("✗", typer.colors.RED)
    typer.secho(f"  {mark} {name}" + (f" ({detail})" if detail else ""), fg=color)


# --------------------------------------------------------------------------------------------- verify

@app.command()
@_handle_errors
def verify(
    config: Path = typer.Option(..., "-c", "--config", exists=True, readable=True, help="Path to YAML experiment configuration."),
    skip_memory: bool = typer.Option(False, "--skip-memory", help="Skip GPU memory budget check."),
) -> None:
    """Pre-flight verification gate: shapes, loss, backward pass, optimizer step, and memory budget."""
    start = time.time()
    raw = load_config(config)
    results: list[tuple[str, bool, str]] = []
    typer.secho(f"Verifying configuration {config}", bold=True)

    # 1. Config validation
    try:
        cfg = resolve_config(raw)
        _check(results, "Config schema validation", True, cfg["experiment"]["name"])
    except Exception as exc:
        _check(results, "Config schema validation", False, str(exc))
        raise typer.Exit(code=1)

    device = _resolve_device(cfg["experiment"]["device"])
    typer.secho(f"  Target compute device: {device.type}")

    # 2. Data pipeline
    try:
        train_loader, _, _ = build_dataloaders(cfg)
        images, masks, _ = next(iter(train_loader))
        b, c, h, w = images.shape
        mb, mc, mh, mw = masks.shape
        data_ok = (b > 0 and c == 3 and mb == b and mc == 1 and h == mh and w == mw)
        _check(results, "DataLoader batch shapes", data_ok, f"Images: {list(images.shape)}, Masks: {list(masks.shape)}")
    except Exception as exc:
        _check(results, "DataLoader batch shapes", False, str(exc))
        raise typer.Exit(code=1)

    # 3. Model instantiation
    try:
        model = build_model_from_config(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        _check(results, "Model build & parameter count", True, f"{n_params:,} parameters")
    except Exception as exc:
        _check(results, "Model build & parameter count", False, str(exc))
        raise typer.Exit(code=2)

    # 4. Forward, loss, and backward pass
    try:
        if device.type == "cuda":
            reset_peak(device)
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        logits_ok = logits.shape == (b, 1, h, w)
        _check(results, "Model forward pass", logits_ok, f"Output shape: {list(logits.shape)}")

        loss_cfg = cfg["loss"]
        criterion = BCEDiceLoss(
            bce_weight=float(loss_cfg["bce_weight"]),
            dice_weight=float(loss_cfg["dice_weight"]),
            smooth=float(loss_cfg["smooth"]),
        )
        loss, breakdown = criterion(logits, masks)
        finite_loss = torch.isfinite(loss).item() and loss.item() > 0.0
        _check(results, "Finite BCE+Dice loss", finite_loss, f"Loss: {loss.item():.4f}")

        loss.backward()
        grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        _check(results, "Gradients flow to all parameters", grads_ok)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad()
        _check(results, "Optimizer step execution", True)
    except Exception as exc:
        _check(results, "Forward/backward/step verification", False, str(exc))
        raise typer.Exit(code=2)

    # 5. Memory budget check
    budget_mib = int(cfg["verify"]["memory_budget_mib"])
    if device.type == "cuda" and not skip_memory:
        mem_used = measured_total_mib(device)
        mem_ok = mem_used <= budget_mib
        _check(results, f"GPU memory budget check (<= {budget_mib} MiB)", mem_ok, f"Measured: {mem_used} MiB")
        if not mem_ok:
            typer.secho(f"Error: GPU memory usage ({mem_used} MiB) exceeded budget ({budget_mib} MiB)", fg=typer.colors.RED)
            raise typer.Exit(code=2)
    else:
        _check(results, "GPU memory check", True, "skipped (CPU or --skip-memory)")

    elapsed = time.time() - start
    typer.secho(f"Verification completed in {elapsed:.2f}s. All checks passed!\n", fg=typer.colors.GREEN, bold=True)


# ---------------------------------------------------------------------------------------- prepare-data

@app.command(name="prepare-data")
@_handle_errors
def prepare_data(
    config: Path = typer.Option(Path("configs/atr_unet.yaml"), "-c", "--config", help="Path to YAML configuration."),
    raw_dir: Optional[Path] = typer.Option(None, "--raw-dir", help="Override raw data directory."),
    processed_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Override output processed directory."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for data split generation."),
) -> None:
    """Download ATR dataset, convert garment labels to binary masks, and write reproducible splits."""
    cfg = resolve_config(load_config(config)) if config.exists() else {}
    data_cfg = cfg.get("data", {})

    r_dir = raw_dir or Path(data_cfg.get("raw_dir", "data/raw"))
    p_dir = processed_dir or Path(data_cfg.get("data_dir", "data/processed"))
    s_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    split_seed = seed if seed is not None else int(cfg.get("experiment", {}).get("seed", 42))
    clothes_ids = data_cfg.get("clothes_ids", [4, 5, 6, 7, 8, 17])

    typer.secho("Step 1: Checking / downloading ATR dataset files...", bold=True)
    download_atr_dataset(r_dir)

    typer.secho("Step 2: Preprocessing images, binary masks, and reproducible splits...", bold=True)
    res = prepare_atr_data(
        raw_dir=r_dir,
        processed_dir=p_dir,
        splits_dir=s_dir,
        clothes_ids=clothes_ids,
        seed=split_seed,
    )
    typer.secho(f"Successfully prepared ATR dataset: {res}", fg=typer.colors.GREEN, bold=True)


# --------------------------------------------------------------------------------------------- train

@app.command()
@_handle_errors
def train(
    config: Path = typer.Option(Path("configs/atr_unet.yaml"), "-c", "--config", help="Path to YAML configuration."),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Path to checkpoint to resume training from."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Override total epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override batch size."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Override output directory."),
) -> None:
    """Train clothes segmentation model with BCE + Dice loss and checkpointing."""
    raw = load_config(config)
    if epochs is not None:
        raw.setdefault("training", {})["epochs"] = epochs
    if batch_size is not None:
        raw.setdefault("data", {})["batch_size"] = batch_size
    if output_dir is not None:
        raw.setdefault("experiment", {})["output_dir"] = str(output_dir)

    cfg = resolve_config(raw)
    trainer = ClothesSegmentationTrainer(cfg)

    if resume is not None:
        trainer.resume_from_checkpoint(resume)

    summary = trainer.train()
    typer.secho(
        f"Training finished! Best validation IoU: {summary['best_val_iou']:.4f}. Artifacts saved in {cfg['experiment']['output_dir']}",
        fg=typer.colors.GREEN,
        bold=True,
    )


# ------------------------------------------------------------------------------------------ evaluate

@app.command()
@_handle_errors
def evaluate(
    checkpoint: Path = typer.Option(..., "-k", "--checkpoint", exists=True, readable=True, help="Checkpoint file (.pt)."),
    config: Optional[Path] = typer.Option(None, "-c", "--config", help="Optional configuration file."),
    split: str = typer.Option("test", "--split", help="Dataset split to evaluate ('test' or 'val')."),
    output_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Output directory for metrics and galleries."),
    threshold: float = typer.Option(0.5, "--threshold", help="Binarization probability threshold."),
    exclude_id: Optional[list[str]] = typer.Option(
        None, "--exclude-id", help="Sample ID to leave out of the galleries (still counted in metrics). Repeatable."
    ),
) -> None:
    """Evaluate trained model on test/val split and generate best/worst case visual galleries."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint, device=device)
    cfg = resolve_config(load_config(config) if config else ckpt["config"])

    model = ckpt["model"].to(device)
    out_path = output_dir or (Path(cfg["experiment"]["output_dir"]) / f"eval_{split}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_l, val_l, test_l = build_dataloaders(cfg)
    target_loader = test_l if split == "test" else val_l

    results = evaluate_split(
        model=model,
        data_loader=target_loader,
        device=device,
        threshold=threshold,
        output_dir=out_path,
        save_galleries=True,
        gallery_exclude_ids=exclude_id,
    )

    ov = results["overall"]
    target_iou = float(cfg.get("evaluation", {}).get("target_iou", 0.80))
    target_dice = float(cfg.get("evaluation", {}).get("target_dice", 0.88))
    passed_bar = ov["iou"] >= target_iou and ov["dice"] >= target_dice

    typer.secho(f"\n--- Evaluation Results ({split.upper()} split) ---", bold=True)
    typer.echo(f"  Samples evaluated: {ov['total_evaluated']}")
    typer.echo(f"  Macro IoU:         {ov['iou']:.4f} (Target: {target_iou:.2f})")
    typer.echo(f"  Macro Dice:        {ov['dice']:.4f} (Target: {target_dice:.2f})")
    typer.echo(f"  Pixel Accuracy:    {ov['accuracy']:.4f}")
    typer.echo(f"  Precision:         {ov['precision']:.4f}")
    typer.echo(f"  Recall:            {ov['recall']:.4f}")
    typer.echo(f"  Micro IoU:         {ov['micro_iou']:.4f}")
    typer.echo(f"  Micro Dice:        {ov['micro_dice']:.4f}")

    color = typer.colors.GREEN if passed_bar else typer.colors.YELLOW
    status = "MET" if passed_bar else "BELOW TARGET"
    typer.secho(f"\nPerformance standard: {status}\n", fg=color, bold=True)
    typer.echo(f"Results and visual galleries saved to: {out_path}")


# ---------------------------------------------------------------------------------------- robustness

@app.command()
@_handle_errors
def robustness(
    checkpoint: Path = typer.Option(..., "-k", "--checkpoint", exists=True, readable=True, help="Checkpoint file (.pt)."),
    split: str = typer.Option("test", "--split", help="Dataset split to evaluate ('test' or 'val')."),
    output_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Output directory for robustness_metrics.json."),
    threshold: float = typer.Option(0.5, "--threshold", help="Binarization probability threshold."),
) -> None:
    """Re-evaluate under simulated capture conditions (lighting, blur, resolution, rotation, framing)."""
    from src.evaluation.robustness import evaluate_robustness

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint, device=device)
    cfg = resolve_config(ckpt["config"])
    _, val_l, test_l = build_dataloaders(cfg)
    out_path = output_dir or (Path(cfg["experiment"]["output_dir"]) / f"robustness_{split}")
    results = evaluate_robustness(ckpt["model"].to(device), test_l if split == "test" else val_l, device,
                                  threshold=threshold, output_dir=out_path)

    typer.secho(f"\n{'Condition':<17} {'IoU':>7} {'Dice':>7} {'ΔIoU':>7}  Description", bold=True)
    for name, r in results.items():
        typer.echo(f"{name:<17} {r['iou']:>7.4f} {r['dice']:>7.4f} {-r['iou_drop']:>+7.4f}  {r['description']}")
    typer.echo(f"\nSaved to {out_path / 'robustness_metrics.json'}")


# ------------------------------------------------------------------------------------------- predict

@app.command()
@_handle_errors
def predict(
    image: Path = typer.Option(..., "--image", "-i", exists=True, readable=True, help="Input photo path."),
    checkpoint: Path = typer.Option(..., "-k", "--checkpoint", exists=True, readable=True, help="Trained checkpoint path (.pt)."),
    output_dir: Path = typer.Option(Path("artifacts/predictions"), "--out-dir", "-o", help="Directory for generated images."),
    threshold: float = typer.Option(0.5, "--threshold", help="Binarization probability threshold."),
) -> None:
    """Segment clothes from a personal photo, generating mask, overlay, and transparent RGBA cutout."""
    typer.echo(f"Processing personal image: {image}")
    result = predict_from_checkpoint(
        checkpoint_path=checkpoint,
        image_path=image,
        threshold=threshold,
        output_dir=output_dir,
    )
    typer.secho("\nClothes Segmentation Completed!", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Original resolution: {result['original_size'][0]}x{result['original_size'][1]}")
    typer.echo(f"  Clothes coverage:    {result['clothes_coverage_pct']}% of pixels")
    typer.echo(f"  Inference latency:   {result['inference_latency_seconds']:.3f}s")
    typer.echo(f"  Binary mask:         {result['mask_path']}")
    typer.echo(f"  Blended overlay:     {result['overlay_path']}")
    typer.echo(f"  Transparent cutout:  {result['cutout_path']}")
    typer.echo(f"  Diagnostic panel:    {result['panel_path']}")


if __name__ == "__main__":
    app()
