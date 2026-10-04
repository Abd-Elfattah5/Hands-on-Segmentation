"""Training engine for clothes segmentation with BCE + Dice loss, metrics, and checkpointing."""

import json
import math
from pathlib import Path
import time
from typing import Any, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs.schema import resolve_config
from src.data.dataset import build_dataloaders
from src.evaluation.metrics import SegmentationMetricTracker, compute_metrics_batch
from src.models.base import BaseSegmentationModel
from src.models.registry import build_model_from_config
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import BCEDiceLoss
from src.utils.logging import get_logger
from src.utils.memory import measured_total_mib, reset_peak
from src.utils.seeding import seed_everything


class NonFiniteLossError(RuntimeError):
    """Raised if loss evaluates to NaN or Inf."""


class ClothesSegmentationTrainer:
    """Orchestrates training, validation, checkpointing, and metrics tracking."""

    def __init__(
        self,
        config: dict[str, Any],
        model: Optional[BaseSegmentationModel] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = resolve_config(config)
        exp_cfg = self.config["experiment"]
        train_cfg = self.config["training"]
        loss_cfg = self.config["loss"]

        self.seed = int(exp_cfg["seed"])
        seed_everything(self.seed)

        if device is None:
            want = exp_cfg.get("device", "auto")
            if want == "cpu":
                device = torch.device("cpu")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        self.output_dir = Path(exp_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"seg.trainer.{self.output_dir.name}", log_file=self.output_dir / "train.log")

        with open(self.output_dir / "resolved_config.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.config, fh, sort_keys=False)

        # Model
        self.model = (model or build_model_from_config(self.config)).to(self.device)

        # Loss function
        self.criterion = BCEDiceLoss(
            bce_weight=float(loss_cfg["bce_weight"]),
            dice_weight=float(loss_cfg["dice_weight"]),
            smooth=float(loss_cfg["smooth"]),
            pos_weight=loss_cfg.get("pos_weight"),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

        # LR Scheduler with warmup + cosine decay
        self.total_epochs = int(train_cfg["epochs"])
        self.warmup_steps = int(train_cfg["warmup_steps"])

        # DataLoaders
        if train_loader is None or val_loader is None:
            tl, vl, _ = build_dataloaders(self.config)
            self.train_loader = train_loader or tl
            self.val_loader = val_loader or vl
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader

        total_steps = len(self.train_loader) * self.total_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - self.warmup_steps),
            eta_min=1e-6,
        )

        self.history: list[dict[str, Any]] = []
        self.best_val_iou = -1.0
        self.global_step = 0
        self.start_epoch = 1

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume model, optimizer, scheduler, and history from a checkpoint."""
        ckpt = load_checkpoint(
            checkpoint_path,
            device=self.device,
            model=self.model,
            optimizer=self.optimizer,
            restore_rng=True,
        )
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.global_step = int(ckpt["global_step"])
        self.best_val_iou = float(ckpt.get("best_val_iou", -1.0))
        self.history = list(ckpt.get("history", []))
        self.logger.info(f"Resumed from {checkpoint_path} at epoch {self.start_epoch}, best val IoU: {self.best_val_iou:.4f}")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        accum_steps = int(self.config["training"]["grad_accum_steps"])
        clip_val = float(self.config["training"]["gradient_clip_val"])

        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        batch_count = 0
        metric_tracker = SegmentationMetricTracker()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.total_epochs} [Train]", leave=False)
        self.optimizer.zero_grad()

        for step, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss, breakdown = self.criterion(logits, masks)

            if not math.isfinite(loss.item()):
                raise NonFiniteLossError(f"Non-finite loss ({loss.item()}) at epoch {epoch}, step {step}")

            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(self.train_loader):
                if clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.global_step < self.warmup_steps:
                    lr_mult = float(self.global_step + 1) / float(max(1, self.warmup_steps))
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = float(self.config["training"]["lr"]) * lr_mult
                else:
                    self.scheduler.step()

                self.global_step += 1

            total_loss += breakdown["loss_total"]
            total_bce += breakdown["loss_bce"]
            total_dice += breakdown["loss_dice"]
            batch_count += 1

            metric_tracker.update(logits.detach(), masks.detach())
            curr_iou = np.mean(metric_tracker.sample_ious[-images.shape[0]:]) if metric_tracker.sample_ious else 0.0
            pbar.set_postfix({"loss": f"{breakdown['loss_total']:.4f}", "iou": f"{curr_iou:.3f}"})

        metrics = metric_tracker.compute()
        return {
            "loss_total": total_loss / max(1, batch_count),
            "loss_bce": total_bce / max(1, batch_count),
            "loss_dice": total_dice / max(1, batch_count),
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "accuracy": metrics["accuracy"],
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Execute validation pass."""
        self.model.eval()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        batch_count = 0
        metric_tracker = SegmentationMetricTracker()

        for images, masks, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss, breakdown = self.criterion(logits, masks)

            total_loss += breakdown["loss_total"]
            total_bce += breakdown["loss_bce"]
            total_dice += breakdown["loss_dice"]
            batch_count += 1

            metric_tracker.update(logits, masks)

        metrics = metric_tracker.compute()
        return {
            "val_loss_total": total_loss / max(1, batch_count),
            "val_loss_bce": total_bce / max(1, batch_count),
            "val_loss_dice": total_dice / max(1, batch_count),
            "val_iou": metrics["iou"],
            "val_dice": metrics["dice"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
        }

    def train(self) -> dict[str, Any]:
        """Full training loop over all epochs."""
        self.logger.info(
            f"Starting training on device={self.device.type} for {self.total_epochs} epochs. "
            f"Train samples: {len(self.train_loader.dataset)}, Val samples: {len(self.val_loader.dataset)}"
        )
        t_start = time.time()

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            ep_start = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            ep_duration = time.time() - ep_start

            current_lr = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "duration_seconds": round(ep_duration, 2),
                "learning_rate": current_lr,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(record)

            is_best = val_metrics["val_iou"] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics["val_iou"]

            best_marker = " [★ BEST VAL IOU]" if is_best else ""
            self.logger.info(
                f"Epoch {epoch:02d}/{self.total_epochs:02d} ({ep_duration:.1f}s) | "
                f"Train Loss: {train_metrics['loss_total']:.4f} (IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}) | "
                f"Val Loss: {val_metrics['val_loss_total']:.4f} (Val IoU: {val_metrics['val_iou']:.4f}, Val Dice: {val_metrics['val_dice']:.4f}){best_marker}"
            )

            # Checkpoints
            save_checkpoint(
                self.output_dir / "latest.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                global_step=self.global_step,
                best_val_iou=self.best_val_iou,
                history=self.history,
                config=self.config,
                metrics=record,
                seed=self.seed,
            )

            if is_best:
                save_checkpoint(
                    self.output_dir / "best_checkpoint.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_val_iou=self.best_val_iou,
                    history=self.history,
                    config=self.config,
                    metrics=record,
                    seed=self.seed,
                )

            save_every = int(self.config["training"]["save_every"])
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(
                    self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_val_iou=self.best_val_iou,
                    history=self.history,
                    config=self.config,
                    metrics=record,
                    seed=self.seed,
                )

            self._save_metrics_and_curves()

        # Final checkpoint
        save_checkpoint(
            self.output_dir / "final_checkpoint.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.total_epochs,
            global_step=self.global_step,
            best_val_iou=self.best_val_iou,
            history=self.history,
            config=self.config,
            metrics=self.history[-1] if self.history else {},
            seed=self.seed,
        )

        total_time = time.time() - t_start
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes. Best Val IoU: {self.best_val_iou:.4f}")

        summary = {
            "total_epochs": self.total_epochs,
            "total_training_time_seconds": round(total_time, 2),
            "best_val_iou": self.best_val_iou,
            "history": self.history,
        }
        return summary

    def _save_metrics_and_curves(self) -> None:
        """Write metrics.json and plot loss and IoU curves."""
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"best_val_iou": self.best_val_iou, "history": self.history}, fh, indent=2)

        if len(self.history) < 2:
            return

        epochs = [r["epoch"] for r in self.history]
        train_loss = [r["loss_total"] for r in self.history]
        val_loss = [r["val_loss_total"] for r in self.history]
        train_iou = [r["iou"] for r in self.history]
        val_iou = [r["val_iou"] for r in self.history]
        train_dice = [r["dice"] for r in self.history]
        val_dice = [r["val_dice"] for r in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        ax1.plot(epochs, train_loss, "b-o", label="Train Loss (BCE+Dice)")
        ax1.plot(epochs, val_loss, "r-s", label="Val Loss (BCE+Dice)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Metrics curve
        ax2.plot(epochs, train_iou, "g-o", label="Train IoU")
        ax2.plot(epochs, val_iou, color="darkgreen", linestyle="-", marker="s", label="Val IoU")
        ax2.plot(epochs, train_dice, "m--^", label="Train Dice")
        ax2.plot(epochs, val_dice, color="darkmagenta", linestyle="--", marker="d", label="Val Dice")
        ax2.axhline(0.80, color="gray", linestyle=":", label="Target IoU (0.80)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric Score")
        ax2.set_title("IoU and Dice Scores")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(self.output_dir / "learning_curves.png", dpi=150)
        plt.close(fig)
