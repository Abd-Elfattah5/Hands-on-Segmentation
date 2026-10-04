"""Atomic checkpoint saving, loading, and provenance tracking."""

from datetime import datetime, timezone
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Optional, Tuple
import numpy as np
import torch

from src.models.base import BaseSegmentationModel
from src.models.registry import build_model_from_config

FORMAT_VERSION = "1.0"


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    model: BaseSegmentationModel,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    best_val_iou: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    metrics: Optional[dict[str, Any]] = None,
    seed: int = 42,
) -> Path:
    """Save training checkpoint atomically using a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    ckpt = {
        "format_version": FORMAT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_iou": float(best_val_iou),
        "history": list(history),
        "rng_state": capture_rng_state(),
        "config": config or getattr(model, "config", {}),
        "metrics": metrics or {},
        "provenance": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": device_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": int(seed),
            "git_commit": _git_commit(),
        },
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp_path)
    tmp_path.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    model: Optional[BaseSegmentationModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load checkpoint and restore model/optimizer/scheduler states."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    device = torch.device(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if model is None:
        model = build_model_from_config(ckpt["config"]).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if restore_rng:
        restore_rng_state(ckpt.get("rng_state"))

    ckpt["model"] = model
    return ckpt
