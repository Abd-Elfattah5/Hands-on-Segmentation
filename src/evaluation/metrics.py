"""Evaluation metrics for binary semantic segmentation: IoU, Dice, Accuracy, Precision, Recall."""

from typing import Any, Dict
import numpy as np
import torch


def compute_metrics_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute mean segmentation metrics across a batch of predictions and targets.

    Args:
        preds: Predicted logits or probabilities [B, 1, H, W]
        targets: Binary ground truth masks [B, 1, H, W] with values in {0, 1}
        threshold: Binarization threshold if inputs are probabilities/logits
        eps: Small constant to avoid zero-division

    Returns:
        Dictionary of mean metrics: iou, dice, accuracy, precision, recall.
    """
    if preds.min() < 0.0 or preds.max() > 1.0:
        probs = torch.sigmoid(preds)
    else:
        probs = preds

    pred_bin = (probs >= threshold).float()
    target_bin = (targets >= 0.5).float()

    B = preds.shape[0]
    pred_flat = pred_bin.view(B, -1)
    target_flat = target_bin.view(B, -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)
    tn = ((1.0 - pred_flat) * (1.0 - target_flat)).sum(dim=1)

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "iou": float(iou.mean().item()),
        "dice": float(dice.mean().item()),
        "accuracy": float(acc.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def compute_per_image_metrics(
    pred_bin: np.ndarray,
    target_bin: np.ndarray,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute segmentation metrics for a single image (2D arrays)."""
    p = (pred_bin > 0).astype(bool).flatten()
    t = (target_bin > 0).astype(bool).flatten()

    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()

    iou = float((tp + eps) / (tp + fp + fn + eps))
    dice = float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))
    acc = float((tp + tn + eps) / (tp + fp + fn + tn + eps))
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))

    return {
        "iou": iou,
        "dice": dice,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
    }


class SegmentationMetricTracker:
    """Accumulates batch statistics to report accurate macro and micro metrics over an evaluation split."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7) -> None:
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.sample_ious: list[float] = []
        self.sample_dices: list[float] = []
        self.sample_accs: list[float] = []
        self.sample_precisions: list[float] = []
        self.sample_recalls: list[float] = []

        self.total_tp = 0.0
        self.total_fp = 0.0
        self.total_fn = 0.0
        self.total_tn = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.min() < 0.0 or preds.max() > 1.0:
            probs = torch.sigmoid(preds)
        else:
            probs = preds

        pred_bin = (probs >= self.threshold).float()
        target_bin = (targets >= 0.5).float()

        B = preds.shape[0]
        p_flat = pred_bin.view(B, -1)
        t_flat = target_bin.view(B, -1)

        tp = (p_flat * t_flat).sum(dim=1)
        fp = (p_flat * (1.0 - t_flat)).sum(dim=1)
        fn = ((1.0 - p_flat) * t_flat).sum(dim=1)
        tn = ((1.0 - p_flat) * (1.0 - t_flat)).sum(dim=1)

        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        dice = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
        acc = (tp + tn + self.eps) / (tp + fp + fn + tn + self.eps)
        prec = (tp + self.eps) / (tp + fp + self.eps)
        rec = (tp + self.eps) / (tp + fn + self.eps)

        self.sample_ious.extend(iou.cpu().tolist())
        self.sample_dices.extend(dice.cpu().tolist())
        self.sample_accs.extend(acc.cpu().tolist())
        self.sample_precisions.extend(prec.cpu().tolist())
        self.sample_recalls.extend(rec.cpu().tolist())

        self.total_tp += float(tp.sum().item())
        self.total_fp += float(fp.sum().item())
        self.total_fn += float(fn.sum().item())
        self.total_tn += float(tn.sum().item())

    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics."""
        # Macro (mean per image)
        mean_iou = float(np.mean(self.sample_ious)) if self.sample_ious else 0.0
        mean_dice = float(np.mean(self.sample_dices)) if self.sample_dices else 0.0
        mean_acc = float(np.mean(self.sample_accs)) if self.sample_accs else 0.0
        mean_prec = float(np.mean(self.sample_precisions)) if self.sample_precisions else 0.0
        mean_rec = float(np.mean(self.sample_recalls)) if self.sample_recalls else 0.0

        # Micro (corpus level sum)
        micro_iou = float((self.total_tp + self.eps) / (self.total_tp + self.total_fp + self.total_fn + self.eps))
        micro_dice = float((2.0 * self.total_tp + self.eps) / (2.0 * self.total_tp + self.total_fp + self.total_fn + self.eps))

        return {
            "iou": mean_iou,
            "dice": mean_dice,
            "accuracy": mean_acc,
            "precision": mean_prec,
            "recall": mean_rec,
            "micro_iou": micro_iou,
            "micro_dice": micro_dice,
            "total_evaluated": len(self.sample_ious),
        }
