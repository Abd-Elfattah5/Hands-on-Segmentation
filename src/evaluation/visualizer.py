"""Visualizer utilities: overlays, RGBA cutouts, comparison grids, best/worst galleries."""

from pathlib import Path
from typing import Any, List, Optional, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image


def create_overlay(
    image_rgb: np.ndarray,
    mask_bin: np.ndarray,
    color: Tuple[int, int, int] = (0, 230, 115),
    alpha: float = 0.5,
) -> np.ndarray:
    """Create a semi-transparent colored mask overlay on an RGB image.

    Args:
        image_rgb: [H, W, 3] uint8 numpy array.
        mask_bin: [H, W] binary uint8 array (0 or 255, or 0/1).
        color: RGB tuple for the mask overlay tint.
        alpha: Blending weight for the overlay (0 = only image, 1 = only color).

    Returns:
        [H, W, 3] uint8 numpy array with overlay blended.
    """
    img = image_rgb.astype(np.float32)
    binary = (mask_bin > 0).astype(bool)

    overlay = img.copy()
    overlay[binary] = (1.0 - alpha) * img[binary] + alpha * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_cutout(
    image_rgb: np.ndarray,
    mask_bin: np.ndarray,
) -> np.ndarray:
    """Create an RGBA image where background is transparent and clothes are preserved.

    Args:
        image_rgb: [H, W, 3] uint8 numpy array.
        mask_bin: [H, W] binary uint8 array (0 or 255).

    Returns:
        [H, W, 4] uint8 numpy array with transparent background.
    """
    alpha = ((mask_bin > 0).astype(np.uint8) * 255)[:, :, np.newaxis]
    rgba = np.concatenate([image_rgb, alpha], axis=2)
    return rgba


def create_comparison_panel(
    image_rgb: np.ndarray,
    target_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    metrics: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Generate a side-by-side diagnostic image: Original | Ground Truth | Prediction | Error Map."""
    h, w, _ = image_rgb.shape
    cols = 4 if target_mask is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4.5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    if target_mask is not None:
        gt_overlay = create_overlay(image_rgb, target_mask, color=(0, 150, 255), alpha=0.45)
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        pred_ax = axes[2]
        err_ax = axes[3]
    else:
        pred_ax = axes[1]
        err_ax = axes[2]

    pred_overlay = create_overlay(image_rgb, pred_mask, color=(0, 230, 115), alpha=0.45)
    title = "Predicted Mask"
    if metrics:
        title += f" (IoU: {metrics.get('iou', 0.0):.3f})"
    pred_ax.imshow(pred_overlay)
    pred_ax.set_title(title)
    pred_ax.axis("off")

    # Error map: Green = TP, Red = FP, Blue = FN
    if target_mask is not None:
        p = (pred_mask > 0).astype(bool)
        t = (target_mask > 0).astype(bool)
        tp = np.logical_and(p, t)
        fp = np.logical_and(p, ~t)
        fn = np.logical_and(~p, t)

        error_map = np.zeros((h, w, 3), dtype=np.uint8)
        error_map[tp] = [0, 230, 115]  # Green: True Positive
        error_map[fp] = [235, 60, 60]   # Red: False Positive
        error_map[fn] = [60, 120, 240]  # Blue: False Negative (missed)

        err_overlay = (0.5 * image_rgb.astype(np.float32) + 0.5 * error_map.astype(np.float32)).astype(np.uint8)
        err_ax.imshow(err_overlay)
        err_ax.set_title("Error: Green=TP, Red=FP, Blue=FN")
        err_ax.axis("off")
    else:
        cutout = create_cutout(image_rgb, pred_mask)
        err_ax.imshow(cutout)
        err_ax.set_title("Clothes Cutout (RGBA)")
        err_ax.axis("off")

    plt.tight_layout()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return data


def save_gallery_grid(
    items: List[dict[str, Any]],
    output_path: Path,
    title: str,
    max_items: int = 10,
) -> Path:
    """Save a grid figure containing multiple sample comparisons."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_items = items[:max_items]
    n = len(display_items)
    if n == 0:
        return output_path

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, it in enumerate(display_items):
        img = it["image_rgb"]
        gt = it["target_mask"]
        pred = it["pred_mask"]
        iou = it.get("iou", 0.0)
        dice = it.get("dice", 0.0)
        s_id = it.get("sample_id", f"sample_{i}")

        # Col 0: Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {s_id}")
        axes[i, 0].axis("off")

        # Col 1: Ground Truth
        gt_overlay = create_overlay(img, gt, color=(0, 120, 255), alpha=0.45)
        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        # Col 2: Prediction
        pred_overlay = create_overlay(img, pred, color=(0, 230, 115), alpha=0.45)
        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title(f"Prediction (IoU: {iou:.3f}, Dice: {dice:.3f})")
        axes[i, 2].axis("off")

        # Col 3: Error map
        p = (pred > 0).astype(bool)
        t = (gt > 0).astype(bool)
        error_map = np.zeros_like(img)
        error_map[np.logical_and(p, t)] = [0, 230, 115]
        error_map[np.logical_and(p, ~t)] = [235, 60, 60]
        error_map[np.logical_and(~p, t)] = [60, 120, 240]
        err_overlay = (0.5 * img.astype(np.float32) + 0.5 * error_map.astype(np.float32)).astype(np.uint8)
        axes[i, 3].imshow(err_overlay)
        axes[i, 3].set_title("Errors (Green=TP, Red=FP, Blue=FN)")
        axes[i, 3].axis("off")

    fig.suptitle(title, fontsize=16, y=1.002)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
