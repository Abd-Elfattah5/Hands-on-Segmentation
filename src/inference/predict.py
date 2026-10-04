"""Inference pipeline for personal photos: high-resolution clothes mask, overlay, and transparent cutout."""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.visualizer import create_comparison_panel, create_cutout, create_overlay
from src.models.base import BaseSegmentationModel
from src.training.checkpoint import load_checkpoint

try:  # HEIC/HEIF (iPhone and many Android cameras) support for PIL
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:  # pragma: no cover - optional dependency
    HEIF_SUPPORTED = False


def load_photo(path: str | Path) -> Image.Image:
    """Open a photo as upright RGB.

    Phone cameras store pixels in sensor orientation and record the intended rotation in the EXIF
    ``Orientation`` tag. Without applying it, portrait photos reach the model rotated 90 degrees,
    a pose the training data almost never contains. HEIC/HEIF files are supported via pillow-heif.
    """
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"} and not HEIF_SUPPORTED:
        raise ValueError(f"{path.name} is HEIC/HEIF; install support with: pip install pillow-heif")
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def predict_image(
    model: BaseSegmentationModel,
    image_path: str | Path,
    device: torch.device | str = "cpu",
    target_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Segment clothes from a single input photo and generate mask, overlay, and cutout.

    Args:
        model: BaseSegmentationModel instance
        image_path: Path to user's personal image (JPEG, PNG, etc.)
        device: Device to run inference on
        target_size: Network input resolution (H, W)
        threshold: Sigmoid probability threshold
        output_dir: Output folder for generated artifacts

    Returns:
        Dictionary containing paths to generated images and metadata.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    device = torch.device(device)
    model.eval()

    orig_image = load_photo(path)
    orig_w, orig_h = orig_image.size
    orig_np = np.array(orig_image)

    # Preprocess
    t_start = time.time()
    resized_img = TF.resize(orig_image, target_size, interpolation=TF.InterpolationMode.BILINEAR)
    img_tensor = TF.to_tensor(resized_img)
    img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    input_batch = img_tensor.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.sigmoid(logits)

        # Upsample probability map to exact original image resolution
        probs_orig = F.interpolate(
            probs,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        mask_binary = (probs_orig[0, 0] >= threshold).cpu().numpy().astype(np.uint8) * 255

    latency = time.time() - t_start

    # Compute clothes coverage percentage
    clothes_pixel_ratio = float((mask_binary > 0).mean()) * 100.0

    # Build outputs
    overlay_np = create_overlay(orig_np, mask_binary)
    cutout_np = create_cutout(orig_np, mask_binary)
    panel_np = create_comparison_panel(orig_np, target_mask=None, pred_mask=mask_binary)

    result: Dict[str, Any] = {
        "image_path": str(path.resolve()),
        "original_size": (orig_w, orig_h),
        "clothes_coverage_pct": round(clothes_pixel_ratio, 2),
        "inference_latency_seconds": round(latency, 4),
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem

        mask_file = out_dir / f"{stem}_mask.png"
        overlay_file = out_dir / f"{stem}_overlay.png"
        cutout_file = out_dir / f"{stem}_clothes.png"
        panel_file = out_dir / f"{stem}_panel.png"

        Image.fromarray(mask_binary).save(mask_file)
        Image.fromarray(overlay_np).save(overlay_file)
        Image.fromarray(cutout_np).save(cutout_file)
        Image.fromarray(panel_np).save(panel_file)

        result.update({
            "mask_path": str(mask_file),
            "overlay_path": str(overlay_file),
            "cutout_path": str(cutout_file),
            "panel_path": str(panel_file),
        })

    return result


def predict_from_checkpoint(
    checkpoint_path: str | Path,
    image_path: str | Path,
    device: torch.device | str = "auto",
    target_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Convenience helper to load model directly from checkpoint and run predict."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    ckpt = load_checkpoint(checkpoint_path, device=dev)
    model = ckpt["model"]
    return predict_image(
        model=model,
        image_path=image_path,
        device=dev,
        target_size=target_size,
        threshold=threshold,
        output_dir=output_dir,
    )
