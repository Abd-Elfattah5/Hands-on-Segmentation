"""Capture-condition robustness: re-evaluate the test split under controlled image perturbations.

Each condition simulates a real capture problem (dark room, blur, low resolution, tilted camera,
cropped framing, ...). Photometric conditions change only the image; geometric conditions apply the
same transform to the image and the ground-truth mask so the metric stays valid.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, unnormalize_image
from src.evaluation.metrics import SegmentationMetricTracker
from src.models.base import BaseSegmentationModel

Perturbation = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def _photometric(fn: Callable[[torch.Tensor], torch.Tensor]) -> Perturbation:
    return lambda img, mask: (fn(img).clamp(0.0, 1.0), mask)


def _low_resolution(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = img.shape[-2:]
    small = F.interpolate(img, size=(h // 4, w // 4), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False), mask


def _noise(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=img.device).manual_seed(0)
    return (img + 0.08 * torch.randn(img.shape, generator=gen, device=img.device)).clamp(0.0, 1.0), mask


def _rotate(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        TF.rotate(img, 30.0, interpolation=TF.InterpolationMode.BILINEAR),
        TF.rotate(mask, 30.0, interpolation=TF.InterpolationMode.NEAREST),
    )


def _upper_body_crop(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = img.shape[-2:]
    img_c = F.interpolate(img[..., : h // 2, :], size=(h, w), mode="bilinear", align_corners=False)
    mask_c = F.interpolate(mask[..., : h // 2, :], size=(h, w), mode="nearest")
    return img_c, mask_c


CONDITIONS: Dict[str, Tuple[str, Perturbation]] = {
    "clean": ("No change (reference)", lambda i, m: (i, m)),
    "dark": ("Dim lighting, brightness x0.4", _photometric(lambda x: TF.adjust_brightness(x, 0.4))),
    "overexposed": ("Overexposed, brightness x1.8", _photometric(lambda x: TF.adjust_brightness(x, 1.8))),
    "low_contrast": ("Haze / flat light, contrast x0.4", _photometric(lambda x: TF.adjust_contrast(x, 0.4))),
    "grayscale": ("No colour information", _photometric(lambda x: TF.rgb_to_grayscale(x, 3))),
    "blur": ("Out of focus / motion, Gaussian blur sigma=3", _photometric(lambda x: TF.gaussian_blur(x, 15, 3.0))),
    "low_resolution": ("Low resolution, 4x down/up-sampled", _low_resolution),
    "noise": ("Sensor noise, Gaussian std=0.08", _noise),
    "rotated_30": ("Tilted camera, 30 degree rotation", _rotate),
    "upper_body_crop": ("Close-up framing, top half zoomed 2x", _upper_body_crop),
}


@torch.no_grad()
def evaluate_robustness(
    model: BaseSegmentationModel,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate ``model`` on ``data_loader`` once per capture condition and return IoU/Dice per condition."""
    model.eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    trackers = {name: SegmentationMetricTracker(threshold=threshold) for name in CONDITIONS}

    for images, masks, _ in tqdm(data_loader, desc="Robustness"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        raw = unnormalize_image(images)
        for name, (_, perturb) in CONDITIONS.items():
            img_p, mask_p = perturb(raw, masks)
            logits = model((img_p - mean) / std)
            trackers[name].update(logits, mask_p)

    results: Dict[str, Dict[str, float]] = {}
    clean_iou = None
    for name, (desc, _) in CONDITIONS.items():
        m = trackers[name].compute()
        clean_iou = m["iou"] if name == "clean" else clean_iou
        results[name] = {
            "description": desc,
            "iou": m["iou"],
            "dice": m["dice"],
            "precision": m["precision"],
            "recall": m["recall"],
            "iou_drop": (clean_iou - m["iou"]) if clean_iou is not None else 0.0,
        }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "robustness_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return results
