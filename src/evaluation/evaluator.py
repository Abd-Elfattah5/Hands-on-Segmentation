"""Full dataset evaluation, per-sample metrics, condition breakdowns, and gallery generation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.transforms import unnormalize_image
from src.evaluation.metrics import SegmentationMetricTracker, compute_per_image_metrics
from src.evaluation.visualizer import save_gallery_grid
from src.models.base import BaseSegmentationModel
from src.training.checkpoint import load_checkpoint


def evaluate_split(
    model: BaseSegmentationModel,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None,
    save_galleries: bool = True,
    max_gallery_items: int = 8,
    gallery_exclude_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate segmentation model on a dataset split.

    Args:
        model: Trained BaseSegmentationModel
        data_loader: DataLoader for the target split (test or val)
        device: torch device
        threshold: Sigmoid probability threshold for binary mask
        output_dir: Directory to save eval_metrics.json and galleries
        save_galleries: Whether to generate best/worst case visualization grids
        max_gallery_items: Number of items in best/worst galleries
        gallery_exclude_ids: Sample IDs left out of the best/worst galleries and lists only
            (e.g. images not suitable for publication). They still count in all metrics.

    Returns:
        Dictionary of aggregated metrics and per-condition breakdown.
    """
    model.eval()
    metric_tracker = SegmentationMetricTracker(threshold=threshold)
    per_sample_results: List[Dict[str, Any]] = []

    # Category breakdowns by garment size: small (<15%), medium (15-40%), large (>40%)
    area_categories = {"small": [], "medium": [], "large": []}

    print(f"Evaluating {len(data_loader.dataset)} samples on {device}...")
    with torch.no_grad():
        for images, masks, sample_ids in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            metric_tracker.update(preds, masks)

            # Unnormalize images for visualization and per-sample analysis
            unnorm_images = unnormalize_image(images).cpu()
            preds_cpu = preds.cpu()
            masks_cpu = masks.cpu()

            B = images.shape[0]
            for i in range(B):
                s_id = sample_ids[i]
                img_rgb = (unnorm_images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gt_mask = (masks_cpu[i, 0].numpy() * 255).astype(np.uint8)
                p_mask = (preds_cpu[i, 0].numpy() * 255).astype(np.uint8)

                metrics = compute_per_image_metrics(p_mask, gt_mask)
                gt_area_ratio = float((gt_mask > 0).mean())

                # Classify into size category
                if gt_area_ratio < 0.15:
                    area_cat = "small"
                elif gt_area_ratio <= 0.40:
                    area_cat = "medium"
                else:
                    area_cat = "large"

                area_categories[area_cat].append(metrics["iou"])

                item_info = {
                    "sample_id": s_id,
                    "image_rgb": img_rgb,
                    "target_mask": gt_mask,
                    "pred_mask": p_mask,
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "area_ratio": gt_area_ratio,
                    "area_category": area_cat,
                }
                per_sample_results.append(item_info)

    overall_metrics = metric_tracker.compute()

    # Per-condition analysis
    condition_breakdown = {
        cat: {
            "count": len(ious),
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "median_iou": float(np.median(ious)) if ious else 0.0,
        }
        for cat, ious in area_categories.items()
    }

    # Sort samples by IoU for best and worst galleries
    excluded = set(gallery_exclude_ids or [])
    sorted_samples = sorted(
        (s for s in per_sample_results if s["sample_id"] not in excluded),
        key=lambda x: x["iou"],
        reverse=True,
    )
    best_cases = sorted_samples[:max_gallery_items]
    worst_cases = sorted_samples[-max_gallery_items:]

    results = {
        "overall": overall_metrics,
        "conditions": condition_breakdown,
        "gallery_excluded_ids": sorted(excluded),
        "best_samples": [
            {"id": s["sample_id"], "iou": s["iou"], "dice": s["dice"], "area": s["area_category"]}
            for s in best_cases
        ],
        "worst_samples": [
            {"id": s["sample_id"], "iou": s["iou"], "dice": s["dice"], "area": s["area_category"]}
            for s in worst_cases
        ],
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if save_galleries:
            save_gallery_grid(
                best_cases,
                out_path / "best_cases_gallery.png",
                title=f"Best Segmentation Cases (Top IoU on Test Set)",
                max_items=max_gallery_items,
            )
            save_gallery_grid(
                worst_cases,
                out_path / "worst_cases_gallery.png",
                title=f"Worst Segmentation Cases / Failure Modes (Lowest IoU on Test Set)",
                max_items=max_gallery_items,
            )

    return results
