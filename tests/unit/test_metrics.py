"""Unit tests for segmentation evaluation metrics."""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    SegmentationMetricTracker,
    compute_metrics_batch,
    compute_per_image_metrics,
)


def test_perfect_overlap_metrics():
    targets = torch.zeros((1, 1, 10, 10))
    targets[:, :, :5, :] = 1.0
    preds = targets.clone()

    metrics = compute_metrics_batch(preds, targets)
    assert pytest.approx(metrics["iou"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["dice"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["accuracy"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["precision"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["recall"], abs=1e-4) == 1.0


def test_partial_overlap_metrics():
    # 10x10 = 100 pixels
    # target: first 40 pixels are 1
    # pred: first 20 pixels are 1 (so TP=20, FP=0, FN=20, TN=60)
    # IoU = 20 / (20 + 0 + 20) = 20/40 = 0.50
    # Dice = 2*20 / (40 + 20) = 40/60 = 0.6667
    target = np.zeros((10, 10), dtype=np.uint8)
    target.flat[:40] = 1
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred.flat[:20] = 1

    metrics = compute_per_image_metrics(pred, target)
    assert pytest.approx(metrics["iou"], abs=1e-3) == 0.50
    assert pytest.approx(metrics["dice"], abs=1e-3) == 2.0 / 3.0
    assert pytest.approx(metrics["accuracy"], abs=1e-3) == 0.80
    assert pytest.approx(metrics["precision"], abs=1e-3) == 1.00
    assert pytest.approx(metrics["recall"], abs=1e-3) == 0.50


def test_tracker_accumulation():
    tracker = SegmentationMetricTracker()
    t1 = torch.ones((2, 1, 16, 16))
    p1 = torch.ones((2, 1, 16, 16))
    tracker.update(p1, t1)

    t2 = torch.zeros((2, 1, 16, 16))
    p2 = torch.zeros((2, 1, 16, 16))
    tracker.update(p2, t2)

    res = tracker.compute()
    assert res["total_evaluated"] == 4
    assert res["accuracy"] == 1.0
