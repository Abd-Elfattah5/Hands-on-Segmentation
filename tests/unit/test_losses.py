"""Unit tests for segmentation losses (BCE + Soft Dice)."""

import pytest
import torch

from src.training.losses import BCEDiceLoss, SoftDiceLoss


def test_soft_dice_perfect_overlap():
    loss_fn = SoftDiceLoss(smooth=1e-5)
    # High positive logits correspond to sigmoid ~ 1.0
    logits = torch.full((1, 1, 32, 32), 20.0)
    targets = torch.ones((1, 1, 32, 32))
    loss = loss_fn(logits, targets)
    assert loss.item() < 1e-4


def test_soft_dice_complete_mismatch():
    loss_fn = SoftDiceLoss(smooth=1e-5)
    # High positive logits (predict 1), but targets are 0
    logits = torch.full((1, 1, 32, 32), 20.0)
    targets = torch.zeros((1, 1, 32, 32))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.99


def test_bce_dice_combined_loss():
    loss_fn = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    total_loss, breakdown = loss_fn(logits, targets)
    assert torch.isfinite(total_loss)
    assert breakdown["loss_total"] > 0
    assert "loss_bce" in breakdown
    assert "loss_dice" in breakdown

    total_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
