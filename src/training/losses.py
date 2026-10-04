"""Loss functions for binary clothes segmentation: BCE + Soft Dice Loss."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation computed on sigmoid probabilities.

    Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Soft Dice loss.

        Args:
            logits: Predicted logits of shape [B, 1, H, W]
            targets: Binary ground truth masks of shape [B, 1, H, W] with values in {0, 1}

        Returns:
            Scalar dice loss tensor.
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: [B, -1]
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Soft Dice Loss.

    Total = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.dice_loss = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss and return total loss plus breakdown dict."""
        pos_weight = self.pos_weight_tensor.to(logits.device) if self.pos_weight_tensor is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        dice = self.dice_loss(logits, targets)

        total = self.bce_weight * bce + self.dice_weight * dice

        breakdown = {
            "loss_total": float(total.detach().item()),
            "loss_bce": float(bce.detach().item()),
            "loss_dice": float(dice.detach().item()),
        }
        return total, breakdown
