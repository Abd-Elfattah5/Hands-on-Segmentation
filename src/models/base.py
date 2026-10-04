"""Base interface for segmentation models."""

from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class BaseSegmentationModel(nn.Module, ABC):
    """Abstract base class for clothes segmentation models."""

    config: dict[str, Any]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, in_channels, H, W]

        Returns:
            logits: Output logits of shape [B, out_channels, H, W]
        """
        raise NotImplementedError

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probability maps."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Compute binary prediction mask [B, 1, H, W] in {0.0, 1.0}."""
        proba = self.predict_proba(x)
        return (proba > threshold).float()
