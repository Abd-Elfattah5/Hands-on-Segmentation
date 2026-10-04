"""Unit tests for U-Net model architectures."""

import pytest
import torch

from src.models.registry import build_model_from_config, get_model
from src.models.unet import ResNet34UNet, StandardUNet


def test_resnet34_unet_forward_shape():
    model = ResNet34UNet(in_channels=3, out_channels=1, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_resnet34_unet_predict_mask():
    model = ResNet34UNet(in_channels=3, out_channels=1, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    mask = model.predict_mask(x, threshold=0.5)
    assert mask.shape == (2, 1, 256, 256)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})


def test_standard_unet_forward_shape():
    model = StandardUNet(in_channels=3, out_channels=1, features=[16, 32, 64])
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)


def test_build_model_from_config():
    cfg = {
        "model": {
            "name": "resnet34_unet",
            "in_channels": 3,
            "out_channels": 1,
            "pretrained": False,
        }
    }
    model = build_model_from_config(cfg)
    assert isinstance(model, ResNet34UNet)
