"""Unit tests for paired transforms and normalization."""

import numpy as np
from PIL import Image
import torch

from src.data.transforms import PairedSegmentationTransform, unnormalize_image


def test_paired_transform_shapes_and_types():
    transform = PairedSegmentationTransform(image_size=(128, 128), is_train=True)
    img = Image.new("RGB", (200, 300), color=(120, 150, 180))
    mask = Image.new("L", (200, 300), color=255)

    img_tensor, mask_tensor = transform(img, mask)

    assert img_tensor.shape == (3, 128, 128)
    assert mask_tensor.shape == (1, 128, 128)
    assert mask_tensor.dtype == torch.float32
    assert set(mask_tensor.unique().tolist()).issubset({0.0, 1.0})


def test_unnormalize_image():
    x = torch.rand((3, 64, 64))
    # Normalize then unnormalize
    from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
    import torchvision.transforms.functional as TF

    norm_x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    unnorm_x = unnormalize_image(norm_x)

    assert torch.allclose(x, unnorm_x, atol=1e-5)
