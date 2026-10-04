"""Paired transforms and data augmentation for images and segmentation masks."""

import random
from typing import Tuple
from PIL import Image
import torch
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PairedSegmentationTransform:
    """Synchronized geometric transformations and image-only photometric augmentations.

    Args:
        image_size: Target (height, width) or int for square resolution.
        is_train: Whether to apply random augmentations (flip, affine, color jitter).
        random_flip_prob: Probability of horizontal flip.
        scale_range: Range for random scaling (min, max).
        rotation_degrees: Maximum degrees of random rotation (+/-).
        color_jitter: Whether to apply color jitter to the image.
    """

    def __init__(
        self,
        image_size: int | Tuple[int, int] = 256,
        is_train: bool = True,
        random_flip_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_degrees: float = 10.0,
        color_jitter: bool = True,
    ) -> None:
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.is_train = is_train
        self.random_flip_prob = random_flip_prob
        self.scale_range = scale_range
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image and mask synchronously.

        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L or 1) where >0 represents foreground clothes.

        Returns:
            image_tensor: [3, H, W] float32 tensor normalized with ImageNet stats.
            mask_tensor: [1, H, W] float32 tensor with values in {0.0, 1.0}.
        """
        # Ensure correct modes
        if image.mode != "RGB":
            image = image.convert("RGB")
        if mask.mode != "L":
            mask = mask.convert("L")

        # 1. Resize first to target canvas (or resize at the end)
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # 2. Random horizontal flip
            if random.random() < self.random_flip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 3. Random affine (rotation, scale, slight translation)
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            max_dx = float(self.image_size[1]) * 0.05
            max_dy = float(self.image_size[0]) * 0.05
            translations = (
                int(round(random.uniform(-max_dx, max_dx))),
                int(round(random.uniform(-max_dy, max_dy))),
            )

            image = TF.affine(
                image,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=0.0,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=0.0,
                interpolation=TF.InterpolationMode.NEAREST,
            )

            # 4. Photometric augmentation (image only)
            if self.color_jitter:
                # Random brightness, contrast, saturation
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                saturation_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)
                image = TF.adjust_saturation(image, saturation_factor)

        # Convert to tensors
        image_tensor = TF.to_tensor(image)  # [3, H, W], float32 in [0, 1]
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Mask: threshold to binary {0.0, 1.0}
        mask_tensor = TF.to_tensor(mask)  # [1, H, W], float32 in [0, 1]
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


def unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization on a [3, H, W] or [B, 3, H, W] tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device, dtype=tensor.dtype)
    std = torch.tensor(IMAGENET_STD, device=tensor.device, dtype=tensor.dtype)
    if tensor.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    elif tensor.ndim == 3:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    unnorm = tensor * std + mean
    return torch.clamp(unnorm, 0.0, 1.0)
