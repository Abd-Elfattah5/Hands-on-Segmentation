"""Deterministic seeding utility for reproducible experiments across runtimes."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Sets global random seeds for Python, NumPy, and PyTorch (CPU and CUDA).

    Enforces deterministic algorithm execution where supported per Constitution Principle II.

    Args:
        seed: The integer random seed to apply globally.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a dedicated, seeded random generator.

    Sampling and evaluation draw all noise from this generator so results do not depend on
    how much of the global RNG stream other code has consumed (research §7, SC-006).

    Args:
        seed: Integer seed.
        device: Device on which random tensors will be drawn.

    Returns:
        A seeded ``torch.Generator`` bound to ``device``.
    """
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))
