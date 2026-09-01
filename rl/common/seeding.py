"""Reproducibility helpers.

Reinforcement learning results vary a lot between random seeds, so seeding every
source of randomness (Python's random, NumPy and, when available, PyTorch)
is what makes runs comparable to each other.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every RNG we rely on.

    Args:
        seed: The seed shared by all random number generators.
        deterministic_torch: If True and PyTorch is installed, force the
            cuDNN backend into deterministic mode. This trades a little speed
            for bit-for-bit reproducibility on the same hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # PyTorch is an optional dependency for the tabular project.
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
