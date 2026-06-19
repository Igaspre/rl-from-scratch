"""Shared utilities used across tabular and deep RL agents."""

from rl.common.metrics import moving_average
from rl.common.seeding import set_global_seed

__all__ = ["moving_average", "set_global_seed"]
