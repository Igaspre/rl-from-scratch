"""Lightweight metric helpers for logging and plotting training progress."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def moving_average(values: Sequence[float], window: int = 50) -> np.ndarray:
    """Return the simple moving average of ``values``.

    The moving average is the standard way to smooth the noisy per-episode
    return curves that RL produces, making the underlying learning trend
    visible.

    Args:
        values: Sequence of scalar measurements (e.g. episode returns).
        window: Number of samples in the averaging window. If the input is
            shorter than ``window`` the window shrinks to the input length.

    Returns:
        A 1D array of length ``len(values)`` where element ``i`` is the mean of
        the up-to-``window`` most recent values ending at ``i``.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    # Cumulative-sum trick gives an O(n) sliding window mean.
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    # Pad the head so the output length matches the input length.
    head = np.array(
        [values[: i + 1].mean() for i in range(window - 1)], dtype=np.float64
    )
    return np.concatenate([head, smoothed])
