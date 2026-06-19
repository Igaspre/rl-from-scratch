"""A fixed-capacity experience replay buffer.

Experience replay is one of the two ideas (with target networks) that made DQN
stable. Storing transitions and sampling them uniformly at random breaks the
strong temporal correlations between consecutive samples, which would otherwise
destabilise the gradient updates of the function approximator.

The buffer is backed by pre-allocated NumPy arrays used as a ring, which keeps
memory bounded and sampling fast.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    """A minibatch of transitions as PyTorch tensors, ready for learning."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Uniform experience replay backed by NumPy ring buffers.

    Args:
        capacity: Maximum number of transitions retained.
        state_dim: Dimensionality of the observation vector.
        seed: Seed for the sampling RNG.
    """

    def __init__(self, capacity: int, state_dim: int, seed: int | None = None) -> None:
        self.capacity = capacity
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Insert a single transition, overwriting the oldest when full."""
        i = self._pos
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Transition:
        """Sample a random minibatch and move it to ``device`` as tensors."""
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Transition(
            states=torch.as_tensor(self._states[idx], device=device),
            actions=torch.as_tensor(self._actions[idx], device=device),
            rewards=torch.as_tensor(self._rewards[idx], device=device),
            next_states=torch.as_tensor(self._next_states[idx], device=device),
            dones=torch.as_tensor(self._dones[idx], device=device),
        )
