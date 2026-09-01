"""Neural network architectures for value-based deep RL."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLPQNetwork(nn.Module):
    """A multilayer-perceptron Q-network.

    Maps a continuous observation to one Q-value per discrete action, i.e. it
    outputs the whole action-value vector Q(s, .) in a single forward pass,
    so one network evaluation yields the values of every action at a state.

    Args:
        state_dim: Dimensionality of the observation vector.
        n_actions: Number of discrete actions.
        hidden_sizes: Width of each hidden layer.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, n_actions)."""
        return self.net(state)
