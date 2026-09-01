"""Deep Q-Network agent (with an optional Double DQN target).

This module ties together the three ingredients of a stable DQN:

1. A Q-network approximating Q(s, a; theta).
2. A target network Q(s, a; theta-) whose weights are a slowly-updated copy
   of the online network. Bootstrapping against a frozen target removes the
   moving-target instability of naive Q-learning with function approximation.
3. An experience replay buffer providing decorrelated minibatches.

Setting double_dqn=True switches the TD target from vanilla DQN to Double
DQN, which decouples action selection (online net) from action evaluation
(target net) and reduces the overestimation bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.deep.networks import MLPQNetwork
from rl.deep.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    """Hyper-parameters for DQNAgent.

    Defaults are tuned to solve the bundled CartPole task on CPU in a couple of
    minutes.
    """

    state_dim: int
    n_actions: int
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 50_000
    min_buffer: int = 1_000  # transitions to collect before learning starts
    target_update_interval: int = 500  # gradient steps between hard updates
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay_steps: int = 10_000  # linear decay horizon (env steps)
    grad_clip_norm: float = 10.0
    double_dqn: bool = True
    device: str = "cpu"
    seed: int | None = None


class DQNAgent:
    """A value-based deep RL agent implementing DQN and Double DQN."""

    def __init__(self, config: DQNConfig) -> None:
        self.cfg = config
        self.device = torch.device(config.device)

        self.online_net = MLPQNetwork(
            config.state_dim, config.n_actions, config.hidden_sizes
        ).to(self.device)
        self.target_net = MLPQNetwork(
            config.state_dim, config.n_actions, config.hidden_sizes
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss, robust to outliers

        self.buffer = ReplayBuffer(
            config.buffer_capacity, config.state_dim, seed=config.seed
        )
        self._rng = np.random.default_rng(config.seed)

        self.epsilon = config.epsilon_start
        self.total_steps = 0  # environment steps taken
        self.learn_steps = 0  # gradient updates performed

    # Acting

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection over the online network."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.cfg.n_actions))
        with torch.no_grad():
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return int(self.online_net(state_t).argmax(dim=1).item())

    def _update_epsilon(self) -> None:
        """Linearly anneal epsilon from start to min over the decay horizon."""
        frac = min(1.0, self.total_steps / self.cfg.epsilon_decay_steps)
        self.epsilon = self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_min - self.cfg.epsilon_start
        )

    # Learning

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition and advance the exploration schedule."""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self._update_epsilon()

    def learn(self) -> float | None:
        """Perform one gradient update. Returns the loss, or None if warming up."""
        if len(self.buffer) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return None

        batch = self.buffer.sample(self.cfg.batch_size, device=self.device)

        # Q(s, a) for the actions that were actually taken.
        q_values = self.online_net(batch.states)
        q_taken = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # Online net selects the action; target net evaluates it.
                next_actions = self.online_net(batch.next_states).argmax(dim=1)
                next_q = (
                    self.target_net(batch.next_states)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
            else:
                next_q = self.target_net(batch.next_states).max(dim=1).values
            td_target = batch.rewards + self.cfg.gamma * next_q * (1.0 - batch.dones)

        loss = self.loss_fn(q_taken, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.cfg.grad_clip_norm
        )
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # Persistence

    def save(self, path: str) -> None:
        """Serialise network weights and optimiser state to path."""
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore agent state previously written by save()."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
