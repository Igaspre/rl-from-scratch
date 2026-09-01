"""Tabular temporal-difference control: Q-Learning and SARSA.

Both agents store an explicit table Q[state, action] and improve it with a
one-step temporal-difference (TD) update. The only difference between them is
the target they bootstrap from:

* Q-Learning (off-policy) bootstraps from the greedy next action:

      target = r + gamma * max_a' Q(s', a')

* SARSA (on-policy) bootstraps from the action actually taken next:

      target = r + gamma * Q(s', a')

Sharing the base class keeps that single difference in one place.
"""

from __future__ import annotations

import numpy as np


class TabularAgent:
    """Base class holding the Q-table and an epsilon-greedy behaviour policy.

    Args:
        n_states: Size of the discrete state space.
        n_actions: Size of the discrete action space.
        learning_rate: Step size alpha for the TD update.
        gamma: Discount factor in [0, 1].
        epsilon_start: Initial exploration rate.
        epsilon_min: Floor for the exploration rate.
        epsilon_decay: Multiplicative decay applied once per episode.
        seed: Seed for the agent's action-selection RNG.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    # Policy

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection; pass greedy=True to skip exploring.

        Ties between equally-valued actions are broken uniformly at random so
        the agent does not get biased toward low-index actions early on.
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q_values = self.q_table[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self._rng.choice(best))

    def decay_epsilon(self) -> None:
        """Decay exploration once per episode, clamped at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Learning, implemented by subclasses

    def update(self, *args, **kwargs) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    # Derived quantities for analysis and visualisation

    def greedy_policy(self) -> np.ndarray:
        """Return the greedy action for every state (shape (n_states,))."""
        return self.q_table.argmax(axis=1)

    def state_values(self) -> np.ndarray:
        """Return V(s) = max_a Q(s, a) for every state."""
        return self.q_table.max(axis=1)


class QLearningAgent(TabularAgent):
    """Off-policy TD control (Watkins' Q-Learning)."""

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Apply the Q-Learning update and return the TD error.

        The done flag masks the bootstrap term: there is no future value once
        the episode has terminated.
        """
        best_next = 0.0 if done else self.q_table[next_state].max()
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return float(td_error)


class SarsaAgent(TabularAgent):
    """On-policy TD control (SARSA)."""

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ) -> float:
        """Apply the SARSA update using the next action that was chosen."""
        next_q = 0.0 if done else self.q_table[next_state, next_action]
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return float(td_error)
