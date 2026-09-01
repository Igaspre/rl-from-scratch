"""A from-scratch CartPole environment (classic control).

The dynamics reproduce the canonical CartPole-v1 task: balance a pole hinged on
a cart by pushing the cart left or right. The implementation follows the same
semi-implicit Euler integration and termination thresholds as the reference
Gymnasium environment, so a policy trained here transfers to the real one.

State (np.float32 of shape (4,)):
    [cart position, cart velocity, pole angle (rad), pole angular velocity]
Actions (discrete, 2):
    0 = push left, 1 = push right
Reward:
    +1 for every step the pole stays upright.
Termination:
    pole angle exceeds +-12 degrees, or the cart leaves +-2.4 units.
Truncation:
    episode reaches max_episode_steps (default 500).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class CartPoleEnv:
    """Continuous-state, discrete-action pole-balancing task."""

    metadata = {"render_modes": []}

    # Physical constants (identical to the classic Barto/Sutton formulation).
    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_half_length: float = 0.5  # actually half the pole's length
    force_mag: float = 10.0
    tau: float = 0.02  # seconds between state updates

    # Termination thresholds.
    x_threshold: float = 2.4
    theta_threshold_radians: float = 12 * math.pi / 180

    def __init__(self, max_episode_steps: int = 500) -> None:
        self.max_episode_steps = max_episode_steps
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_mass_length = self.mass_pole * self.pole_half_length

        self.state_dim = 4
        self.n_actions = 2

        self._state: Optional[np.ndarray] = None
        self._steps = 0
        self._rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """Start a new episode from a small random state near the origin."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Reference CartPole samples each component uniformly in [-0.05, 0.05].
        self._state = self._rng.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self._steps = 0
        return self._state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the physics one tick given a discrete action (0 or 1)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if action not in (0, 1):
            raise ValueError(f"Invalid action {action!r}; expected 0 or 1.")

        x, x_dot, theta, theta_dot = self._state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Equations of motion for the cart-pole system.
        temp = (
            force + self.pole_mass_length * theta_dot**2 * sin_theta
        ) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_half_length
            * (4.0 / 3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        # Semi-implicit Euler integration (matches Gymnasium's default).
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._steps += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = self._steps >= self.max_episode_steps
        reward = 1.0  # survival bonus
        return self._state.copy(), reward, terminated, truncated, {}
