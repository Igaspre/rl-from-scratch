"""Tests for the from-scratch environments."""

import numpy as np
import pytest

from rl.envs import CartPoleEnv, GridWorldEnv
from rl.envs.gridworld import DOWN, RIGHT


# --------------------------------------------------------------------------- #
# GridWorld
# --------------------------------------------------------------------------- #
def test_gridworld_reset_returns_start():
    env = GridWorldEnv()
    state, info = env.reset()
    assert state == env.start_state
    assert isinstance(info, dict)


def test_gridworld_walls_block_movement():
    # A 1x3 corridor with a wall in the middle: S # G
    env = GridWorldEnv(grid_map=("S#G",))
    state, _ = env.reset()
    next_state, reward, terminated, truncated, _ = env.step(RIGHT)
    # Moving into the wall keeps the agent in place and does not terminate.
    assert next_state == state
    assert not terminated


def test_gridworld_reaches_goal_with_reward():
    env = GridWorldEnv(grid_map=("SG",), step_penalty=-0.1, goal_reward=1.0)
    env.reset()
    _, reward, terminated, _, _ = env.step(RIGHT)
    assert terminated
    assert reward == pytest.approx(1.0)


def test_gridworld_trap_terminates_with_penalty():
    env = GridWorldEnv(grid_map=("STG",), trap_reward=-1.0)
    env.reset()
    _, reward, terminated, _, _ = env.step(RIGHT)
    assert terminated
    assert reward == pytest.approx(-1.0)


def test_gridworld_truncates_at_max_steps():
    # No way to reach the goal quickly; verify truncation triggers.
    env = GridWorldEnv(grid_map=("S.", "G."), max_steps=3)
    env.reset()
    truncated = False
    for _ in range(3):
        # Bounce against the right wall so we never reach G.
        _, _, terminated, truncated, _ = env.step(RIGHT)
    assert truncated


def test_gridworld_rejects_invalid_action():
    env = GridWorldEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_gridworld_rejects_ragged_map():
    with pytest.raises(ValueError):
        GridWorldEnv(grid_map=("SG", "..."))


# --------------------------------------------------------------------------- #
# CartPole
# --------------------------------------------------------------------------- #
def test_cartpole_reset_shape_and_range():
    env = CartPoleEnv()
    state, _ = env.reset(seed=0)
    assert state.shape == (4,)
    assert state.dtype == np.float32
    assert np.all(np.abs(state) <= 0.05)


def test_cartpole_step_requires_reset():
    env = CartPoleEnv()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_cartpole_reward_is_one_per_step():
    env = CartPoleEnv()
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(1)
    assert reward == 1.0


def test_cartpole_truncates_at_horizon():
    env = CartPoleEnv(max_episode_steps=5)
    env.reset(seed=0)
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated:
            break
    assert truncated or terminated


def test_cartpole_reproducible_with_seed():
    env = CartPoleEnv()
    s1, _ = env.reset(seed=42)
    s2, _ = env.reset(seed=42)
    assert np.allclose(s1, s2)


def test_cartpole_rejects_invalid_action():
    env = CartPoleEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(5)
