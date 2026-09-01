"""Tests for tabular agents, including an end-to-end learning sanity check."""

import numpy as np

from rl.envs import GridWorldEnv
from rl.tabular import QLearningAgent, SarsaAgent


def test_qtable_shape():
    agent = QLearningAgent(n_states=10, n_actions=4, seed=0)
    assert agent.q_table.shape == (10, 4)


def test_greedy_action_picks_argmax():
    agent = QLearningAgent(n_states=2, n_actions=3, seed=0)
    agent.q_table[0] = [0.1, 0.9, 0.3]
    assert agent.select_action(0, greedy=True) == 1


def test_epsilon_decays_to_floor():
    agent = QLearningAgent(
        n_states=1, n_actions=2, epsilon_start=1.0, epsilon_min=0.1,
        epsilon_decay=0.5, seed=0,
    )
    for _ in range(100):
        agent.decay_epsilon()
    assert agent.epsilon == 0.1


def test_qlearning_update_moves_value_toward_target():
    agent = QLearningAgent(n_states=2, n_actions=2, learning_rate=0.5, gamma=0.9, seed=0)
    agent.q_table[1] = [2.0, 0.0]  # max next value = 2.0
    td_error = agent.update(state=0, action=0, reward=1.0, next_state=1, done=False)
    # target = 1 + 0.9 * 2.0 = 2.8; update = 0.5 * 2.8 = 1.4
    assert np.isclose(agent.q_table[0, 0], 1.4)
    assert np.isclose(td_error, 2.8)


def test_qlearning_done_masks_bootstrap():
    agent = QLearningAgent(n_states=2, n_actions=2, learning_rate=1.0, gamma=0.9, seed=0)
    agent.q_table[1] = [5.0, 5.0]
    agent.update(state=0, action=0, reward=1.0, next_state=1, done=True)
    assert np.isclose(agent.q_table[0, 0], 1.0)  # no bootstrap when done


def test_sarsa_uses_next_action_value():
    agent = SarsaAgent(n_states=2, n_actions=2, learning_rate=1.0, gamma=1.0, seed=0)
    agent.q_table[1] = [10.0, 1.0]
    # Bootstraps from the chosen next action (index 1), not the max.
    agent.update(state=0, action=0, reward=0.0, next_state=1, next_action=1, done=False)
    assert np.isclose(agent.q_table[0, 0], 1.0)


def test_qlearning_solves_small_gridworld():
    """After training, the greedy policy should reach the goal with positive return."""
    env = GridWorldEnv()
    agent = QLearningAgent(env.n_states, env.n_actions, seed=0)

    for _ in range(800):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
        agent.decay_epsilon()

    # Greedy rollout should reach the goal (positive return) in few steps.
    state, _ = env.reset()
    done = False
    total = 0.0
    steps = 0
    while not done and steps < 50:
        action = agent.select_action(state, greedy=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
        steps += 1
    assert total > 0.0
