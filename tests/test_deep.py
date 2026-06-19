"""Tests for the deep RL building blocks (network, buffer, agent)."""

import numpy as np
import torch

from rl.deep import DQNAgent, DQNConfig, MLPQNetwork, ReplayBuffer


def test_network_output_shape():
    net = MLPQNetwork(state_dim=4, n_actions=2, hidden_sizes=(16, 16))
    out = net(torch.zeros(8, 4))
    assert out.shape == (8, 2)


def test_replay_buffer_fills_and_overwrites():
    buf = ReplayBuffer(capacity=3, state_dim=2, seed=0)
    for i in range(5):  # push more than capacity
        buf.push(np.array([i, i], dtype=np.float32), i % 2, float(i),
                 np.array([i + 1, i + 1], dtype=np.float32), False)
    assert len(buf) == 3  # capped at capacity


def test_replay_buffer_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=100, state_dim=4, seed=0)
    for i in range(50):
        buf.push(np.zeros(4, dtype=np.float32), 1, 1.0,
                 np.ones(4, dtype=np.float32), i % 2 == 0)
    batch = buf.sample(16)
    assert batch.states.shape == (16, 4)
    assert batch.actions.shape == (16,)
    assert batch.actions.dtype == torch.int64
    assert batch.dones.dtype == torch.float32


def test_agent_select_action_in_range():
    cfg = DQNConfig(state_dim=4, n_actions=2, seed=0)
    agent = DQNAgent(cfg)
    action = agent.select_action(np.zeros(4, dtype=np.float32))
    assert action in (0, 1)


def test_agent_learn_warms_up_then_trains():
    cfg = DQNConfig(state_dim=4, n_actions=2, batch_size=8, min_buffer=8, seed=0)
    agent = DQNAgent(cfg)
    # Before enough data, learn() is a no-op.
    assert agent.learn() is None
    for _ in range(20):
        agent.observe(np.random.randn(4).astype(np.float32), 0, 1.0,
                      np.random.randn(4).astype(np.float32), False)
    loss = agent.learn()
    assert loss is not None and loss >= 0.0


def test_epsilon_anneals_with_steps():
    cfg = DQNConfig(state_dim=4, n_actions=2, epsilon_start=1.0,
                    epsilon_min=0.1, epsilon_decay_steps=100, seed=0)
    agent = DQNAgent(cfg)
    for _ in range(200):
        agent.observe(np.zeros(4, dtype=np.float32), 0, 0.0,
                      np.zeros(4, dtype=np.float32), False)
    assert np.isclose(agent.epsilon, 0.1)


def test_agent_save_and_load_roundtrip(tmp_path):
    cfg = DQNConfig(state_dim=4, n_actions=2, seed=0)
    agent = DQNAgent(cfg)
    path = tmp_path / "agent.pt"
    agent.save(str(path))

    other = DQNAgent(DQNConfig(state_dim=4, n_actions=2, seed=1))
    other.load(str(path))
    for p1, p2 in zip(agent.online_net.parameters(), other.online_net.parameters()):
        assert torch.allclose(p1, p2)
