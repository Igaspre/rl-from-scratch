"""Deep value-based agents (DQN / Double DQN) built on PyTorch."""

from rl.deep.dqn import DQNAgent, DQNConfig
from rl.deep.networks import MLPQNetwork
from rl.deep.replay_buffer import ReplayBuffer, Transition

__all__ = [
    "DQNAgent",
    "DQNConfig",
    "MLPQNetwork",
    "ReplayBuffer",
    "Transition",
]
