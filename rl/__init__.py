"""rl: a compact, from-scratch Reinforcement Learning library.

This package contains clean, dependency-light implementations of classic and
deep RL algorithms together with self-contained environments that follow the
Gymnasium API (``reset`` / ``step``). It is organised as a small library so the
same components can be reused across the two showcase projects:

* ``rl.tabular`` -- tabular value-based agents (Q-Learning, SARSA).
* ``rl.deep``    -- deep value-based agents (DQN, Double DQN).
* ``rl.envs``    -- from-scratch environments (GridWorld, CartPole).
* ``rl.common``  -- shared utilities (seeding, metrics).
"""

__version__ = "1.0.0"
