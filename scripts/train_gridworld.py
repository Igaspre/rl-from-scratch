"""Train and compare tabular Q-Learning vs SARSA on GridWorld.

Example
-------
::

    python scripts/train_gridworld.py --episodes 600 --seed 0

Outputs (written to ``assets/``):
    * ``gridworld_learning_curves.png`` -- smoothed return for both agents.
    * ``gridworld_policy_qlearning.png`` -- learned value/policy (Q-Learning).
    * ``gridworld_policy_sarsa.png``     -- learned value/policy (SARSA).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from rl.common import moving_average, set_global_seed
from rl.envs import GridWorldEnv
from rl.tabular import QLearningAgent, SarsaAgent
from rl.tabular.visualize import plot_learning_curves, plot_policy_and_values

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def train_q_learning(env: GridWorldEnv, episodes: int, seed: int) -> tuple:
    agent = QLearningAgent(env.n_states, env.n_actions, seed=seed)
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            ep_return += reward
        agent.decay_epsilon()
        returns.append(ep_return)
    return agent, np.array(returns)


def train_sarsa(env: GridWorldEnv, episodes: int, seed: int) -> tuple:
    agent = SarsaAgent(env.n_states, env.n_actions, seed=seed)
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        action = agent.select_action(state)
        done = False
        ep_return = 0.0
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, terminated)
            state, action = next_state, next_action
            ep_return += reward
        agent.decay_epsilon()
        returns.append(ep_return)
    return agent, np.array(returns)


def greedy_rollout_return(env: GridWorldEnv, agent) -> float:
    """Run one greedy episode and report the achieved return."""
    state, _ = env.reset()
    done = False
    total = 0.0
    while not done:
        action = agent.select_action(state, greedy=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(ASSETS_DIR, exist_ok=True)
    set_global_seed(args.seed)

    env = GridWorldEnv()

    q_agent, q_returns = train_q_learning(env, args.episodes, args.seed)
    sarsa_agent, sarsa_returns = train_sarsa(env, args.episodes, args.seed)

    plot_learning_curves(
        {
            "Q-Learning": moving_average(q_returns, args.smooth_window),
            "SARSA": moving_average(sarsa_returns, args.smooth_window),
        },
        title="GridWorld: Q-Learning vs SARSA",
        save_path=os.path.join(ASSETS_DIR, "gridworld_learning_curves.png"),
    )

    plot_policy_and_values(
        env, q_agent.state_values(), q_agent.greedy_policy(),
        title="Q-Learning: learned values & greedy policy",
        save_path=os.path.join(ASSETS_DIR, "gridworld_policy_qlearning.png"),
    )
    plot_policy_and_values(
        env, sarsa_agent.state_values(), sarsa_agent.greedy_policy(),
        title="SARSA: learned values & greedy policy",
        save_path=os.path.join(ASSETS_DIR, "gridworld_policy_sarsa.png"),
    )

    print(f"Trained for {args.episodes} episodes (seed={args.seed}).")
    print(f"  Q-Learning greedy return: {greedy_rollout_return(env, q_agent):+.2f}")
    print(f"  SARSA      greedy return: {greedy_rollout_return(env, sarsa_agent):+.2f}")
    print(f"Plots written to {os.path.abspath(ASSETS_DIR)}")


if __name__ == "__main__":
    main()
