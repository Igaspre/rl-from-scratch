"""Load a trained DQN agent and evaluate it greedily on CartPole.

Example:
    python scripts/evaluate_dqn.py --model models/dqn_cartpole.pt --episodes 20
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from rl.common import set_global_seed
from rl.deep import DQNAgent, DQNConfig
from rl.envs import CartPoleEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "..", "models", "dqn_cartpole.pt"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true",
                        help="Print the state every 50 steps.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Model not found at {args.model}. Train one first with "
            "`python scripts/train_dqn.py`."
        )

    set_global_seed(args.seed)
    env = CartPoleEnv()
    config = DQNConfig(state_dim=env.state_dim, n_actions=env.n_actions, seed=args.seed)
    agent = DQNAgent(config)
    agent.load(args.model)

    returns = []
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        done = False
        total = 0.0
        step = 0
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            step += 1
            if args.render and step % 50 == 0:
                print(f"  ep {ep} step {step}: x={state[0]:+.2f} theta={state[2]:+.3f}")
        returns.append(total)
        print(f"Episode {ep + 1:2d}: return = {total:.0f}")

    returns = np.array(returns)
    print(f"\nMean return over {args.episodes} episodes: "
          f"{returns.mean():.1f} +/- {returns.std():.1f} (max possible 500)")


if __name__ == "__main__":
    main()
