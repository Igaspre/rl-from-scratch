"""Train a DQN / Double DQN agent on the from-scratch CartPole environment.

Example:
    python scripts/train_dqn.py --episodes 400 --seed 0          # Double DQN
    python scripts/train_dqn.py --episodes 400 --no-double-dqn   # vanilla DQN

Outputs:
    * models/dqn_cartpole.pt: trained network weights.
    * assets/dqn_cartpole_returns.png: per-episode and smoothed return.
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl.common import moving_average, set_global_seed
from rl.deep import DQNAgent, DQNConfig
from rl.envs import CartPoleEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS_DIR = os.path.join(ROOT, "assets")
MODELS_DIR = os.path.join(ROOT, "models")

# CartPole-v1 is considered "solved" at an average return of 475 over 100
# consecutive episodes (max possible is 500).
SOLVED_THRESHOLD = 475.0
SOLVED_WINDOW = 100


def evaluate(agent: DQNAgent, env: CartPoleEnv, episodes: int, seed: int) -> float:
    """Run the given number of greedy episodes and return the mean return."""
    totals = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + 1000 + i)
        done = False
        total = 0.0
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        totals.append(total)
    return float(np.mean(totals))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--double-dqn", dest="double_dqn", action="store_true",
                        default=True, help="Use the Double DQN target (default).")
    parser.add_argument("--no-double-dqn", dest="double_dqn", action="store_false",
                        help="Use the vanilla DQN target instead.")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="Episodes between checkpoint evaluations.")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Greedy episodes per checkpoint evaluation.")
    args = parser.parse_args()

    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    set_global_seed(args.seed)

    env = CartPoleEnv()
    config = DQNConfig(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        double_dqn=args.double_dqn,
        seed=args.seed,
    )
    agent = DQNAgent(config)

    model_path = os.path.join(MODELS_DIR, "dqn_cartpole.pt")

    returns: list[float] = []
    recent = deque(maxlen=SOLVED_WINDOW)
    solved_episode = None
    best_score = -float("inf")
    best_episode = None

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_return = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.observe(state, action, reward, next_state, terminated)
            agent.learn()
            state = next_state
            ep_return += reward

        returns.append(ep_return)
        recent.append(ep_return)
        avg = float(np.mean(recent))

        # Keep the best policy rather than whichever one training ends on.
        if episode % args.eval_interval == 0 or episode == args.episodes:
            score = evaluate(agent, env, args.eval_episodes, seed=args.seed + 5000)
            if score > best_score:
                best_score = score
                best_episode = episode
                agent.save(model_path)

        if episode % args.log_interval == 0:
            print(
                f"Episode {episode:4d} | return {ep_return:6.1f} | "
                f"avg{SOLVED_WINDOW} {avg:6.1f} | epsilon {agent.epsilon:.3f}"
            )

        if solved_episode is None and len(recent) == SOLVED_WINDOW and avg >= SOLVED_THRESHOLD:
            solved_episode = episode
            print(f"Solved at episode {episode}! (avg{SOLVED_WINDOW} = {avg:.1f})")

    algo = "Double DQN" if args.double_dqn else "DQN"

    # Report on what is actually on disk, not on whatever the last episode
    # happened to leave in memory.
    agent.load(model_path)
    print(
        f"\nBest checkpoint: episode {best_episode} "
        f"(selection score {best_score:.1f}/500 over {args.eval_episodes} episodes)"
    )

    eval_score = evaluate(agent, env, episodes=20, seed=args.seed)
    print(f"{algo} greedy evaluation of the saved policy (20 episodes): "
          f"{eval_score:.1f}/500")
    print(f"Model saved to {os.path.abspath(model_path)}")

    # Plot raw and smoothed returns.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(returns, alpha=0.3, color="tab:blue", label="Return")
    ax.plot(moving_average(returns, 25), color="tab:blue", linewidth=2,
            label="Return (moving avg, 25)")
    ax.axhline(SOLVED_THRESHOLD, color="tab:green", linestyle="--",
               label=f"Solved ({SOLVED_THRESHOLD:.0f})")
    if solved_episode is not None:
        ax.axvline(solved_episode, color="tab:red", linestyle=":",
                   label=f"Solved @ ep {solved_episode}")
 
    if best_episode is not None:
        ax.axvline(best_episode, color="tab:orange", linestyle="-.",
                   label=f"Best checkpoint @ ep {best_episode}")
    
    ax.set_title(f"{algo} on CartPole")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(ASSETS_DIR, "dqn_cartpole_returns.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Training curve saved to {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    main()
