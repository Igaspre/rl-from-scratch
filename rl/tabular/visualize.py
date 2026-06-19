"""Matplotlib helpers to visualise tabular agents on a GridWorld.

These functions turn an otherwise opaque Q-table into the two pictures that
make a tabular RL result legible at a glance: the learned greedy policy
(arrows) and the state-value function (a heatmap).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend so scripts run without a display
import matplotlib.pyplot as plt
import numpy as np

from rl.envs.gridworld import ACTION_ARROWS, GridWorldEnv


def plot_policy_and_values(
    env: GridWorldEnv,
    state_values: np.ndarray,
    greedy_policy: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Render the value heatmap with the greedy policy overlaid as arrows."""
    values = state_values.reshape(env.n_rows, env.n_cols).copy()

    # Mask walls so they do not skew the colour scale.
    mask = np.zeros_like(values, dtype=bool)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.is_wall(r * env.n_cols + c):
                mask[r, c] = True
    masked_values = np.ma.masked_where(mask, values)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="dimgray")
    im = ax.imshow(masked_values, cmap=cmap)
    fig.colorbar(im, ax=ax, shrink=0.8, label="V(s) = max_a Q(s, a)")

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            state = r * env.n_cols + c
            if env.is_wall(state):
                ax.text(c, r, "#", ha="center", va="center", color="white")
                continue
            if state in env._goals:
                ax.text(c, r, "G", ha="center", va="center",
                        color="white", fontweight="bold")
                continue
            if state in env._traps:
                ax.text(c, r, "T", ha="center", va="center",
                        color="red", fontweight="bold")
                continue
            arrow = ACTION_ARROWS[int(greedy_policy[state])]
            label = arrow + ("\nS" if state == env.start_state else "")
            ax.text(c, r, label, ha="center", va="center",
                    color="white", fontsize=14)

    ax.set_title(title)
    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_learning_curves(
    curves: dict[str, np.ndarray],
    title: str,
    save_path: str,
    xlabel: str = "Episode",
    ylabel: str = "Return (moving average)",
) -> None:
    """Plot one or more smoothed learning curves on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in curves.items():
        ax.plot(values, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
