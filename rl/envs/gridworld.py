"""A small, fully observable GridWorld with a discrete state space.

The environment is intentionally tiny and tabular: every cell of the grid is a
state, and there are four actions (up, right, down, left). That makes it a good
fit for tabular methods such as Q-Learning and SARSA, because the optimal value
function and policy can be inspected directly.

Map legend:
    S   start
    G   goal (terminal, +1)
    T   trap (terminal, -1)
    #   wall (impassable)
    .   free cell

Every non-terminal step incurs a small living penalty to encourage short paths.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Action encoding: the index into ACTION_DELTAS is the action id.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
ACTION_ARROWS: dict[int, str] = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}

# A 5x5 map with two walls and a trap next to the goal. Reaching G while
# avoiding T is non-trivial, so the learned policy is worth looking at.
DEFAULT_MAP: tuple[str, ...] = (
    "S....",
    ".##..",
    "...#.",
    ".#.T.",
    ".#..G",
)


class GridWorldEnv:
    """A deterministic grid world exposing a minimal Gymnasium-style API.

    Args:
        grid_map: Sequence of equal-length strings describing the layout.
        step_penalty: Reward added on every non-terminal transition (negative
            to encourage the agent to reach the goal quickly).
        goal_reward: Terminal reward for reaching G.
        trap_reward: Terminal reward for stepping on T.
        max_steps: Episode is truncated after this many steps.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_map: Optional[tuple[str, ...]] = None,
        step_penalty: float = -0.04,
        goal_reward: float = 1.0,
        trap_reward: float = -1.0,
        max_steps: int = 100,
    ) -> None:
        grid_map = grid_map or DEFAULT_MAP
        if len({len(row) for row in grid_map}) != 1:
            raise ValueError("All rows of grid_map must have the same length.")

        self.grid = [list(row) for row in grid_map]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.max_steps = max_steps

        # Discrete spaces (kept as plain ints to avoid a Gymnasium dependency).
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self._walls: set[int] = set()
        self._goals: set[int] = set()
        self._traps: set[int] = set()
        self._start: Optional[int] = None
        self._parse_map()

        self._agent: int = self._start  # current state index
        self._steps: int = 0
        self._rng = np.random.default_rng()

    # Map parsing helpers

    def _parse_map(self) -> None:
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                cell = self.grid[r][c]
                state = self._to_state(r, c)
                if cell == "#":
                    self._walls.add(state)
                elif cell == "G":
                    self._goals.add(state)
                elif cell == "T":
                    self._traps.add(state)
                elif cell == "S":
                    self._start = state
        if self._start is None:
            raise ValueError("grid_map must contain a start cell 'S'.")
        if not self._goals:
            raise ValueError("grid_map must contain at least one goal 'G'.")

    def _to_state(self, row: int, col: int) -> int:
        return row * self.n_cols + col

    def _to_rowcol(self, state: int) -> tuple[int, int]:
        return divmod(state, self.n_cols)

    # Gymnasium-style API

    def reset(self, seed: Optional[int] = None) -> tuple[int, dict]:
        """Reset the agent to the start cell and return (state, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent = self._start
        self._steps = 0
        return self._agent, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Apply action and return (state, reward, terminated, truncated, info).

        Movement is deterministic. Attempting to walk into a wall or off the
        grid leaves the agent in place (but still costs a step).
        """
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action!r}; expected 0-3.")

        self._steps += 1
        row, col = self._to_rowcol(self._agent)
        d_row, d_col = ACTION_DELTAS[action]
        new_row, new_col = row + d_row, col + d_col

        # Stay put if the move is blocked by a boundary or a wall.
        if (
            0 <= new_row < self.n_rows
            and 0 <= new_col < self.n_cols
            and self._to_state(new_row, new_col) not in self._walls
        ):
            self._agent = self._to_state(new_row, new_col)

        terminated = False
        reward = self.step_penalty
        if self._agent in self._goals:
            reward = self.goal_reward
            terminated = True
        elif self._agent in self._traps:
            reward = self.trap_reward
            terminated = True

        truncated = self._steps >= self.max_steps
        return self._agent, reward, terminated, truncated, {}

    # Introspection helpers (used for visualisation, not required for RL)

    def is_terminal(self, state: int) -> bool:
        return state in self._goals or state in self._traps

    def is_wall(self, state: int) -> bool:
        return state in self._walls

    def is_goal(self, state: int) -> bool:
        return state in self._goals

    def is_trap(self, state: int) -> bool:
        return state in self._traps

    @property
    def start_state(self) -> int:
        return self._start

    def render(self) -> str:
        """Return an ASCII rendering of the current grid with the agent as A."""
        rows = []
        for r in range(self.n_rows):
            chars = []
            for c in range(self.n_cols):
                state = self._to_state(r, c)
                if state == self._agent:
                    chars.append("A")
                else:
                    chars.append(self.grid[r][c])
            rows.append(" ".join(chars))
        return "\n".join(rows)
