"""The swarm state record.

Coordinates are ``(row, col)`` throughout the codebase. Row 0 is the top of the
grid, so "up" decreases the row index.

``occupancy`` and ``positions`` hold the same information in two layouts on
purpose: the grid is what neighbour lookups need, the position array is what
per-module work needs. :meth:`SwarmState.validate` is the guard that keeps them
in agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSITION_DTYPE = np.int32


class StateError(ValueError):
    """Raised when a state violates one of its invariants."""


@dataclass
class SwarmState:
    """Occupancy, module positions, and the target shape for one episode.

    Module ids are row indices into ``positions`` and are stable for the whole
    episode, which is what lets an agent name map to a module.
    """

    occupancy: np.ndarray  # (H, W) bool
    positions: np.ndarray  # (N, 2) int32, row-major (row, col)
    target: np.ndarray  # (H, W) bool
    step: int = 0

    # -- construction ----------------------------------------------------

    @classmethod
    def from_positions(
        cls,
        positions: np.ndarray,
        grid_shape: tuple[int, int],
        target: np.ndarray | None = None,
        step: int = 0,
    ) -> SwarmState:
        """Build a state from module positions, deriving the occupancy grid."""
        positions = np.asarray(positions, dtype=POSITION_DTYPE).reshape(-1, 2)

        # Bounds are checked before indexing so an out-of-range position raises
        # StateError rather than a bare IndexError from the assignment below.
        if positions.size:
            if positions[:, 0].min() < 0 or positions[:, 0].max() >= grid_shape[0]:
                raise StateError(f"module row out of bounds for grid {grid_shape}")
            if positions[:, 1].min() < 0 or positions[:, 1].max() >= grid_shape[1]:
                raise StateError(f"module column out of bounds for grid {grid_shape}")

        occupancy = np.zeros(grid_shape, dtype=bool)
        occupancy[positions[:, 0], positions[:, 1]] = True

        if target is None:
            target = np.zeros(grid_shape, dtype=bool)
        else:
            target = np.asarray(target, dtype=bool)

        state = cls(occupancy=occupancy, positions=positions, target=target, step=step)
        state.validate()
        return state

    def copy(self) -> SwarmState:
        """Deep copy. ``target`` is shared, as it never mutates within an episode."""
        return SwarmState(
            occupancy=self.occupancy.copy(),
            positions=self.positions.copy(),
            target=self.target,
            step=self.step,
        )

    # -- shape and size --------------------------------------------------

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.occupancy.shape[0]), int(self.occupancy.shape[1]))

    @property
    def height(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def width(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def n_modules(self) -> int:
        return int(self.positions.shape[0])

    # -- target progress -------------------------------------------------

    @property
    def n_targets(self) -> int:
        return int(self.target.sum())

    @property
    def filled_mask(self) -> np.ndarray:
        """Target cells that currently hold a module."""
        return self.occupancy & self.target

    @property
    def unfilled_mask(self) -> np.ndarray:
        """Target cells that do not yet hold a module."""
        return self.target & ~self.occupancy

    @property
    def n_filled(self) -> int:
        return int(self.filled_mask.sum())

    @property
    def fill_fraction(self) -> float:
        n_targets = self.n_targets
        return self.n_filled / n_targets if n_targets else 1.0

    @property
    def is_complete(self) -> bool:
        """True when every target cell is occupied."""
        return not self.unfilled_mask.any()

    def on_target(self) -> np.ndarray:
        """(N,) bool: whether each module currently sits on a target cell."""
        return self.target[self.positions[:, 0], self.positions[:, 1]]

    # -- invariants ------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`StateError` if any structural invariant is broken.

        Checks shape agreement, bounds, module uniqueness, and that the
        occupancy grid matches the position array exactly. Connectivity is a
        rules-level invariant and lives in :mod:`swarm.sim.moves`.
        """
        if self.occupancy.dtype != bool:
            raise StateError(f"occupancy must be bool, got {self.occupancy.dtype}")
        if self.target.dtype != bool:
            raise StateError(f"target must be bool, got {self.target.dtype}")
        if self.occupancy.shape != self.target.shape:
            raise StateError(
                f"occupancy shape {self.occupancy.shape} != target shape {self.target.shape}"
            )
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise StateError(f"positions must have shape (N, 2), got {self.positions.shape}")

        rows, cols = self.positions[:, 0], self.positions[:, 1]
        if rows.size:
            if rows.min() < 0 or rows.max() >= self.height:
                raise StateError("module row out of bounds")
            if cols.min() < 0 or cols.max() >= self.width:
                raise StateError("module column out of bounds")

        if np.unique(self.positions, axis=0).shape[0] != self.n_modules:
            raise StateError("two modules occupy the same cell")

        if int(self.occupancy.sum()) != self.n_modules:
            raise StateError(
                f"occupancy has {int(self.occupancy.sum())} cells but there are "
                f"{self.n_modules} modules"
            )
        if not self.occupancy[rows, cols].all():
            raise StateError("occupancy grid disagrees with positions")
