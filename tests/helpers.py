"""Shared test helpers."""

from __future__ import annotations

import numpy as np

from swarm.sim.shapes import random_connected_shape
from swarm.sim.state import SwarmState


def state_from_cells(
    cells: list[tuple[int, int]],
    grid_shape: tuple[int, int],
    target: list[tuple[int, int]] | None = None,
) -> SwarmState:
    """Build a state from an explicit list of occupied cells."""
    target_mask = None
    if target is not None:
        target_mask = np.zeros(grid_shape, dtype=bool)
        for row, col in target:
            target_mask[row, col] = True
    return SwarmState.from_positions(np.array(cells, dtype=np.int32), grid_shape, target_mask)


def grid_from_rows(rows: list[str]) -> np.ndarray:
    """Parse an occupancy grid from strings where '#' is occupied."""
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def random_state(rng: np.random.Generator, n_modules: int = 8, grid_shape=(10, 10)) -> SwarmState:
    """A random connected swarm with no target shape."""
    positions = random_connected_shape(rng, n_modules, grid_shape)
    return SwarmState.from_positions(positions, grid_shape)
