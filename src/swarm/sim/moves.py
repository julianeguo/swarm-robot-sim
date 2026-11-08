"""The movement rules. This is the only file that decides what is legal.

Every other part of the codebase -- observations, the environment wrapper,
baselines -- asks this module rather than reimplementing legality, so the action
mask can never drift away from what :func:`apply_actions` actually permits.

Current rule set: a module either stays put or translates one cell to an
orthogonally adjacent cell. A move is legal when the destination is in bounds,
the destination is empty, and the swarm is still a single 4-connected group
afterwards. Swapping in sliding or pivoting rules means replacing the body of
:func:`_is_legal` and nothing else.

Moves are chosen simultaneously but applied one at a time in a random order,
with legality re-checked against the partially-updated state. A move that was
legal when the mask was computed can therefore still be rejected -- another
module took the cell, or the connectivity picture changed. Rejected moves become
no-ops and are counted in the returned info so the rejection rate can be
monitored; if it climbs, the simultaneity assumption is the thing to revisit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from swarm.sim.connectivity import is_connected
from swarm.sim.state import SwarmState

STAY, UP, DOWN, LEFT, RIGHT = range(5)
N_ACTIONS = 5
ACTION_NAMES: tuple[str, ...] = ("stay", "up", "down", "left", "right")

# Row/column delta per action index. Row 0 is the top of the grid, so up is -1.
DELTAS: np.ndarray = np.array(
    [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    dtype=np.int32,
)


def _is_legal(occupancy: np.ndarray, row: int, col: int, action: int) -> bool:
    """Whether the module at (row, col) may take ``action`` from this occupancy.

    Mutates ``occupancy`` temporarily and restores it before returning, so the
    caller sees no change.
    """
    if action == STAY:
        return True

    dr, dc = DELTAS[action]
    dest_row, dest_col = row + int(dr), col + int(dc)

    height, width = occupancy.shape
    if not (0 <= dest_row < height and 0 <= dest_col < width):
        return False
    if occupancy[dest_row, dest_col]:
        return False

    occupancy[row, col] = False
    occupancy[dest_row, dest_col] = True
    connected = is_connected(occupancy)
    occupancy[row, col] = True
    occupancy[dest_row, dest_col] = False

    return connected


def is_legal(state: SwarmState, module: int, action: int) -> bool:
    """Whether ``module`` may take ``action`` from ``state``, considered alone."""
    row, col = (int(v) for v in state.positions[module])
    return _is_legal(state.occupancy, row, col, action)


def legal_action_mask(state: SwarmState) -> np.ndarray:
    """(N, 5) bool mask of legal actions, each considered in isolation.

    Column 0 (stay) is always True, so no row is ever all-False and a masked
    policy always has at least one action to sample.
    """
    mask = np.zeros((state.n_modules, N_ACTIONS), dtype=bool)
    mask[:, STAY] = True

    occupancy = state.occupancy
    for module in range(state.n_modules):
        row, col = int(state.positions[module, 0]), int(state.positions[module, 1])
        for action in (UP, DOWN, LEFT, RIGHT):
            mask[module, action] = _is_legal(occupancy, row, col, action)

    return mask


def apply_actions(
    state: SwarmState,
    actions: np.ndarray,
    rng: np.random.Generator,
    validate: bool = False,
) -> tuple[SwarmState, dict[str, Any]]:
    """Apply one simultaneous action per module, resolving conflicts by random order.

    Returns a new state (the input is not mutated) and an info dict describing
    what actually happened.
    """
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if actions.shape[0] != state.n_modules:
        raise ValueError(
            f"expected {state.n_modules} actions, got {actions.shape[0]}"
        )
    if actions.size and (actions.min() < 0 or actions.max() >= N_ACTIONS):
        raise ValueError(f"actions must be in [0, {N_ACTIONS}), got {actions}")

    new_state = state.copy()
    occupancy = new_state.occupancy
    positions = new_state.positions

    order = rng.permutation(state.n_modules)
    applied = np.zeros(state.n_modules, dtype=bool)

    for module in order:
        action = int(actions[module])
        if action == STAY:
            continue

        row, col = int(positions[module, 0]), int(positions[module, 1])
        if not _is_legal(occupancy, row, col, action):
            continue

        dr, dc = DELTAS[action]
        dest_row, dest_col = row + int(dr), col + int(dc)
        occupancy[row, col] = False
        occupancy[dest_row, dest_col] = True
        positions[module, 0] = dest_row
        positions[module, 1] = dest_col
        applied[module] = True

    new_state.step = state.step + 1

    if validate:
        new_state.validate()
        if not is_connected(new_state.occupancy):
            raise AssertionError("swarm became disconnected")

    n_attempted = int((actions != STAY).sum())
    n_applied = int(applied.sum())
    info: dict[str, Any] = {
        "applied": applied,
        "order": order,
        "n_attempted": n_attempted,
        "n_applied": n_applied,
        "n_rejected": n_attempted - n_applied,
        "rejection_rate": (n_attempted - n_applied) / n_attempted if n_attempted else 0.0,
    }
    return new_state, info


def sample_legal_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick one legal action per module, uniformly over that module's legal set.

    Used by the random-legal baseline and by the random-walk target generator.
    """
    weights = mask.astype(np.float64)
    cumulative = weights.cumsum(axis=1)
    draws = rng.random(mask.shape[0]) * cumulative[:, -1]
    return (cumulative < draws[:, None]).sum(axis=1).astype(np.int64)
