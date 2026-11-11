# Hexagonal lattice geometry (pointy-top).
#
# Unit steps are: a (0°), b (+60°), c (−60°), where c = a − b.
# Every position is written as (i, j), meaning i steps along a plus j steps along b.
# Direction c still exists for moving: it's just "+1 along a, −1 along b", i.e. (1, −1).
#
# This file is pure geometry: no state, no movement rules, no RL. sim.py, shapes.py,
# env.py and render.py all import from here so they agree on what the lattice looks like.
# All functions accept NumPy arrays whose last axis is (i, j), so they work on one
# position or the whole swarm at once.

from collections import deque

import numpy as np

# The six neighbour directions in counterclockwise order: a, b, −c, −a, −b, c.
# Index k into this table is "direction k". Stepping +1 through the list is a 60° CCW turn.
DIRS = np.array(
    [
        [1, 0],  # 0: a   (0°)
        [0, 1],  # 1: b   (60°)
        [-1, 1],  # 2: −c  (120°)
        [-1, 0],  # 3: −a  (180°)
        [0, -1],  # 4: −b  (240°)
        [1, -1],  # 5: c   (300°)
    ],
    dtype=np.int64,
)

# Rotation sense for a pivot move. sim.py and env.py use these to encode actions.
CCW = 1
CW = -1

# Reverse lookup: a unit step like (1, -1) -> its direction index (5).
_DIR_INDEX = {tuple(d): k for k, d in enumerate(DIRS.tolist())}


def direction_index(step):
    """Return the direction index k (0-5) of a single unit step (i, j).

    Used when you know two cells are adjacent and need to know *which* neighbour one is,
    e.g. turning "module at p, pivot at q" into k for pivot_destination.
    """
    return _DIR_INDEX[(int(step[0]), int(step[1]))]


def neighbors(pos):
    """Return the 6 neighbour cells of each position, in DIRS order.

    Shape (..., 2) -> (..., 6, 2). sim.py uses this to find which neighbours are occupied
    (= possible pivots). env.py uses it to build the action mask.
    """
    pos = np.asarray(pos, dtype=np.int64)
    return pos[..., None, :] + DIRS


def pivot_destination(pos, k, rotation):
    """Where a module lands after rotating 60° around its neighbour in direction k.

    A module at p with its pivot at p + DIRS[k] always moves exactly one step:
      CCW (rotation=+1) -> p + DIRS[k - 1]
      CW  (rotation=-1) -> p + DIRS[k + 1]
    The landing cell is next to both the module and the pivot.

    This function only answers "where would it land". The rule for whether the move is
    allowed (destination empty) lives in sim.py's single movement-rule function.
    pos is (..., 2); k and rotation broadcast against pos[..., 0].
    """
    pos = np.asarray(pos, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    rotation = np.asarray(rotation, dtype=np.int64)
    return pos + DIRS[(k - rotation) % 6]


def hex_distance(delta):
    """Number of single steps needed to cover the offset delta = (i, j).

    Pass the difference of two positions to get the distance between them.
    Used for the observation window, the shape extents, and board sizing.
    """
    delta = np.asarray(delta, dtype=np.int64)
    i, j = delta[..., 0], delta[..., 1]
    return (np.abs(i) + np.abs(j) + np.abs(i + j)) // 2


def to_xy(pos):
    """Convert lattice (i, j) to Cartesian (x, y) with unit spacing between neighbours.

    x = i + j/2, y = j·√3/2. shapes.py uses this to score cells against a continuous
    shape. render.py uses it to draw the cells.
    """
    pos = np.asarray(pos, dtype=np.float64)
    i, j = pos[..., 0], pos[..., 1]
    return np.stack([i + 0.5 * j, (np.sqrt(3) / 2) * j], axis=-1)


def window_offsets(radius):
    """All offsets within hex distance <= radius of a cell, in one fixed order.

    Returns a (3R(R+1)+1, 2) array (19 offsets for R=2). env.py adds these to each
    module's position to read its local window. Because the order is always the same,
    observation entry n always means the same relative cell for every module and step.
    """
    offsets = [
        (i, j)
        for j in range(-radius, radius + 1)
        for i in range(-radius, radius + 1)
        if hex_distance((i, j)) <= radius
    ]
    return np.array(offsets, dtype=np.int64)


def is_connected(cells):
    """True if the given cells form one connected group (adjacent = one step apart).

    cells is (N, 2). Used to log "is the swarm connected at the end" (logged, never
    rewarded) and to check that shapes from shapes.py come out as one piece.
    An empty set counts as connected.
    """
    cell_set = {tuple(c) for c in np.asarray(cells, dtype=np.int64).tolist()}
    if not cell_set:
        return True
    start = next(iter(cell_set))
    seen = {start}
    queue = deque([start])
    while queue:
        i, j = queue.popleft()
        for di, dj in DIRS.tolist():
            nb = (i + di, j + dj)
            if nb in cell_set and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen) == len(cell_set)


# --- Board: mapping lattice positions onto sim.py's dense occupancy array ---
#
# sim.py stores occupancy as a (2M+1, 2M+1) boolean array, with the origin in the middle.
# Lattice cell (i, j) lives at array index (i + M, j + M). That square of indices covers
# every cell within hex distance M of the origin.
#
# Of the 8 array entries around a cell, only 6 are hex neighbours (the DIRS offsets):
# up, down, left, right, and the diagonals (1, -1) and (-1, 1). The other two diagonals,
# (1, 1) and (-1, -1), are 2 steps away. Because b points at 60°, the square array is really
# a slanted parallelogram on the lattice, so to draw it, convert with to_xy rather than
# showing the raw array as an image.
#
# Suggested state in sim.py (all kept in sync on every move):
#   occupied : (2M+1, 2M+1) bool  -> "is this cell empty?" in one lookup
#   positions: (N, 2) int (i, j)  -> "where is module n?"
#   target   : (2M+1, 2M+1) bool  -> target shape cells, read by the observation window


def board_radius(shape_radius, episode_length, window_radius):
    """Choose M so that no module can ever reach the board edge.

    Each module moves at most one step per env step, so after T steps no module is
    further than shape_radius + T from the origin. Adding window_radius means the
    observation window never reads past the edge either. This keeps "a move is valid
    if the destination is empty" as the only movement rule (no edge rule needed).
    """
    return int(shape_radius) + int(episode_length) + int(window_radius)


def to_index(pos, m):
    """Lattice (i, j) -> array index (row, col) = (i + M, j + M) into the board."""
    return np.asarray(pos, dtype=np.int64) + m


def from_index(idx, m):
    """Array index (row, col) -> lattice (i, j). Inverse of to_index."""
    return np.asarray(idx, dtype=np.int64) - m


def cells_to_mask(cells, m):
    """Turn a list of cells (N, 2) into a (2M+1, 2M+1) bool board mask, True on those cells.

    This is how sim.py builds its board arrays from shapes.make_shape output:
    target = cells_to_mask(target_cells, M), occupied = cells_to_mask(start_cells, M).
    Shapes come as cell lists because the board size (M) isn't known until the env is set
    up. This helper does the conversion once M is known.
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    assert in_bounds(cells, m).all(), "cells fall outside the board; is M too small?"
    mask = np.zeros((2 * m + 1, 2 * m + 1), dtype=bool)
    idx = to_index(cells, m)
    mask[idx[:, 0], idx[:, 1]] = True
    return mask


def in_bounds(pos, m):
    """True where position (i, j) fits on a board of radius M.

    With board_radius sized correctly this should never be False during an episode.
    sim.py can assert on it to catch a sizing bug.
    """
    pos = np.asarray(pos, dtype=np.int64)
    return np.all(np.abs(pos) <= m, axis=-1)
