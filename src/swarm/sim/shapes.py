"""Connected shape geometry: the random growth sampler and the fixed library.

Everything here is pure geometry and depends only on connectivity, never on the
movement rules. Target generation needs the rules to guarantee solvability and
therefore lives in :mod:`swarm.sim.episodes`.

Every generator returns an ``(N, 2)`` int32 array of distinct cells forming a
single 4-connected group, and every library shape is built in a connected order
so that taking a prefix of it is still connected -- which is what lets one
pattern serve every swarm size.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from swarm.sim.connectivity import NEIGHBOUR_OFFSETS


def random_connected_shape(
    rng: np.random.Generator,
    n_modules: int,
    grid_shape: tuple[int, int],
    origin: tuple[int, int] | None = None,
) -> np.ndarray:
    """Sample a 4-connected set of ``n_modules`` cells as an (N, 2) int32 array.

    Grows outward from a seed cell by repeatedly annexing a uniformly random
    empty cell adjacent to the shape so far. The distribution is biased towards
    blobby shapes and will essentially never produce long thin structures, so
    evaluation should not rely on it alone for shape diversity.

    ``origin`` seeds the growth at a specific cell; otherwise the seed is
    uniform over the grid.
    """
    height, width = grid_shape
    if n_modules < 1:
        raise ValueError(f"n_modules must be >= 1, got {n_modules}")
    if n_modules > height * width:
        raise ValueError(f"n_modules ({n_modules}) exceeds grid cells ({height * width})")

    if origin is None:
        seed = (int(rng.integers(height)), int(rng.integers(width)))
    else:
        seed = (int(origin[0]), int(origin[1]))
        if not (0 <= seed[0] < height and 0 <= seed[1] < width):
            raise ValueError(f"origin {origin} is outside grid {grid_shape}")

    in_shape = np.zeros(grid_shape, dtype=bool)
    in_shape[seed] = True
    cells = [seed]

    # Candidate cells on the shape's frontier, kept as a list plus a membership
    # grid so draws are uniform over distinct frontier cells.
    in_frontier = np.zeros(grid_shape, dtype=bool)
    frontier: list[tuple[int, int]] = []

    def push_neighbours(row: int, col: int) -> None:
        for dr, dc in NEIGHBOUR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                if not in_shape[nr, nc] and not in_frontier[nr, nc]:
                    in_frontier[nr, nc] = True
                    frontier.append((nr, nc))

    push_neighbours(*seed)

    while len(cells) < n_modules:
        if not frontier:
            # Unreachable for a rectangular grid, but keeps the failure explicit
            # if a bounded-region variant is added later.
            raise RuntimeError("shape growth ran out of frontier cells")
        index = int(rng.integers(len(frontier)))
        frontier[index], frontier[-1] = frontier[-1], frontier[index]
        cell = frontier.pop()
        in_frontier[cell] = False
        in_shape[cell] = True
        cells.append(cell)
        push_neighbours(*cell)

    return np.array(cells, dtype=np.int32)


def place(
    cells: np.ndarray,
    grid_shape: tuple[int, int],
    anchor: tuple[int, int] | None = None,
) -> np.ndarray:
    """Translate a shape so its bounding box starts at ``anchor``.

    With ``anchor=None`` the shape is centred in the grid. Raises if the shape
    cannot fit.
    """
    cells = np.asarray(cells, dtype=np.int32).reshape(-1, 2)
    height, width = grid_shape
    extent_rows = int(cells[:, 0].max() - cells[:, 0].min()) + 1
    extent_cols = int(cells[:, 1].max() - cells[:, 1].min()) + 1

    if extent_rows > height or extent_cols > width:
        raise ValueError(
            f"shape extent {extent_rows}x{extent_cols} does not fit in grid {grid_shape}"
        )

    if anchor is None:
        anchor = ((height - extent_rows) // 2, (width - extent_cols) // 2)
    if not (0 <= anchor[0] <= height - extent_rows and 0 <= anchor[1] <= width - extent_cols):
        raise ValueError(f"anchor {anchor} puts a {extent_rows}x{extent_cols} shape out of bounds")

    offset = np.array(
        [anchor[0] - cells[:, 0].min(), anchor[1] - cells[:, 1].min()], dtype=np.int32
    )
    return (cells + offset).astype(np.int32)


# -- the fixed library -------------------------------------------------
#
# Used for held-out evaluation. The growth sampler above produces blobs and
# essentially never produces a long thin structure, so reporting results on
# sampled shapes alone would hide exactly the cases the rules find hardest.


def line_h(n_modules: int) -> np.ndarray:
    """A horizontal bar. Rigid under translation-only movement."""
    return np.array([(0, i) for i in range(n_modules)], dtype=np.int32)


def line_v(n_modules: int) -> np.ndarray:
    """A vertical bar. Rigid under translation-only movement."""
    return np.array([(i, 0) for i in range(n_modules)], dtype=np.int32)


def square(n_modules: int) -> np.ndarray:
    """A solid block, filled row-major, as square as ``n_modules`` allows."""
    side = int(np.ceil(np.sqrt(n_modules)))
    cells = [(i // side, i % side) for i in range(n_modules)]
    return np.array(cells, dtype=np.int32)


def l_shape(n_modules: int) -> np.ndarray:
    """A right angle: a vertical arm and a horizontal arm sharing a corner."""
    down = (n_modules + 1) // 2
    cells = [(i, 0) for i in range(down)]
    cells += [(down - 1, j) for j in range(1, n_modules - down + 1)]
    return np.array(cells, dtype=np.int32)


def t_shape(n_modules: int) -> np.ndarray:
    """A horizontal bar with a stem hanging from its middle."""
    bar = max(1, (n_modules + 1) // 2)
    cells = [(0, j) for j in range(bar)]
    cells += [(i, bar // 2) for i in range(1, n_modules - bar + 1)]
    return np.array(cells, dtype=np.int32)


def plus(n_modules: int) -> np.ndarray:
    """A centre cell with four arms grown one cell at a time, round-robin."""
    cells = [(0, 0)]
    arms = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    length = 1
    while len(cells) < n_modules:
        for dr, dc in arms:
            if len(cells) == n_modules:
                break
            cells.append((dr * length, dc * length))
        length += 1
    return np.array(cells[:n_modules], dtype=np.int32)


def ring(n_modules: int) -> np.ndarray:
    """A walk around the perimeter of the smallest square that can hold it.

    For small sizes this is a prefix of the perimeter walk, i.e. an open path
    rather than a closed ring, which is still connected.
    """
    side = 2
    while 4 * side - 4 < n_modules:
        side += 1

    perimeter: list[tuple[int, int]] = []
    perimeter += [(0, j) for j in range(side)]
    perimeter += [(i, side - 1) for i in range(1, side)]
    perimeter += [(side - 1, j) for j in range(side - 2, -1, -1)]
    perimeter += [(i, 0) for i in range(side - 2, 0, -1)]
    return np.array(perimeter[:n_modules], dtype=np.int32)


def staircase(n_modules: int) -> np.ndarray:
    """A diagonal staircase, alternating one cell right and one cell down."""
    cells = [(0, 0)]
    row = col = 0
    while len(cells) < n_modules:
        col += 1
        cells.append((row, col))
        if len(cells) == n_modules:
            break
        row += 1
        cells.append((row, col))
    return np.array(cells[:n_modules], dtype=np.int32)


SHAPE_LIBRARY: dict[str, Callable[[int], np.ndarray]] = {
    "line_h": line_h,
    "line_v": line_v,
    "square": square,
    "l_shape": l_shape,
    "t_shape": t_shape,
    "plus": plus,
    "ring": ring,
    "staircase": staircase,
}


def library_shape(name: str, n_modules: int, grid_shape: tuple[int, int]) -> np.ndarray:
    """Build a named library shape of ``n_modules`` cells, centred in the grid."""
    if name not in SHAPE_LIBRARY:
        raise KeyError(f"unknown shape '{name}'; known shapes: {sorted(SHAPE_LIBRARY)}")
    if n_modules < 1:
        raise ValueError(f"n_modules must be >= 1, got {n_modules}")
    return place(SHAPE_LIBRARY[name](n_modules), grid_shape)
