"""Connected shape generation.

This milestone provides the growth sampler only -- enough to produce a random
connected start shape. The random-walk target generator (target = start shape
after K legal moves, which is what makes an episode provably solvable) and the
fixed shape library for held-out evaluation arrive with the episode module.
"""

from __future__ import annotations

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
