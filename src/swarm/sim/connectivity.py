"""4-neighbour connectivity over an occupancy grid.

Everything here is the brute-force reference implementation: a flood fill per
query, O(number of occupied cells). The articulation-point fast path arrives in
a later milestone and will be checked for exact agreement against these
functions, so they stay in the tree as the oracle.
"""

from __future__ import annotations

import numpy as np

# (row, col) offsets of the four orthogonal neighbours.
NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def is_connected(occupancy: np.ndarray) -> bool:
    """True if all occupied cells form a single 4-connected group.

    An empty grid counts as connected: it is the degenerate case produced when
    the only module in a one-module swarm is lifted out, and treating it as
    disconnected would make every move illegal at N=1.
    """
    flat = occupancy.reshape(-1)
    total = int(flat.sum())
    if total <= 1:
        return True

    height, width = occupancy.shape
    start = int(np.argmax(flat))
    seen = np.zeros(flat.shape, dtype=bool)
    seen[start] = True
    stack = [start]
    found = 1

    while stack:
        index = stack.pop()
        row, col = divmod(index, width)
        if row > 0:
            n = index - width
            if flat[n] and not seen[n]:
                seen[n] = True
                found += 1
                stack.append(n)
        if row < height - 1:
            n = index + width
            if flat[n] and not seen[n]:
                seen[n] = True
                found += 1
                stack.append(n)
        if col > 0:
            n = index - 1
            if flat[n] and not seen[n]:
                seen[n] = True
                found += 1
                stack.append(n)
        if col < width - 1:
            n = index + 1
            if flat[n] and not seen[n]:
                seen[n] = True
                found += 1
                stack.append(n)

    return found == total


def component_labels(occupancy: np.ndarray) -> np.ndarray:
    """Label each occupied cell with its component index; empty cells get -1.

    Used by tests and debugging output, not on the hot path.
    """
    height, width = occupancy.shape
    labels = np.full(occupancy.shape, -1, dtype=np.int32)
    next_label = 0

    for row in range(height):
        for col in range(width):
            if not occupancy[row, col] or labels[row, col] != -1:
                continue
            labels[row, col] = next_label
            stack = [(row, col)]
            while stack:
                r, c = stack.pop()
                for dr, dc in NEIGHBOUR_OFFSETS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if occupancy[nr, nc] and labels[nr, nc] == -1:
                            labels[nr, nc] = next_label
                            stack.append((nr, nc))
            next_label += 1

    return labels


def n_components(occupancy: np.ndarray) -> int:
    """Number of 4-connected groups of occupied cells (0 for an empty grid)."""
    labels = component_labels(occupancy)
    return int(labels.max()) + 1 if labels.max() >= 0 else 0


def has_occupied_neighbour(occupancy: np.ndarray, row: int, col: int) -> bool:
    """True if any of the four orthogonal neighbours of (row, col) is occupied."""
    height, width = occupancy.shape
    for dr, dc in NEIGHBOUR_OFFSETS:
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width and occupancy[nr, nc]:
            return True
    return False
