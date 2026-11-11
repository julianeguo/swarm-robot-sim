# Checks for the lattice geometry in hexgrid.py. Everything else (sim, shapes, env,
# render) builds on these, so a mistake here would break things silently downstream.

import numpy as np
import pytest

from swarm import hexgrid as h


def _rotate_ccw(vec, times):
    """Rotate a lattice vector by 60° CCW `times` times: (i, j) -> (-j, i + j)."""
    i, j = vec
    for _ in range(times % 6):
        i, j = -j, i + j
    return np.array([i, j])


def test_dirs_are_unit_steps_at_60_degree_spacing():
    xy = h.to_xy(h.DIRS)
    assert np.allclose(np.linalg.norm(xy, axis=1), 1.0)
    angles = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 360
    assert np.allclose(angles, [0, 60, 120, 180, 240, 300])
    assert (h.DIRS.sum(axis=0) == 0).all()


def test_next_direction_is_a_60_degree_ccw_turn():
    for k in range(6):
        assert (_rotate_ccw(h.DIRS[k], 1) == h.DIRS[(k + 1) % 6]).all()
        assert (_rotate_ccw(h.DIRS[k], 6) == h.DIRS[k]).all()


def test_direction_index_inverts_dirs():
    for k in range(6):
        assert h.direction_index(h.DIRS[k]) == k


def test_neighbors_shape_and_distance():
    pos = np.array([[0, 0], [3, -2], [-1, 4]])
    nbs = h.neighbors(pos)
    assert nbs.shape == (3, 6, 2)
    assert (h.hex_distance(nbs - pos[:, None, :]) == 1).all()


@pytest.mark.parametrize("k", range(6))
@pytest.mark.parametrize("rotation", [h.CCW, h.CW])
def test_pivot_destination_matches_real_rotation(k, rotation):
    # Rotate the module's offset from the pivot explicitly and compare.
    p = np.array([3, -2])
    q = p + h.DIRS[k]
    expected = q + _rotate_ccw(p - q, rotation)
    got = h.pivot_destination(p, k, rotation)
    assert (got == expected).all()
    # The landing cell is next to both the module's old cell and the pivot.
    assert h.hex_distance(got - p) == 1
    assert h.hex_distance(got - q) == 1


def test_hex_distance_examples():
    assert h.hex_distance((0, 0)) == 0
    assert h.hex_distance((1, 1)) == 2  # a + b: the 30° direction, two steps away
    assert h.hex_distance((2, -1)) == 2
    assert h.hex_distance((3, 0)) == 3


@pytest.mark.parametrize("radius", range(5))
def test_window_offsets_count_and_order(radius):
    offs = h.window_offsets(radius)
    assert len(offs) == 3 * radius * (radius + 1) + 1
    assert (h.hex_distance(offs) <= radius).all()
    assert (offs == h.window_offsets(radius)).all()  # same order every call


def test_is_connected():
    assert h.is_connected(np.zeros((0, 2), dtype=int))
    assert h.is_connected([[0, 0], [1, 0], [0, 1]])
    assert not h.is_connected([[0, 0], [1, 1]])  # 2 steps apart: not adjacent


def test_board_indexing_round_trip_and_coverage():
    m = 4
    p = np.array([3, -2])
    assert (h.from_index(h.to_index(p, m), m) == p).all()
    # The board covers every cell within hex distance M of the origin.
    assert h.in_bounds(h.window_offsets(m), m).all()
    assert not h.in_bounds(np.array([m + 1, 0]), m)


def test_cells_to_mask():
    m = 3
    cells = np.array([[0, 0], [1, -1], [-3, 2]])
    mask = h.cells_to_mask(cells, m)
    assert mask.shape == (7, 7) and mask.dtype == bool
    assert mask.sum() == 3
    for c in cells:
        r, col = h.to_index(c, m)
        assert mask[r, col]
    with pytest.raises(AssertionError):
        h.cells_to_mask([[m + 1, 0]], m)
