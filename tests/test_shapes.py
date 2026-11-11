# Checks for shapes.py: every shape has exactly N cells, is centred, is one connected
# piece, and comes out the same every time. sim.py relies on all four.

import numpy as np
import pytest

from swarm import hexgrid
from swarm.shapes import SHAPE_NAMES, make_shape, overlap_fraction, shape_radius

SIZES = [1, 7, 20, 50, 100]


@pytest.mark.parametrize("name", SHAPE_NAMES)
@pytest.mark.parametrize("n", SIZES)
def test_exactly_n_unique_cells(name, n):
    cells = make_shape(name, n)
    assert cells.shape == (n, 2)
    assert len({tuple(c) for c in cells.tolist()}) == n


@pytest.mark.parametrize("name", SHAPE_NAMES)
@pytest.mark.parametrize("n", SIZES)
def test_contains_origin_and_is_connected(name, n):
    cells = make_shape(name, n)
    assert (0, 0) in {tuple(c) for c in cells.tolist()}
    assert hexgrid.is_connected(cells)


@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_roughly_centred(name):
    # Tie-breaking can tilt a shape by a cell or two, but the centre of mass stays near 0.
    centre = hexgrid.to_xy(make_shape(name, 50)).mean(axis=0)
    assert np.hypot(*centre) < 1.0


def test_cached_and_read_only():
    a = make_shape("star", 50)
    assert make_shape("star", 50) is a
    with pytest.raises(ValueError):
        a[0, 0] = 99


def test_params_change_shape_and_keyword_order_does_not():
    a = make_shape("star", 50, points=5, inner_ratio=0.5)
    b = make_shape("star", 50, inner_ratio=0.5, points=5)
    assert a is b
    # Lattice rows are discrete, so a small change (w=2 -> 3) can give the same 5-row shape.
    # w=5 is different enough to drop to 3 rows.
    assert not np.array_equal(make_shape("rectangle", 50, w=5.0), make_shape("rectangle", 50))


def test_rectangle_is_wider_than_tall():
    xy = hexgrid.to_xy(make_shape("rectangle", 50))
    assert np.ptp(xy[:, 0]) > np.ptp(xy[:, 1])


def test_shapes_differ_at_n50():
    # If two shapes were identical at N=50, an episode between them would be trivial.
    for i, a in enumerate(SHAPE_NAMES):
        for b in SHAPE_NAMES[i + 1 :]:
            assert overlap_fraction(make_shape(a, 50), make_shape(b, 50)) < 1.0


def test_overlap_and_radius_helpers():
    cells = make_shape("circle", 7)  # centre + its 6 neighbours
    assert overlap_fraction(cells, cells) == 1.0
    assert shape_radius(cells) == 1


@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_connected_at_every_n(name):
    # Before the grow-from-the-origin fix, the star broke at N=6, 20, 45-47, 81-83, ...
    for n in range(1, 151):
        assert hexgrid.is_connected(make_shape(name, n))


def test_unknown_name_and_bad_n():
    with pytest.raises(ValueError):
        make_shape("triangle", 50)
    with pytest.raises(ValueError):
        make_shape("circle", 0)
