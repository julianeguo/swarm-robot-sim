import numpy as np
import pytest

from swarm.sim.connectivity import is_connected
from swarm.sim.shapes import (
    SHAPE_LIBRARY,
    library_shape,
    place,
    random_connected_shape,
)


def occupancy_of(positions, grid_shape):
    occupancy = np.zeros(grid_shape, dtype=bool)
    occupancy[positions[:, 0], positions[:, 1]] = True
    return occupancy


def test_shape_has_exactly_n_distinct_cells():
    rng = np.random.default_rng(0)
    for n_modules in (1, 2, 5, 17, 40):
        positions = random_connected_shape(rng, n_modules, (12, 12))
        assert positions.shape == (n_modules, 2)
        assert np.unique(positions, axis=0).shape[0] == n_modules
        assert positions.dtype == np.int32


def test_shape_is_connected_and_in_bounds():
    rng = np.random.default_rng(1)
    grid_shape = (10, 10)
    for _ in range(200):
        n_modules = int(rng.integers(1, 25))
        positions = random_connected_shape(rng, n_modules, grid_shape)
        assert positions.min() >= 0
        assert positions[:, 0].max() < grid_shape[0]
        assert positions[:, 1].max() < grid_shape[1]
        assert is_connected(occupancy_of(positions, grid_shape))


def test_shape_is_reproducible_for_a_given_seed():
    first = random_connected_shape(np.random.default_rng(42), 12, (10, 10))
    second = random_connected_shape(np.random.default_rng(42), 12, (10, 10))
    assert np.array_equal(first, second)


def test_different_seeds_give_different_shapes():
    first = random_connected_shape(np.random.default_rng(0), 12, (20, 20))
    second = random_connected_shape(np.random.default_rng(1), 12, (20, 20))
    assert not np.array_equal(first, second)


def test_origin_is_included_in_the_shape():
    positions = random_connected_shape(np.random.default_rng(0), 8, (10, 10), origin=(4, 5))
    assert any(tuple(cell) == (4, 5) for cell in positions)


def test_shape_fills_a_grid_exactly():
    positions = random_connected_shape(np.random.default_rng(0), 9, (3, 3))
    assert np.unique(positions, axis=0).shape[0] == 9


@pytest.mark.parametrize("n_modules", [0, -1])
def test_rejects_non_positive_size(n_modules):
    with pytest.raises(ValueError, match="n_modules must be >= 1"):
        random_connected_shape(np.random.default_rng(0), n_modules, (5, 5))


def test_rejects_size_larger_than_grid():
    with pytest.raises(ValueError, match="exceeds grid cells"):
        random_connected_shape(np.random.default_rng(0), 10, (3, 3))


def test_rejects_origin_outside_grid():
    with pytest.raises(ValueError, match="outside grid"):
        random_connected_shape(np.random.default_rng(0), 3, (5, 5), origin=(9, 9))


# -- the fixed library -------------------------------------------------


def test_every_library_shape_is_connected_and_exact_size():
    """One pattern per name has to work at every swarm size.

    Each generator builds its cells in a connected order so a prefix is still
    connected, which is what makes a single pattern usable for any n.
    """
    for name in SHAPE_LIBRARY:
        for n_modules in range(1, 30):
            cells = library_shape(name, n_modules, (40, 40))
            assert cells.shape == (n_modules, 2), f"{name} at n={n_modules}"
            assert np.unique(cells, axis=0).shape[0] == n_modules, f"{name} at n={n_modules}"
            assert is_connected(occupancy_of(cells, (40, 40))), f"{name} at n={n_modules}"


def test_library_shapes_are_deterministic():
    first = library_shape("l_shape", 12, (20, 20))
    second = library_shape("l_shape", 12, (20, 20))
    assert np.array_equal(first, second)


def test_library_shapes_are_centred():
    cells = library_shape("line_h", 4, (8, 8))
    assert set(map(tuple, cells)) == {(3, 2), (3, 3), (3, 4), (3, 5)}


def test_line_and_square_are_the_rigid_cases():
    # Pinned because these are the shapes that cannot move at all under the
    # current rules, which is what makes them useful in evaluation.
    assert set(map(tuple, library_shape("line_h", 3, (9, 9)))) == {(4, 3), (4, 4), (4, 5)}
    assert set(map(tuple, library_shape("square", 4, (8, 8)))) == {(3, 3), (3, 4), (4, 3), (4, 4)}


def test_unknown_library_shape_raises():
    with pytest.raises(KeyError, match="unknown shape"):
        library_shape("spiral", 5, (10, 10))


def test_library_shape_rejects_bad_size():
    with pytest.raises(ValueError, match="n_modules must be >= 1"):
        library_shape("line_h", 0, (10, 10))


# -- placement ---------------------------------------------------------


def test_place_centres_by_default():
    cells = place(np.array([(0, 0), (0, 1)], dtype=np.int32), (6, 6))
    assert set(map(tuple, cells)) == {(2, 2), (2, 3)}


def test_place_honours_an_anchor():
    cells = place(np.array([(5, 5), (5, 6)], dtype=np.int32), (10, 10), anchor=(0, 0))
    assert set(map(tuple, cells)) == {(0, 0), (0, 1)}


def test_place_preserves_shape_and_connectivity():
    original = library_shape("staircase", 9, (20, 20))
    moved = place(original, (20, 20), anchor=(1, 1))
    assert is_connected(occupancy_of(moved, (20, 20)))
    # Relative geometry is unchanged.
    assert np.array_equal(original - original.min(axis=0), moved - moved.min(axis=0))


def test_place_rejects_a_shape_that_cannot_fit():
    with pytest.raises(ValueError, match="does not fit"):
        place(library_shape("line_h", 10, (20, 20)), (4, 4))


def test_place_rejects_an_out_of_bounds_anchor():
    with pytest.raises(ValueError, match="out of bounds"):
        place(np.array([(0, 0), (0, 1)], dtype=np.int32), (6, 6), anchor=(5, 5))
