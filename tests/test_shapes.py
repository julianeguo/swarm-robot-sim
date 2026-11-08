import numpy as np
import pytest

from swarm.sim.connectivity import is_connected
from swarm.sim.shapes import random_connected_shape


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
