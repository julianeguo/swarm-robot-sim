import numpy as np
import pytest

from swarm.sim.connectivity import (
    has_occupied_neighbour,
    is_connected,
    n_components,
)
from swarm.sim.shapes import random_connected_shape

from .helpers import grid_from_rows


def test_empty_grid_is_connected():
    # Degenerate case: lifting the only module out of a one-module swarm leaves
    # an empty grid, and calling that disconnected would freeze N=1 entirely.
    assert is_connected(np.zeros((4, 4), dtype=bool))


def test_single_cell_is_connected():
    assert is_connected(grid_from_rows(["...", ".#.", "..."]))


def test_line_is_connected():
    assert is_connected(grid_from_rows(["###", "...", "..."]))


def test_disjoint_cells_are_not_connected():
    assert not is_connected(grid_from_rows(["#..", "...", "..#"]))


def test_diagonal_adjacency_does_not_count():
    assert not is_connected(grid_from_rows(["#..", ".#.", "..."]))


def test_u_shape_is_connected():
    assert is_connected(grid_from_rows(["#.#", "#.#", "###"]))


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([".....", "....."], 0),
        (["#....", "....."], 1),
        (["#..#.", "....."], 2),
        (["##.##", "##.##"], 2),
        (["#.#.#", "....."], 3),
    ],
)
def test_component_count(rows, expected):
    assert n_components(grid_from_rows(rows)) == expected


def test_has_occupied_neighbour():
    occupancy = grid_from_rows(["#..", "...", "..#"])
    assert has_occupied_neighbour(occupancy, 0, 1)  # (0,0) is occupied
    assert has_occupied_neighbour(occupancy, 1, 0)  # (0,0) is occupied
    assert has_occupied_neighbour(occupancy, 1, 2)  # (2,2) is occupied
    assert not has_occupied_neighbour(occupancy, 1, 1)  # both cells are diagonal only
    assert not has_occupied_neighbour(occupancy, 2, 0)


def test_sampled_shapes_are_always_connected():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n_modules = int(rng.integers(1, 20))
        grid_shape = (int(rng.integers(5, 12)), int(rng.integers(5, 12)))
        positions = random_connected_shape(rng, n_modules, grid_shape)
        occupancy = np.zeros(grid_shape, dtype=bool)
        occupancy[positions[:, 0], positions[:, 1]] = True
        assert is_connected(occupancy)
        assert n_components(occupancy) == 1
