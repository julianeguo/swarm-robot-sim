import numpy as np
import pytest

from swarm.sim.state import StateError, SwarmState

from .helpers import state_from_cells

GRID = (6, 6)


def test_from_positions_derives_occupancy():
    state = state_from_cells([(1, 1), (1, 2), (2, 2)], GRID)
    assert state.n_modules == 3
    assert int(state.occupancy.sum()) == 3
    assert state.occupancy[1, 1] and state.occupancy[1, 2] and state.occupancy[2, 2]


def test_target_defaults_to_empty():
    state = state_from_cells([(1, 1)], GRID)
    assert state.n_targets == 0
    assert state.is_complete, "a swarm with no target cells is trivially complete"
    assert state.fill_fraction == 1.0


def test_target_progress_accounting():
    state = state_from_cells([(1, 1), (1, 2)], GRID, target=[(1, 2), (3, 3)])
    assert state.n_targets == 2
    assert state.n_filled == 1
    assert state.fill_fraction == 0.5
    assert not state.is_complete
    assert state.unfilled_mask[3, 3]
    assert not state.unfilled_mask[1, 2]


def test_is_complete_when_every_target_is_occupied():
    state = state_from_cells([(1, 1), (1, 2)], GRID, target=[(1, 1), (1, 2)])
    assert state.is_complete
    assert state.fill_fraction == 1.0


def test_on_target_is_per_module():
    state = state_from_cells([(1, 1), (1, 2)], GRID, target=[(1, 2)])
    assert list(state.on_target()) == [False, True]


def test_copy_is_independent():
    state = state_from_cells([(1, 1)], GRID)
    clone = state.copy()
    clone.occupancy[0, 0] = True
    clone.positions[0, 0] = 5
    clone.step = 99
    assert not state.occupancy[0, 0]
    assert state.positions[0, 0] == 1
    assert state.step == 0


def test_validate_catches_grid_position_disagreement():
    state = state_from_cells([(1, 1), (1, 2)], GRID)
    state.occupancy[1, 1] = False
    state.occupancy[4, 4] = True
    with pytest.raises(StateError, match="disagrees with positions"):
        state.validate()


def test_validate_catches_duplicate_modules():
    positions = np.array([[1, 1], [1, 1]], dtype=np.int32)
    with pytest.raises(StateError, match="same cell"):
        SwarmState.from_positions(positions, GRID)


def test_validate_catches_out_of_bounds():
    positions = np.array([[0, 0], [9, 9]], dtype=np.int32)
    with pytest.raises(StateError, match="out of bounds"):
        SwarmState.from_positions(positions, GRID)


def test_validate_catches_module_count_mismatch():
    state = state_from_cells([(1, 1)], GRID)
    state.occupancy[4, 4] = True
    with pytest.raises(StateError, match="cells but there are"):
        state.validate()


def test_validate_catches_target_shape_mismatch():
    state = state_from_cells([(1, 1)], GRID)
    state.target = np.zeros((3, 3), dtype=bool)
    with pytest.raises(StateError, match="target shape"):
        state.validate()


def test_grid_accessors():
    state = state_from_cells([(1, 1)], (4, 7))
    assert state.grid_shape == (4, 7)
    assert state.height == 4
    assert state.width == 7
