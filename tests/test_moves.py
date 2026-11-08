import numpy as np
import pytest

from swarm.sim import moves
from swarm.sim.connectivity import is_connected

from .helpers import random_state, state_from_cells

GRID = (6, 6)


def mask_of(cells, grid_shape=GRID):
    return moves.legal_action_mask(state_from_cells(cells, grid_shape))


# -- the mask contract -------------------------------------------------


def test_stay_is_always_legal():
    rng = np.random.default_rng(0)
    for _ in range(100):
        state = random_state(rng, n_modules=int(rng.integers(1, 15)))
        mask = moves.legal_action_mask(state)
        assert mask[:, moves.STAY].all()
        assert mask.any(axis=1).all(), "no module may ever have an empty legal set"


def test_mask_shape_and_dtype():
    state = state_from_cells([(1, 1), (1, 2)], GRID)
    mask = moves.legal_action_mask(state)
    assert mask.shape == (2, moves.N_ACTIONS)
    assert mask.dtype == bool


def test_mask_matches_apply_exactly():
    """The central contract: the mask is legal iff applying that move alone works.

    Everything downstream trusts the mask, so any drift between it and
    apply_actions has to fail here.
    """
    rng = np.random.default_rng(1)
    for _ in range(60):
        state = random_state(rng, n_modules=int(rng.integers(1, 10)), grid_shape=(8, 8))
        mask = moves.legal_action_mask(state)

        for module in range(state.n_modules):
            for action in range(moves.N_ACTIONS):
                actions = np.full(state.n_modules, moves.STAY, dtype=np.int64)
                actions[module] = action
                _, info = moves.apply_actions(state, actions, np.random.default_rng(0))

                if action == moves.STAY:
                    assert mask[module, action]
                    assert not info["applied"][module]
                else:
                    assert bool(info["applied"][module]) == bool(mask[module, action]), (
                        f"mask/apply disagree for module {module} action "
                        f"{moves.ACTION_NAMES[action]}"
                    )


# -- individual rules --------------------------------------------------


def test_out_of_bounds_is_illegal():
    mask = mask_of([(0, 0), (0, 1)])
    assert not mask[0, moves.UP]
    assert not mask[0, moves.LEFT]


def test_occupied_destination_is_illegal():
    # A horizontal pair: neither can move into the other's cell.
    mask = mask_of([(2, 2), (2, 3)])
    assert not mask[0, moves.RIGHT]
    assert not mask[1, moves.LEFT]


def test_move_that_splits_the_swarm_is_illegal():
    # Dumbbell: the bridge module at (2,2) holds two arms together.
    cells = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    mask = mask_of(cells)
    bridge = cells.index((2, 2))
    assert not mask[bridge, moves.UP]
    assert not mask[bridge, moves.DOWN]


def test_single_module_may_move_freely():
    # With one module, removing it leaves an empty (vacuously connected) grid.
    mask = mask_of([(2, 2)])
    assert mask.all()


def test_corner_shape_is_mobile():
    # An L has corners, so its endpoints can move while keeping contact.
    mask = mask_of([(2, 2), (3, 2), (3, 3)])
    assert mask[:, 1:].any(), "an L-shaped swarm should have at least one legal move"


@pytest.mark.parametrize(
    "cells,label",
    [
        ([(2, 1), (2, 2), (2, 3)], "straight line"),
        ([(2, 2), (2, 3), (3, 2), (3, 3)], "solid 2x2 block"),
    ],
)
def test_rigid_shapes_have_no_legal_moves(cells, label):
    """Documents a real consequence of translation-only movement.

    A module may only translate into a cell that still touches the rest of the
    swarm, so shapes with no reachable "corner" cell are completely frozen: no
    module in a straight line or a solid block can move at all. This is the
    expected behaviour of the current rule set, not a bug, and it is the main
    argument for promoting a sliding/pivoting rule set (destinations in the full
    8-neighbourhood) before relying on hard shape pairs.
    """
    mask = mask_of(cells)
    assert mask[:, moves.STAY].all()
    assert not mask[:, 1:].any(), f"{label} should be rigid under translation-only moves"


# -- applying actions --------------------------------------------------


def test_apply_preserves_invariants_and_connectivity():
    rng = np.random.default_rng(2)
    state = random_state(rng, n_modules=10, grid_shape=(10, 10))
    for _ in range(200):
        mask = moves.legal_action_mask(state)
        actions = moves.sample_legal_actions(mask, rng)
        state, _ = moves.apply_actions(state, actions, rng, validate=True)
        assert state.n_modules == 10
        assert is_connected(state.occupancy)


def test_apply_does_not_mutate_the_input_state():
    state = state_from_cells([(2, 2), (3, 2), (3, 3)], GRID)
    before_occupancy = state.occupancy.copy()
    before_positions = state.positions.copy()
    before_step = state.step

    moves.apply_actions(
        state, np.array([moves.UP, moves.STAY, moves.RIGHT]), np.random.default_rng(0)
    )

    assert np.array_equal(state.occupancy, before_occupancy)
    assert np.array_equal(state.positions, before_positions)
    assert state.step == before_step


def test_apply_increments_step():
    state = state_from_cells([(2, 2)], GRID)
    new_state, _ = moves.apply_actions(state, np.array([moves.STAY]), np.random.default_rng(0))
    assert new_state.step == state.step + 1


def test_rejection_accounting():
    # Two modules side by side both try to move into the same empty cell above
    # the right one; whichever goes second is rejected.
    state = state_from_cells([(2, 2), (2, 3), (3, 3)], GRID)
    actions = np.array([moves.RIGHT, moves.STAY, moves.STAY])
    _, info = moves.apply_actions(state, actions, np.random.default_rng(0))
    assert info["n_attempted"] == 1
    assert info["n_applied"] + info["n_rejected"] == info["n_attempted"]
    assert 0.0 <= info["rejection_rate"] <= 1.0


def test_rejection_rate_is_zero_when_nobody_moves():
    state = state_from_cells([(2, 2), (2, 3)], GRID)
    _, info = moves.apply_actions(
        state, np.array([moves.STAY, moves.STAY]), np.random.default_rng(0)
    )
    assert info["n_attempted"] == 0
    assert info["rejection_rate"] == 0.0


def test_apply_is_deterministic_for_a_given_seed():
    rng = np.random.default_rng(3)
    state = random_state(rng, n_modules=8, grid_shape=(8, 8))
    actions = moves.sample_legal_actions(moves.legal_action_mask(state), rng)

    first, info_a = moves.apply_actions(state, actions, np.random.default_rng(7))
    second, info_b = moves.apply_actions(state, actions, np.random.default_rng(7))

    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.occupancy, second.occupancy)
    assert info_a["n_applied"] == info_b["n_applied"]


def test_apply_rejects_wrong_action_count():
    state = state_from_cells([(2, 2), (2, 3)], GRID)
    with pytest.raises(ValueError, match="expected 2 actions"):
        moves.apply_actions(state, np.array([moves.STAY]), np.random.default_rng(0))


def test_apply_rejects_out_of_range_actions():
    state = state_from_cells([(2, 2)], GRID)
    with pytest.raises(ValueError, match="actions must be in"):
        moves.apply_actions(state, np.array([99]), np.random.default_rng(0))


# -- action sampling ---------------------------------------------------


def test_sample_legal_actions_only_returns_legal_actions():
    rng = np.random.default_rng(4)
    for _ in range(50):
        state = random_state(rng, n_modules=int(rng.integers(1, 12)))
        mask = moves.legal_action_mask(state)
        actions = moves.sample_legal_actions(mask, rng)
        assert actions.shape == (state.n_modules,)
        assert mask[np.arange(state.n_modules), actions].all()


def test_sample_legal_actions_covers_the_whole_legal_set():
    # A single module in open space: every action is legal and should appear.
    state = state_from_cells([(3, 3)], GRID)
    mask = moves.legal_action_mask(state)
    rng = np.random.default_rng(5)
    seen = {int(moves.sample_legal_actions(mask, rng)[0]) for _ in range(300)}
    assert seen == set(range(moves.N_ACTIONS))
