"""
Checks for the things most likely to break silently:
hex neighbors and coordinate conversions, every shape having exactly N cells, pivot legality,
and conflict resolution (two robots targeting one cell, a pivot robot moving away during the
same tick).
"""

import random

import numpy as np
import pytest

from swarm import hexgrid
from swarm.shapes import make_shape, overlap_fraction, shape_radius
from swarm.sim import SwarmSim

# --- reset ---------------------------------------------------------------------------------


def test_reset_places_robots_in_given_order():
    sim = SwarmSim(m=3)
    start = np.array([[0, 0], [1, 0], [0, 1]])
    target = np.array([[0, 0], [-1, 0], [0, -1]])
    sim.reset(start, target)

    assert sim.n == 3
    assert (sim.positions == start).all()  # robot n is on start[n]
    assert sim.occupied.sum() == 3 and sim.target.sum() == 3
    for n, cell in enumerate(start):
        r, c = hexgrid.to_index(cell, sim.m)
        assert sim.occupied[r, c], n


def test_reset_with_real_shapes_and_copies_them():
    start, target = make_shape("circle", 20), make_shape("star", 20)
    m = max(shape_radius(start), shape_radius(target))
    sim = SwarmSim(m)
    sim.reset(start, target)

    assert (sim.positions == start).all()
    # make_shape's arrays are read-only and shared; the sim must hold its own writable copy.
    assert sim.positions is not start and sim.positions.flags.writeable
    sim.positions[0] = [99, 99]
    assert (make_shape("circle", 20)[0] != [99, 99]).any()


def test_reset_replaces_previous_episode():
    sim = SwarmSim(m=3)
    sim.reset([[0, 0], [1, 0]], [[0, 0], [0, 1]])
    sim.reset([[2, 0]], [[-2, 0]])
    assert sim.n == 1
    assert sim.occupied.sum() == 1 and sim.target.sum() == 1


def test_reset_rejects_bad_input():
    sim = SwarmSim(m=2)
    with pytest.raises(ValueError, match="must match"):
        sim.reset([[0, 0], [1, 0]], [[0, 0]])
    with pytest.raises(ValueError, match="duplicate"):
        sim.reset([[0, 0], [0, 0]], [[0, 0], [1, 0]])
    with pytest.raises(ValueError, match="duplicate"):
        sim.reset([[0, 0], [1, 0]], [[1, 0], [1, 0]])
    with pytest.raises(ValueError, match="off the board"):
        sim.reset([[3, 0]], [[0, 0]])
    with pytest.raises(ValueError, match="off the board"):
        sim.reset([[0, 0]], [[0, -3]])


# --- legal_action_mask ---------------------------------------------------------------------
# Action a (1..6) moves toward hexgrid.DIRS[a - 1]. Directions: 0 east (1,0), 1 (0,1),
# 2 (-1,1), 3 west (-1,0), 4 (0,-1), 5 (1,-1).


def _mask(cells, m=4):
    sim = SwarmSim(m)
    sim.reset(cells, cells)
    return sim.legal_action_mask()


def test_mask_shape_and_stay_always_legal():
    mask = _mask(make_shape("circle", 20), m=6)
    assert mask.shape == (20, 7) and mask.dtype == bool
    assert mask[:, 0].all()


def test_lone_robot_can_only_stay():
    # No neighbours = nothing to pivot around.
    assert _mask([[0, 0]]).tolist() == [[True] + [False] * 6]


def test_two_robots():
    # P at the origin, Q to its east. P can't go east (taken), but it can pivot around Q
    # into the two cells that touch both: direction 1 (0,1) and direction 5 (1,-1).
    mask = _mask([[0, 0], [1, 0]])
    assert mask[0].tolist() == [True, False, True, False, False, False, True]
    # Mirror image for Q: P is west of it (direction 3), so it can reach directions 2 and 4.
    assert mask[1].tolist() == [True, False, False, True, False, True, False]


def test_both_flanks_occupied_is_still_legal():
    # P with robots at (0,1) and (1,-1): east (1,0) touches both. Simple rule: legal.
    mask = _mask([[0, 0], [0, 1], [1, -1]])
    assert mask[0, 1]


def test_mask_matches_pivot_destination():
    # Independent check: move d is legal iff the destination is empty and some occupied
    # neighbour, rotated around CW or CCW, lands the robot exactly there.
    sim = SwarmSim(8)
    cells = make_shape("star", 30)
    sim.reset(cells, cells)
    mask = sim.legal_action_mask()
    taken = {tuple(c) for c in cells.tolist()}
    for n, p in enumerate(cells):
        for d in range(6):
            dest = tuple((p + hexgrid.DIRS[d]).tolist())
            reachable = any(
                tuple(hexgrid.pivot_destination(p, k, rot).tolist()) == dest
                for k in range(6)
                if tuple((p + hexgrid.DIRS[k]).tolist()) in taken
                for rot in (hexgrid.CCW, hexgrid.CW)
            )
            assert mask[n, d + 1] == (dest not in taken and reachable), (n, d)


# --- apply_actions -------------------------------------------------------------------------


def _sim(cells, m=4):
    sim = SwarmSim(m)
    sim.reset(cells, cells)
    return sim


def _in_sync(sim):
    """positions and occupied describe the same board, with no two robots on one cell."""
    unique = len({tuple(p) for p in sim.positions.tolist()}) == sim.n
    return unique and (hexgrid.cells_to_mask(sim.positions, sim.m) == sim.occupied).all()


def test_single_move_updates_positions_and_board():
    # P at the origin pivots around Q (east) into (0, 1): action 2 = direction 1.
    sim = _sim([[0, 0], [1, 0]])
    moved = sim.apply_actions([2, 0])
    assert moved.tolist() == [True, False]
    assert sim.positions.tolist() == [[0, 1], [1, 0]]  # only robot 0 changed
    assert _in_sync(sim)


def test_all_stay_changes_nothing():
    sim = _sim(make_shape("circle", 7))
    before = sim.positions.copy()
    assert not sim.apply_actions(np.zeros(7, dtype=int)).any()
    assert (sim.positions == before).all() and _in_sync(sim)


def test_two_robots_same_cell_one_random_winner():
    # A at (1,0) and B at (0,1) pivot on each other into the empty origin:
    # A goes west (direction 3 -> action 4), B goes (0,-1) (direction 4 -> action 5).
    winners = set()
    random.seed(0)
    for _ in range(50):
        sim = _sim([[1, 0], [0, 1]])
        moved = sim.apply_actions([4, 5])
        assert moved.sum() == 1  # exactly one gets the cell, the other stays
        assert _in_sync(sim)
        winners.add(int(np.flatnonzero(moved)[0]))
    assert winners == {0, 1}  # the tie-break is random, not always the same robot


def test_moving_pivot_is_ignored():
    # P at (0,0) pivots around Q into (0,1) while Q, pivoting around P, leaves to (1,-1).
    # Different destinations, so both moves happen even though each lost its pivot.
    sim = _sim([[0, 0], [1, 0]])
    assert sim.apply_actions([2, 5]).tolist() == [True, True]
    assert sim.positions.tolist() == [[0, 1], [1, -1]]
    assert _in_sync(sim)


def test_bad_actions_raise():
    sim = _sim([[0, 0], [1, 0]])
    with pytest.raises(ValueError, match="illegal"):
        sim.apply_actions([1, 0])  # east is taken by robot 1
    with pytest.raises(ValueError, match="expected 2"):
        sim.apply_actions([0, 0, 0])
    with pytest.raises(ValueError, match="0..6"):
        sim.apply_actions([7, 0])
    assert sim.positions.tolist() == [[0, 0], [1, 0]]  # nothing moved


def test_random_legal_play_keeps_state_consistent():
    # 100 steps of random legal moves for 30 robots: the board and positions never disagree.
    random.seed(1)
    rng = np.random.default_rng(1)
    cells = make_shape("star", 30)
    steps = 100
    sim = _sim(cells, m=hexgrid.board_radius(shape_radius(cells), steps, 1))
    for _ in range(steps):
        mask = sim.legal_action_mask()
        actions = [rng.choice(np.flatnonzero(row)) for row in mask]
        sim.apply_actions(actions)
        assert _in_sync(sim)


# --- completion ----------------------------------------------------------------------------


def test_completion_values():
    sim = SwarmSim(4)
    sim.reset([[0, 0], [1, 0]], [[0, 0], [1, 0]])  # start is the target
    assert sim.filled() == 2 and sim.completion() == 1.0
    sim.reset([[0, 0], [1, 0]], [[-2, 0], [-3, 0]])  # no overlap
    assert sim.filled() == 0 and sim.completion() == 0.0
    sim.reset([[0, 0], [1, 0]], [[0, 0], [0, 1]])  # one of two target cells filled
    assert sim.filled() == 1 and sim.completion() == 0.5


def test_completion_follows_moves():
    # Robot 0 pivots around robot 1 from (0,0) into the empty target cell (0,1).
    sim = SwarmSim(4)
    sim.reset([[0, 0], [1, 0]], [[0, 1], [1, 0]])
    assert sim.completion() == 0.5
    sim.apply_actions([2, 0])
    assert sim.completion() == 1.0


def test_completion_matches_overlap_fraction():
    # At reset, completion is the "do nothing" baseline that shapes.py already computes.
    start, target = make_shape("circle", 30), make_shape("star", 30)
    sim = SwarmSim(8)
    sim.reset(start, target)
    assert sim.completion() == pytest.approx(overlap_fraction(start, target))
