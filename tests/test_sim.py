"""
Checks for the things most likely to break silently:
hex neighbors and coordinate conversions, every shape having exactly N cells, pivot legality,
and conflict resolution (two robots targeting one cell, a pivot robot moving away during the
same tick).
"""

import numpy as np
import pytest

from swarm import hexgrid
from swarm.shapes import make_shape, shape_radius
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
