"""
The simulator itself, with no RL code. It holds the state (the set of occupied cells and each
robot's position) and has:

reset(start_cells, target_cells) to place the robots and set the target
legal_action_mask(), which gives each robot's 7 actions with the pivot check
apply_actions(), which applies the moves and resolves conflicts
a completion score comparing the current configuration against the target cells

State (all kept in sync on every move):
  positions : (N, 2) int (i, j)   -> robot n is at positions[n]. Row order = robot identity,
                                     and matches the order of env.possible_agents.
  occupied  : (2M+1, 2M+1) bool   -> "is this cell taken?" in one lookup (see hexgrid board helpers)
  target    : (2M+1, 2M+1) bool   -> the target shape's cells

The board radius M is chosen by the caller (env.py, via hexgrid.board_radius) and stays fixed
for the sim's lifetime, so every episode uses the same board size.

Movement is rotation: a robot swings 60 degrees around one occupied neighbour (its pivot), and
its path sweeps toward the other cell that touches both its cell and the destination, so that
cell must be empty. allow_squeeze=True drops that check (the original rule, kept so earlier
runs can be reproduced): a robot may then pass between two occupied flanking cells.

keep_connected=True adds two local rules that together mean the swarm can never split:
  * a robot may only move if its occupied neighbours form one unbroken run around its 6-cell
    ring (so they stay connected to each other without it), checked in legal_action_mask
  * two adjacent robots may not both move in the same step; one is picked at random and the
    other stays (a neighbour lock), applied in apply_actions
Proof sketch: every mover's old neighbours stay put and stay connected around it, and every
mover lands next to its pivot, which is one of those stationary neighbours.
"""

import random

import numpy as np

from swarm import hexgrid

N_ACTIONS = 7  # 0 = stay, 1..6 = move toward hexgrid.DIRS[a - 1] (same encoding as train.py)


class SwarmSim:
    def __init__(self, m, keep_connected=False, allow_squeeze=False):
        """Make an empty board of radius m. Call reset() before anything else.

        keep_connected: use the connectivity-preserving rules described at the top of this file.
        allow_squeeze: allow moves with both flanking cells occupied (no sweep check).
        """
        self.m = int(m)
        self.keep_connected = bool(keep_connected)
        self.allow_squeeze = bool(allow_squeeze)
        size = 2 * self.m + 1
        self.positions = np.zeros((0, 2), dtype=np.int64)
        self.occupied = np.zeros((size, size), dtype=bool)
        self.target = np.zeros((size, size), dtype=bool)

    @property
    def n(self):
        """Number of robots."""
        return len(self.positions)

    def reset(self, start_cells, target_cells):
        """Place one robot on each start cell (robot n on start_cells[n]) and set the target.

        Both are (N, 2) arrays of lattice cells (i, j), e.g. from shapes.make_shape. The cells
        are copied, so passing make_shape's read-only arrays is fine.

        Raises ValueError if the two shapes have different sizes, a shape repeats a cell, or a
        cell is off the board. The env should never trigger these. They catch bugs early.
        """
        start = _as_cells(start_cells, "start")
        target = _as_cells(target_cells, "target")
        if len(start) != len(target):
            raise ValueError(
                f"start has {len(start)} cells but target has {len(target)}; they must match"
            )
        for name, cells in (("start", start), ("target", target)):
            if not hexgrid.in_bounds(cells, self.m).all():
                raise ValueError(f"{name} has cells off the board (M={self.m}); is M too small?")

        self.positions = start
        self.occupied = hexgrid.cells_to_mask(start, self.m)
        self.target = hexgrid.cells_to_mask(target, self.m)

    def neighbor_occupancy(self):
        """Return an (N, 6) bool array: True where robot n's neighbour in direction k is taken.

        Columns follow hexgrid.DIRS order. Used for the legal action mask, and by env.py for
        each robot's 6 neighbour bits and to count robots with no neighbours.
        """
        nbs = hexgrid.neighbors(self.positions)  # (N, 6, 2)
        # Off-board neighbours would wrap around to the far side of the array silently.
        assert hexgrid.in_bounds(nbs, self.m).all(), "a robot is at the board edge; M too small"
        idx = hexgrid.to_index(nbs, self.m)
        return self.occupied[idx[..., 0], idx[..., 1]]

    def legal_action_mask(self):
        """Return an (N, 7) bool array: True where robot n may take action a.

        Actions: 0 = stay (always legal), a = 1..6 = move one step toward hexgrid.DIRS[a - 1].

        Moving in direction d is legal when
          1. the destination p + DIRS[d] is empty, and
          2. exactly one of the two cells touching both p and the destination (the flanks,
             p + DIRS[d - 1] and p + DIRS[d + 1]) is occupied. That robot is the pivot the mover
             rotates around (hexgrid.pivot_destination gives the same landing cell); the other
             flank is the cell its path sweeps through, so it must be empty.
             With allow_squeeze, at least one flank occupied is enough.
        With keep_connected, also
          3. the robot's occupied neighbours form exactly one unbroken run around its ring.

        This only looks at the current board. Two robots may both be allowed into the same
        cell here; apply_actions resolves that.
        """
        occ = self.neighbor_occupancy()  # (N, 6): is neighbour k taken?

        # np.roll(occ, -1)[:, d] == occ[:, d + 1] and np.roll(occ, 1)[:, d] == occ[:, d - 1].
        flank_ccw, flank_cw = np.roll(occ, -1, axis=1), np.roll(occ, 1, axis=1)
        if self.allow_squeeze:
            has_pivot = flank_ccw | flank_cw
        else:
            has_pivot = flank_ccw ^ flank_cw  # one pivot, and a clear sweep on the other side
        mask = np.ones((self.n, N_ACTIONS), dtype=bool)
        mask[:, 1:] = ~occ & has_pivot
        if self.keep_connected:
            # A run starts wherever an occupied neighbour follows an empty one going round.
            runs = (occ & ~np.roll(occ, 1, axis=1)).sum(axis=1)
            mask[:, 1:] &= (runs == 1)[:, None]
        return mask

    def apply_actions(self, actions):
        """Move every robot at once. actions is (N,) ints in 0..6, one per robot, in robot order.

        Every action must be legal under legal_action_mask() for the current board; an illegal
        one raises ValueError (the trained policy never picks one, so this means a bug).

        Conflicts, since all robots move at the same time:
          * Several robots heading for the same empty cell: one of them, picked at random,
            moves there; the others stay where they are.
          * A robot's pivot moving away in the same step is ignored: the pivot only has to be
            there at the start of the step. A robot can end up with no neighbours; it then
            can only stay until another robot comes next to it.
          * With keep_connected, a mover next to a robot that already moved this step stays
            put instead (neighbour lock), so the case above can't happen.

        Returns an (N,) bool array: True for robots that actually moved.
        """
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError(f"expected {self.n} actions, got shape {actions.shape}")
        if ((actions < 0) | (actions >= N_ACTIONS)).any():
            raise ValueError(f"actions must be in 0..{N_ACTIONS - 1}")
        legal = self.legal_action_mask()[np.arange(self.n), actions]
        if not legal.all():
            raise ValueError(f"illegal actions for robots {np.flatnonzero(~legal).tolist()}")

        # Visit the movers in random order; the first to claim a cell gets it.
        movers = np.flatnonzero(actions != 0).tolist()
        random.shuffle(movers)
        moved = np.zeros(self.n, dtype=bool)
        claimed = set()
        moved_from = set()  # old cells of robots that moved this step (for the neighbour lock)
        for n in movers:
            dest = self.positions[n] + hexgrid.DIRS[actions[n] - 1]
            if tuple(dest.tolist()) in claimed:
                continue
            if self.keep_connected and any(
                tuple(nb) in moved_from for nb in hexgrid.neighbors(self.positions[n]).tolist()
            ):
                continue
            claimed.add(tuple(dest.tolist()))
            moved_from.add(tuple(self.positions[n].tolist()))
            # The destination was empty at the start of the step (the mask checked it), and
            # no other robot has claimed it, so the write order here doesn't matter.
            old_r, old_c = hexgrid.to_index(self.positions[n], self.m)
            new_r, new_c = hexgrid.to_index(dest, self.m)
            self.occupied[old_r, old_c] = False
            self.occupied[new_r, new_c] = True
            self.positions[n] = dest
            moved[n] = True
        return moved

    def filled(self):
        """Number of target cells that currently have a robot on them (0..N)."""
        return int((self.occupied & self.target).sum())

    def completion(self):
        """Fraction of target cells filled, 0.0 to 1.0. 1.0 means the target shape is done.

        Meant to be read once per step, after apply_actions, never partway through it: all
        robots move at the same time, so the order apply_actions handles them in has no
        meaning. The env reads it before and after each step and uses the change as the
        per-step reward.
        """
        return self.filled() / self.n


def _as_cells(cells, name):
    """Return cells as a new (N, 2) int64 array, after checking that no cell appears twice.

    The copy is what reset stores as self.positions. It has to be our own writable array:
    make_shape's output is read-only and shared across episodes, and apply_actions moves
    robots by writing into positions. Converting also accepts plain lists of (i, j).
    The duplicate check (np.unique) is separate and only counts distinct rows.
    """
    cells = np.array(cells, dtype=np.int64).reshape(-1, 2)
    if len(np.unique(cells, axis=0)) != len(cells):
        raise ValueError(f"{name} has duplicate cells")
    return cells
