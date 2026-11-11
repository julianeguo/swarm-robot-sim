# Shape library: builds start and target shapes with exactly N cells, centred on the origin.
#
# How a shape is built (score-ranking):
#   Each shape has one score function that answers "how big would this shape have to be
#   to include the cell at (x, y)?". Cells near the centre score low, cells far out score
#   high. make_shape grows the shape outward from the origin one cell at a time, always
#   adding the lowest-scoring cell that touches the shape so far, and stops at N.
#   This is the same as inflating the shape from size 0 and stopping at exactly N cells,
#   so every shape hits N exactly, with no search for the right size. Only adding cells
#   that touch the shape means it can never come out in disconnected pieces.
#   The params (w, h, number of star points...) only set the proportions. N sets the size.
#
# This file holds no state and makes no random choices. sim.py (or the env's reset) picks
# which two shapes an episode uses and calls make_shape for each one.

import heapq
from functools import cache

import numpy as np

from swarm import hexgrid

# --- Score functions -------------------------------------------------------------------
# Each takes Cartesian coordinates x, y (arrays, from hexgrid.to_xy) plus that shape's
# proportion params and returns one score per cell. Every score function is "scale-free":
# doubling a point's distance from the centre doubles its score, so sorting by score
# grows the shape evenly outward.


def circle_score(x, y):
    """Circle: the score is the distance from the centre."""
    return np.hypot(x, y)


def rectangle_score(x, y, w=2.0, h=1.0):
    """Rectangle with proportions w : h (default 2 : 1, wide along x).

    The score is how much the w-by-h rectangle must be scaled to reach the cell.
    """
    return np.maximum(np.abs(x) / w, np.abs(y) / h)


def star_score(x, y, points=5, inner_ratio=0.5):
    """Star with straight edges, one tip pointing straight up.

    The score is distance / r_star(angle), where r_star is the radius of the star's
    outline in that direction: 1 at a tip, inner_ratio halfway between two tips.
    Cells in line with a tip score lower than cells between tips, so the tips grow out further.
    """
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    # Fold every angle into phi = angular distance to the nearest tip, in [0, s].
    s = np.pi / points  # angle from a tip to the next inner corner
    u = np.mod(theta - np.pi / 2, 2 * s)
    phi = np.minimum(u, 2 * s - u)

    # Radius of the straight edge from the tip (radius 1) to the inner corner
    # (radius inner_ratio), along the ray at angle phi from the tip.
    rho = inner_ratio
    r_star = rho * np.sin(s) / (rho * np.sin(s) * np.cos(phi) + (1 - rho * np.cos(s)) * np.sin(phi))
    return r / r_star


# --- Registry --------------------------------------------------------------------------
# Maps each shape name to its score function. make_shape looks names up here. Adding a
# shape = write one score function above + add one line here. Nothing else changes.
SHAPES = {
    "circle": circle_score,
    "rectangle": rectangle_score,
    "star": star_score,
}

# The valid names. The env uses this to check the shapes listed in the config, to pick a
# random start/target pair, and to size the board (by building every shape once).
SHAPE_NAMES = tuple(SHAPES)


# --- Building shapes -------------------------------------------------------------------


def make_shape(name, n, **params):
    """Return shape `name` with exactly n cells, as a sorted (n, 2) int array of (i, j).

    This is the main entry point. sim.py / the env's reset calls it for the start shape
    (to place modules) and the target shape (to fill the target array). Optional params
    override that shape's proportions, e.g. make_shape("star", 50, points=6).

    Results are cached: the first call builds the shape, later calls with the same
    arguments return the stored array. The array is read-only because every caller shares
    it. Modifying it would silently change the shape for every later episode. Call .copy()
    if you need to edit it.

    The shape is always one connected piece (see _build_shape).
    """
    if name not in SHAPES:
        raise ValueError(f"unknown shape {name!r}; choose from {SHAPE_NAMES}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    # Sort params so make_shape("star", 50, points=5, inner_ratio=0.5) and the same call
    # with the keywords swapped share one cache entry.
    return _build_shape(name, int(n), tuple(sorted(params.items())))


@cache
def _build_shape(name, n, params):
    score_fn = SHAPES[name]
    kwargs = dict(params)

    def sort_key(cell):
        # Most important first: score, then distance, then angle (CCW from 0°). The extra
        # keys break ties the same way every time. Distance makes a partly-filled
        # rectangle edge fill from its middle outward. Rounding makes values that are
        # equal on paper, but differ in the last float digit, tie.
        x, y = hexgrid.to_xy(cell)
        score = float(score_fn(x, y, **kwargs))
        dist = float(np.hypot(x, y))
        angle = float(np.mod(np.arctan2(y, x), 2 * np.pi))
        return (round(score, 9), round(dist, 9), round(angle, 9))

    # Grow from the origin. `frontier` holds every cell touching the shape so far, in a
    # heap so the lowest sort key comes out first. Whenever the plain "N lowest scores"
    # set is already connected, this picks exactly that set. It only differs when that
    # set would leave a loose cell, e.g. at a thin star tip.
    origin = (0, 0)
    frontier = [(sort_key(origin), origin)]
    seen = {origin}
    chosen = []
    while len(chosen) < n:
        _, cell = heapq.heappop(frontier)
        chosen.append(cell)
        for nb in hexgrid.neighbors(cell).tolist():
            nb = tuple(nb)
            if nb not in seen:
                seen.add(nb)
                heapq.heappush(frontier, (sort_key(nb), nb))

    # Sort the output by (i, j) so it's in a predictable order (the order means nothing).
    chosen = np.array(sorted(chosen), dtype=np.int64)

    # Can't fail with the growth above. It catches anyone breaking the growth loop later.
    assert hexgrid.is_connected(chosen), f"shape {name!r} n={n} came out disconnected"

    chosen.setflags(write=False)
    return chosen


# --- Helpers for other files -----------------------------------------------------------


def shape_radius(cells):
    """How far the shape reaches from the origin (max hex distance of any cell).

    The env takes the largest shape_radius over all shapes and passes it to
    hexgrid.board_radius, so one board size works for every start/target pair.
    """
    return int(hexgrid.hex_distance(cells).max())


def overlap_fraction(start, target):
    """Fraction of target cells that the start shape already covers.

    This is the "do nothing" baseline. Both shapes are centred on the origin, so an
    episode where no module moves already scores this much. Logged next to the final
    completion fraction so learning can be compared against it. It also flags pairs too
    similar to make an interesting task.
    """
    start_set = {tuple(c) for c in np.asarray(start).tolist()}
    hits = sum(tuple(c) in start_set for c in np.asarray(target).tolist())
    return hits / len(target)
