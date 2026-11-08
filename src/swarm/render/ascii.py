"""Text rendering. Always available, used by tests and terminal debugging."""

from __future__ import annotations

import numpy as np

from swarm.sim.state import SwarmState

FILLED_TARGET = "#"  # module sitting on a target cell
MODULE = "o"  # module not on a target cell
UNFILLED_TARGET = "+"  # target cell with no module
EMPTY = "."


def render(state: SwarmState, crop: bool = False) -> str:
    """Render a state as framed ASCII art.

    With ``crop``, trims to the bounding box of everything interesting (modules
    plus target cells) so large sparse grids stay readable.
    """
    occupancy, target = state.occupancy, state.target

    chars = np.full(occupancy.shape, EMPTY, dtype="<U1")
    chars[target & ~occupancy] = UNFILLED_TARGET
    chars[occupancy & ~target] = MODULE
    chars[occupancy & target] = FILLED_TARGET

    if crop:
        interesting = occupancy | target
        if interesting.any():
            rows = np.flatnonzero(interesting.any(axis=1))
            cols = np.flatnonzero(interesting.any(axis=0))
            chars = chars[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    border = "+" + "-" * chars.shape[1] + "+"
    body = ["|" + "".join(row) + "|" for row in chars]
    return "\n".join([border, *body, border])


def summary(state: SwarmState) -> str:
    """One-line state summary: step, modules, and target progress."""
    if state.n_targets:
        progress = f"{state.n_filled}/{state.n_targets} filled ({state.fill_fraction:.0%})"
    else:
        progress = "no target"
    return f"step {state.step:>4}  modules {state.n_modules:>3}  {progress}"


def render_with_summary(state: SwarmState, crop: bool = False) -> str:
    return f"{summary(state)}\n{render(state, crop=crop)}"
