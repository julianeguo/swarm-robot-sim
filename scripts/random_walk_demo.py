#!/usr/bin/env python3
"""Random-walk a connected swarm, asserting every invariant at every step.

The milestone-1 deliverable: exercises the state record, the connectivity
oracle, and the movement rules with nothing else in the loop. No target shape,
no reward, no observations.

    python3 scripts/random_walk_demo.py --config configs/debug_tiny.yaml --steps 40 --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm import config as config_module  # noqa: E402
from swarm.render.ascii import render_with_summary  # noqa: E402
from swarm.sim import moves  # noqa: E402
from swarm.sim.shapes import random_connected_shape  # noqa: E402
from swarm.sim.state import SwarmState  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug_tiny.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    parser.add_argument("--render", action="store_true", help="print the grid every step")
    parser.add_argument("--every", type=int, default=1, help="render every Nth step")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = config_module.load(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    positions = random_connected_shape(rng, cfg.episode.n_modules, cfg.grid.shape)
    state = SwarmState.from_positions(positions, cfg.grid.shape)

    print(f"config {args.config}  grid {cfg.grid.shape}  modules {state.n_modules}  seed {seed}")
    if args.render:
        print(render_with_summary(state, crop=True))

    attempted = applied = 0
    for step in range(args.steps):
        mask = moves.legal_action_mask(state)
        assert mask[:, moves.STAY].all(), "stay must always be legal"

        actions = moves.sample_legal_actions(mask, rng)
        state, info = moves.apply_actions(state, actions, rng, validate=True)

        attempted += info["n_attempted"]
        applied += info["n_applied"]

        if args.render and (step + 1) % args.every == 0:
            print(render_with_summary(state, crop=True))

    rejected = attempted - applied
    rate = rejected / attempted if attempted else 0.0
    print(
        f"\n{args.steps} steps  attempted {attempted}  applied {applied}  "
        f"rejected {rejected} ({rate:.1%})"
    )
    print("all invariants held: count conserved, in bounds, no overlap, connected every step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
