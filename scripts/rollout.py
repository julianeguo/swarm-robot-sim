#!/usr/bin/env python3
"""Run a policy over many episodes headless and print aggregate statistics.

The milestone-2 deliverable. Works with any policy satisfying the Policy
interface, so the same script measures the baselines now and a trained
checkpoint later.

    python3 scripts/rollout.py --config configs/debug_tiny.yaml --policy random \
        --episodes 100
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm import config as config_module  # noqa: E402
from swarm.render.ascii import render_with_summary  # noqa: E402
from swarm.rl.policies import RandomLegalPolicy, StayPolicy  # noqa: E402
from swarm.sim.episodes import (  # noqa: E402
    Curriculum,
    make_episode,
    make_library_episode,
    make_unpaired_episode,
)
from swarm.sim.simulator import Simulator  # noqa: E402

POLICIES = {"random": RandomLegalPolicy, "stay": StayPolicy}
EPISODE_SOURCES = ("random_walk", "unpaired", "library")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug_tiny.yaml")
    parser.add_argument("--policy", default="random", choices=sorted(POLICIES))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    parser.add_argument("--source", default="random_walk", choices=EPISODE_SOURCES)
    parser.add_argument("--walk-steps", type=int, default=None, help="overrides config K")
    parser.add_argument("--start-shape", default="square", help="library source only")
    parser.add_argument("--target-shape", default="line_h", help="library source only")
    parser.add_argument("--curriculum", action="store_true", help="advance K on success")
    parser.add_argument("--render-first", action="store_true", help="show episode 1 start/end")
    return parser.parse_args()


def build_policy(name: str, seed: int):
    """RandomLegalPolicy takes a seed; the deterministic baselines take nothing."""
    policy_cls = POLICIES[name]
    return policy_cls(seed=seed) if policy_cls is RandomLegalPolicy else policy_cls()


def build_episode(args, cfg, rng, curriculum):
    if args.source == "library":
        return make_library_episode(cfg, args.start_shape, args.target_shape)
    if args.source == "unpaired":
        return make_unpaired_episode(cfg, rng)
    walk_steps = curriculum.walk_steps if curriculum.enabled else args.walk_steps
    return make_episode(cfg, rng, walk_steps=walk_steps)


def run_episode(simulator, policy, spec):
    """Roll one episode to termination and return its summary statistics."""
    state = simulator.reset(spec)
    start_filled = state.n_filled
    attempted = applied = 0

    while not simulator.done:
        mask = simulator.legal_action_mask()
        actions = policy.act(None, mask)
        state, info = simulator.step(actions)
        attempted += info["n_attempted"]
        applied += info["n_applied"]

    return {
        "success": simulator.terminated,
        "steps": state.step,
        "fill_fraction": state.fill_fraction,
        "fill_gain": state.n_filled - start_filled,
        "attempted": attempted,
        "applied": applied,
        "degenerate": spec.is_degenerate,
        "walk_steps": spec.walk_steps,
    }


def main() -> int:
    args = parse_args()
    cfg = config_module.load(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    policy = build_policy(args.policy, seed)
    simulator = Simulator(seed=seed)
    # The curriculum only advances when asked for, so a plain rollout measures a
    # single fixed difficulty rather than a moving one.
    curriculum = Curriculum(
        replace(cfg.curriculum, enabled=args.curriculum and cfg.curriculum.enabled),
        cfg.episode.target_walk_steps,
    )

    print(
        f"policy {policy.name}  config {args.config}  source {args.source}  "
        f"grid {cfg.grid.shape}  modules {cfg.episode.n_modules}  "
        f"max_steps {cfg.episode.max_steps}  seed {seed}"
    )

    results = []
    for episode in range(args.episodes):
        spec = build_episode(args, cfg, rng, curriculum)

        if args.render_first and episode == 0:
            print("\n-- episode 1 start --")
            print(render_with_summary(spec.initial_state(), crop=True))

        result = run_episode(simulator, policy, spec)
        results.append(result)
        curriculum.record(result["success"])

        if args.render_first and episode == 0:
            print("-- episode 1 end --")
            print(render_with_summary(simulator.state, crop=True))
            print()

    report(results, curriculum if args.curriculum else None)
    return 0


def report(results, curriculum) -> None:
    n = len(results)
    successes = sum(r["success"] for r in results)
    degenerate = sum(r["degenerate"] for r in results)
    attempted = sum(r["attempted"] for r in results)
    applied = sum(r["applied"] for r in results)
    solved_steps = [r["steps"] for r in results if r["success"] and not r["degenerate"]]

    print(f"\nepisodes            {n}")
    print(f"success rate        {successes / n:.1%}  ({successes}/{n})")
    print(f"degenerate episodes {degenerate / n:.1%}  (target already satisfied at reset)")
    print(f"mean target filled  {np.mean([r['fill_fraction'] for r in results]):.1%}")
    print(f"mean steps          {np.mean([r['steps'] for r in results]):.1f}")
    if solved_steps:
        print(f"mean steps (solved) {np.mean(solved_steps):.1f}   non-degenerate only")
    print(f"move rejection rate {(attempted - applied) / attempted:.1%}" if attempted else "")

    if curriculum is not None:
        print(
            f"curriculum          level {curriculum.level_index} "
            f"(K={curriculum.walk_steps})  promotions {curriculum.promotions}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
