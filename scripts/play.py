#!/usr/bin/env python3
"""Watch a policy run, live in a window or recorded to a GIF.

The milestone-4 deliverable. Useful for seeing *why* a policy fails rather than
only that it did -- in particular, pure greedy freezing solid versus greedy with
a deadlock escape working its way through.

    # watch greedy succeed
    python3 scripts/play.py --config configs/exp_small.yaml --policy greedy \
        --epsilon 0.2

    # record pure greedy deadlocking, no window needed
    python3 scripts/play.py --config configs/exp_small.yaml --policy greedy \
        --epsilon 0 --record runs/greedy_frozen.gif --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm import config as config_module  # noqa: E402
from swarm.obs.local_window import ObservationBuilder  # noqa: E402
from swarm.render.pygame_view import PygameRenderer  # noqa: E402
from swarm.render.recording import save_gif  # noqa: E402
from swarm.reward import build_reward  # noqa: E402
from swarm.rl.policies import GreedyPolicy, RandomLegalPolicy, StayPolicy  # noqa: E402
from swarm.sim.episodes import (  # noqa: E402
    make_episode,
    make_library_episode,
    make_unpaired_episode,
)
from swarm.sim.simulator import Simulator  # noqa: E402

POLICIES = {"random": RandomLegalPolicy, "greedy": GreedyPolicy, "stay": StayPolicy}
EPISODE_SOURCES = ("random_walk", "unpaired", "library")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp_small.yaml")
    parser.add_argument("--policy", default="greedy", choices=sorted(POLICIES))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    parser.add_argument("--source", default="random_walk", choices=EPISODE_SOURCES)
    parser.add_argument("--walk-steps", type=int, default=None, help="overrides config K")
    parser.add_argument("--start-shape", default="square", help="library source only")
    parser.add_argument("--target-shape", default="line_h", help="library source only")
    parser.add_argument("--epsilon", type=float, default=0.0, help="greedy escape rate")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--cell-size", type=int, default=None)
    parser.add_argument("--record", default=None, help="write the run to this GIF path")
    parser.add_argument("--headless", action="store_true", help="never open a window")
    parser.add_argument(
        "--hold", type=int, default=6, help="repeat the final frame this many times"
    )
    return parser.parse_args()


def build_policy(args, seed: int):
    policy_cls = POLICIES[args.policy]
    if policy_cls is StayPolicy:
        return policy_cls()
    if policy_cls is GreedyPolicy:
        return policy_cls(seed=seed, epsilon=args.epsilon)
    return policy_cls(seed=seed)


def build_episode(args, cfg, rng):
    if args.source == "library":
        return make_library_episode(cfg, args.start_shape, args.target_shape)
    if args.source == "unpaired":
        return make_unpaired_episode(cfg, rng)
    return make_episode(cfg, rng, walk_steps=args.walk_steps)


def hud_lines(args, episode: int, spec, info: dict | None, total_reward: float) -> list[str]:
    label = args.policy
    if args.policy == "greedy":
        label = f"greedy eps={args.epsilon:g}"
    lines = [label, f"episode {episode + 1}/{args.episodes}", f"K={spec.walk_steps}"]
    lines.append(f"return {total_reward:+.2f}")
    if info is not None:
        lines.append(f"moved {info['n_applied']}/{info['n_attempted']}")
    return lines


def main() -> int:
    args = parse_args()
    cfg = config_module.load(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    policy = build_policy(args, seed)
    simulator = Simulator(seed=seed)
    builder = ObservationBuilder(cfg.observation, cfg.grid.shape)
    reward = build_reward(cfg.reward, cfg.episode.max_steps)
    renderer = PygameRenderer(
        cfg.grid.shape, cell_size=args.cell_size, headless=args.headless
    )

    frames: list[np.ndarray] = []
    live = not args.headless
    running = True

    for episode in range(args.episodes):
        if not running:
            break
        spec = build_episode(args, cfg, rng)
        state = simulator.reset(spec)
        builder.reset(state.n_modules, rng)
        reward.reset(state)

        total_reward = 0.0
        info = None
        frozen_steps = 0

        while True:
            hud = hud_lines(args, episode, spec, info, total_reward)
            if args.record:
                frames.append(renderer.frame(state, hud))
            if live:
                running = renderer.show(state, hud)
                renderer.tick(args.fps)
                if not running:
                    break
            if simulator.done:
                break

            mask = simulator.legal_action_mask()
            actions = policy.act(builder.build(state, mask), mask)
            previous = state
            state, info = simulator.step(actions)
            total_reward += reward(previous, state, info)
            frozen_steps += info["n_applied"] == 0

        outcome = "solved" if simulator.terminated else "timed out"
        print(
            f"episode {episode + 1}: {outcome} in {state.step} steps  "
            f"{state.n_filled}/{state.n_targets} filled  "
            f"return {total_reward:+.3f}  "
            f"frozen steps {frozen_steps}/{max(1, state.step)}"
        )

        if args.record and args.hold:
            frames.extend([frames[-1]] * args.hold)

    if args.record and frames:
        path = save_gif(frames, args.record, fps=args.fps)
        print(f"wrote {len(frames)} frames to {path}")

    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
