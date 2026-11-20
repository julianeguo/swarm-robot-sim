"""
The first milestone.
It runs random legal moves and renders them, so you can confirm by eye that movement,
conflicts, and masking all behave correctly.

With --policy it instead plays a trained policy.pt from swarm.train, so you can watch what
training learned.

Examples:
  python3 scripts/demo_random.py --config configs/debug_tiny.yaml --render
  python3 scripts/demo_random.py --config configs/exp_small.yaml --episodes 3 --render
  python3 scripts/demo_random.py --config configs/debug_tiny.yaml --gif demo.gif
  python3 scripts/demo_random.py --config configs/debug_tiny.yaml --policy policy.pt --render
"""

import argparse
import random

import numpy as np

from swarm.env import SwarmEnv
from swarm.train import load_config

VIEW_MARGIN = 2  # extra cells drawn around the target crop radius


def parse_args():
    p = argparse.ArgumentParser(description="Watch random (or trained) robots reconfigure.")
    p.add_argument("--config", default="configs/debug_tiny.yaml")
    p.add_argument("--steps", type=int, default=None,
                   help="max steps per episode (default: the config's episode.max_steps)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    p.add_argument("--render", action="store_true", help="open a pygame window")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--gif", default=None, help="also save the frames to this GIF path")
    p.add_argument("--policy", default=None, help="policy.pt from swarm.train; default: random")
    return p.parse_args()


def make_chooser(env, policy_path, rng):
    """Return a function obs -> {agent: action}, random-legal or from a trained policy."""
    agents = env.possible_agents
    if policy_path is None:
        def random_actions(obs):
            return {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in agents}
        return random_actions

    import torch

    from swarm.train import DEFAULTS, Policy, _stack_obs

    state = torch.load(policy_path, map_location="cpu")
    obs_size = env.observation_space(agents[0])["observation"].shape[0]
    hidden = state["body.0.weight"].shape[0]  # read the width back from the saved weights
    if state["body.0.weight"].shape[1] != obs_size:
        raise SystemExit(
            f"{policy_path} was trained with obs size {state['body.0.weight'].shape[1]}, but "
            f"this config gives {obs_size}; use the config it was trained on"
        )
    policy = Policy(obs_size, hidden or DEFAULTS["hidden"])
    policy.load_state_dict(state)
    policy.eval()

    def policy_actions(obs):
        o, m = _stack_obs(obs, agents)
        with torch.no_grad():
            dist, _ = policy(o, m)
            action = dist.sample()
        return {a: int(action[n]) for n, a in enumerate(agents)}
    return policy_actions


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 0) if args.seed is None else args.seed
    random.seed(seed)  # sim.apply_actions breaks move conflicts with Python's `random`
    rng = np.random.default_rng(seed)

    env = SwarmEnv(cfg)
    choose = make_chooser(env, args.policy, rng)

    renderer = None
    if args.render or args.gif:
        from swarm.render import Renderer

        view = min(env.r + VIEW_MARGIN, env.sim.m)
        renderer = Renderer(view, mode="human" if args.render else "rgb_array", fps=args.fps)
    frames = []

    def draw(step, moved=None):
        if renderer is None:
            return
        frame = renderer.draw(env.sim, step=step, moved=moved)
        if args.gif:
            if frame is None:  # human mode returns nothing; grab the window surface instead
                frame = np.transpose(
                    renderer.pygame.surfarray.array3d(renderer.surface), (1, 0, 2)
                ).astype(np.uint8)
            frames.append(frame)

    max_steps = args.steps or env.max_steps
    successes = 0
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=seed + ep)
            draw(0)
            info = {}
            for t in range(1, max_steps + 1):
                before = env.sim.positions.copy()
                obs, _, terms, truncs, infos = env.step(choose(obs))
                moved = (env.sim.positions != before).any(axis=1)
                info = infos[env.possible_agents[0]]
                draw(t, moved)
                if any(terms.values()) or any(truncs.values()):
                    break
            success = bool(info.get("success", False))
            successes += success
            print(
                f"episode {ep}: start={env.start_name} steps={t} "
                f"completion={info.get('completion', 0.0):.2f} success={success}"
            )
    finally:
        if renderer is not None:
            renderer.close()

    print(f"solved {successes}/{args.episodes}")
    if args.gif:
        import imageio.v2 as imageio

        imageio.mimsave(args.gif, frames, duration=1 / args.fps, loop=0)
        print(f"saved {len(frames)} frames to {args.gif}")


if __name__ == "__main__":
    main()
