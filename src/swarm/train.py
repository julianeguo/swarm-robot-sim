"""
Minimal masked IPPO training for the swarm (first working version).

One small network is shared by every module. Each step, every module looks at its own
observation, picks one of 7 actions (illegal ones masked out), and the whole team gets
one shared reward. After a batch of steps we do a few PPO updates and print progress.

Kept out on purpose for now (add later, one at a time, when needed):
evaluation runs, TensorBoard, minibatches, CNN encoder.

Curriculum: env.py decides the level and when to move up (see SwarmEnv / ReverseCurriculum).
train.py only shows the level, saves checkpoints/level_<j>.pt when a level is passed, and only
allows early stopping once the final level is reached.

Critic (train.critic):
  "shared"   one network, actor and critic heads on a shared body, same local input (original)
  "central"  separate networks: the actor sees only the local observation; the critic sees the
             local observation + env.state(), global information that exists only in training
             (centralized training, decentralized execution). Only the actor is deployed.

Outputs (in the working directory):
  checkpoints/latest.pt  the weights at the most recent progress line
  checkpoints/best.pt    the weights with the best success rate seen in a progress window
  policy.pt              the final weights (actor and critic)
  actor.pt               the final actor weights only: what would run on the robots
  checkpoints/resume.pt  everything needed to continue training (see Resuming)

Resuming: python -m swarm.train <config.yaml> --resume [path]   (default checkpoints/resume.pt)
  resume.pt is rewritten at every progress line. It holds the weights, the optimizer state,
  the update count, best-so-far and early-stop counters, the curriculum level, and every
  random-number generator. Training continues from the saved update with a fresh episode (the
  episode that was in progress isn't saved), so it's a continuation, not a bit-exact replay.
  The config must match the one the run started with, except train.total_steps, which may be
  raised to extend a finished run.
Early stopping (off by default): set train.stop_success, e.g. 0.99, and training ends once
the success rate reaches it in stop_patience progress windows in a row.

Run with:  python -m swarm.train configs/debug_tiny.yaml

-------------------------------------------------------------------------------------------
What train.py expects from env.py (the only file it talks to)
-------------------------------------------------------------------------------------------
env is a PettingZoo ParallelEnv with a fixed number of modules N.
("Module" here and "robot" in sim.py mean the same thing.)

  obs, infos = env.reset(seed=...)
  obs, rewards, terminations, truncations, infos = env.step(actions)

  obs[agent] = {
      "observation": 1-D float array, the same length for every module:
          6 neighbour bits (hexgrid.DIRS order)
        + 2 numbers: position (i, j), divided by R
        + the target bitmap around the origin, radius R, flattened
          (R = largest start shape radius + K, the number of random-walk moves that make the
           target. The target is always within R of the origin, so this small crop holds the
           whole target. env.py builds it.)
      "action_mask": 7 bools, True = legal (from sim.legal_action_mask())
  }
  actions[agent] = int in 0..6   (0 = stay, 1..6 = move toward hexgrid.DIRS[a - 1])
  rewards[agent] = the team reward for this step, the same for every module: built from the
                   change in sim.completion() over the step, scaled by the config `reward:`
                   block (which also sets the time cost and the completion bonus)
  infos[agent]["success"] = True if sim.completion() == 1.0 (set when the episode ends)

An episode ends by termination (target completed) or truncation (max_steps). All modules end
together, and train.py treats both the same way.

Everything about shapes, the board, and the movement rules lives in env.py / sim.py.
train.py never imports hexgrid or shapes. From sim.py it takes only N_ACTIONS.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from swarm.sim import N_ACTIONS  # 7: stay + 6 moves. One definition, shared with the sim.

try:
    from swarm.env import SwarmEnv
except ImportError:  # env.py doesn't define SwarmEnv yet. Tests swap in a fake env instead.
    SwarmEnv = None

# Training settings. Values in the config's `train:` block override these.
DEFAULTS = {
    "total_steps": 200_000,  # env steps to train for
    "rollout_steps": 256,  # env steps collected before each update
    "lr": 3e-4,  # Adam learning rate
    "gamma": 0.99,  # discount: how much future reward counts
    "gae_lambda": 0.95,  # GAE smoothing between short- and long-term estimates
    "clip_eps": 0.2,  # PPO: how far one update may move the policy
    "n_epochs": 4,  # passes over each rollout
    "ent_coef": 0.01,  # bonus for keeping the policy random (exploration)
    "hidden": 128,  # width of the network layers
    "critic": "shared",  # "shared" or "central" (critic also sees env.state(); see above)
    "log_every": 10,  # print progress (and save checkpoints) every this many updates
    "checkpoint_dir": "checkpoints",  # latest.pt and best.pt go here
    "stop_success": None,  # e.g. 0.99 to stop early once training is solved; None = never
    "stop_patience": 3,  # ...for this many progress windows in a row
}


def load_config(path):
    """Read a YAML config. If it says `inherits: other.yaml`, load that first and put this
    file's values on top (nested sections are merged key by key)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "inherits" not in cfg:
        return cfg

    base = load_config(path.parent / cfg.pop("inherits"))
    for section, value in cfg.items():
        if isinstance(value, dict) and isinstance(base.get(section), dict):
            base[section] = {**base[section], **value}
        else:
            base[section] = value
    return base


class Policy(nn.Module):
    """The shared actor-critic: a small MLP body with two heads.

        observation --> Linear -> Tanh -> Linear -> Tanh --> actor head  -> 7 scores (logits)
                                                        \\-> critic head -> 1 value

    The actor scores the 7 actions. The critic guesses how much reward is still to come.
    All N modules go through it as one batch of N rows: same weights, different inputs.
    """

    uses_state = False  # the critic sees only the local observation

    def __init__(self, obs_size, hidden):
        super().__init__()
        self.body = _mlp(obs_size, hidden)
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs, mask, state=None):
        """Return (dist, value) for a batch of observations.

        obs: (B, obs_size) float. mask: (B, 7) bool. state is ignored (see CentralCriticPolicy).
        Illegal actions get a very negative score, so their probability is ~0.
        """
        h = self.body(obs)
        return _masked_dist(self.actor(h), mask), self.critic(h).squeeze(-1)


class CentralCriticPolicy(nn.Module):
    """Separate actor and critic networks; only the critic gets the global state.

        local obs ----------------> actor MLP  -> 7 scores (logits)    deployed on the robots
        local obs + global state -> critic MLP -> 1 value              training only

    The actor's weights use the same names as Policy's (body.*, actor.*), so actor_weights()
    of either one loads into a plain Policy for running or the demo.
    """

    uses_state = True

    def __init__(self, obs_size, state_size, hidden):
        super().__init__()
        self.body = _mlp(obs_size, hidden)
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic_net = nn.Sequential(_mlp(obs_size + state_size, hidden), nn.Linear(hidden, 1))

    def forward(self, obs, mask, state):
        """obs: (B, obs_size), mask: (B, 7), state: (B, state_size) (the global state, one
        copy per row). The actor never touches state."""
        dist = _masked_dist(self.actor(self.body(obs)), mask)
        value = self.critic_net(torch.cat([obs, state], dim=-1)).squeeze(-1)
        return dist, value


def _mlp(n_in, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())


def _masked_dist(logits, mask):
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def actor_weights(policy):
    """Just the actor's weights (body.*, actor.*): what would be deployed on the robots."""
    return {k: v for k, v in policy.state_dict().items() if k.startswith(("body.", "actor."))}


def _stack_obs(obs, agents):
    """Turn the env's per-agent dict into two tensors: (N, obs_size) and (N, 7)."""
    o = np.stack([obs[a]["observation"] for a in agents]).astype(np.float32)
    m = np.stack([obs[a]["action_mask"] for a in agents]).astype(bool)
    return torch.from_numpy(o), torch.from_numpy(m)


def _global_state(env, policy, n):
    """env.state() repeated once per robot, or None if the critic doesn't use it."""
    if not policy.uses_state:
        return None
    return torch.from_numpy(np.asarray(env.state(), dtype=np.float32)).expand(n, -1).clone()


def collect_rollout(env, policy, obs, n_steps):
    """Play n_steps env steps with the current policy and record what happened.

    Returns (batch, obs, successes):
      batch: dict of tensors shaped (n_steps, N, ...):
             obs, mask, action, log_prob, value, reward, done (+ state for a central critic)
             plus last_value (N,) = critic's guess for the step after the last one
      obs:   the env's current observation, so the next rollout continues from there
      successes: list of True/False, one per episode that finished during this rollout
    """
    agents = env.possible_agents  # fixed order, so row n is always the same module
    keys = ["obs", "mask", "action", "log_prob", "value", "reward", "done"]
    if policy.uses_state:
        keys.append("state")
    record = {k: [] for k in keys}
    successes = []

    for _ in range(n_steps):
        o, m = _stack_obs(obs, agents)
        s = _global_state(env, policy, len(agents))  # read before stepping, matching obs
        with torch.no_grad():
            dist, value = policy(o, m, s)
            action = dist.sample()

        actions = {a: int(action[n]) for n, a in enumerate(agents)}
        obs, rewards, terms, truncs, infos = env.step(actions)
        done = any(terms.values()) or any(truncs.values())  # all modules end together

        record["obs"].append(o)
        record["mask"].append(m)
        record["action"].append(action)
        record["log_prob"].append(dist.log_prob(action))
        record["value"].append(value)
        record["reward"].append(torch.tensor([float(rewards[a]) for a in agents]))
        record["done"].append(torch.full((len(agents),), float(done)))
        if s is not None:
            record["state"].append(s)

        if done:
            successes.append(bool(infos[agents[0]].get("success", False)))
            obs, _ = env.reset()

    batch = {k: torch.stack(v) for k, v in record.items()}
    with torch.no_grad():
        last_state = _global_state(env, policy, len(agents))
        _, batch["last_value"] = policy(*_stack_obs(obs, agents), last_state)
    return batch, obs, successes


def compute_advantages(batch, gamma, gae_lambda):
    """Work out how much better each action was than the critic expected (GAE).

    Goes backwards through time, separately for each module (all N at once):
        next_value = last_value at the final step, otherwise value[t + 1]
        not_done   = 0 if the episode ended at step t, else 1
        delta      = reward[t] + gamma * next_value * not_done - value[t]
        gae        = delta + gamma * gae_lambda * not_done * gae
    returns = advantage + value   (what the critic should have predicted)

    Every module gets the same reward but sees a different observation, so its value and
    advantage differ. That's how one team reward becomes credit for each module.
    Simplification: hitting max_steps is treated like a real ending (small bias, fine for now).
    """
    reward, value, done = batch["reward"], batch["value"], batch["done"]
    n_steps = reward.shape[0]
    advantages = torch.zeros_like(reward)
    gae = torch.zeros_like(batch["last_value"])

    for t in reversed(range(n_steps)):
        next_value = batch["last_value"] if t == n_steps - 1 else value[t + 1]
        not_done = 1.0 - done[t]
        delta = reward[t] + gamma * next_value * not_done - value[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae

    return advantages, advantages + value


def ppo_update(policy, optimizer, batch, advantages, returns, cfg):
    """A few PPO steps on the whole rollout at once (no minibatches yet).

    Returns a dict of the last policy loss, value loss and entropy, for printing.
    """
    # Every module-step is one training sample: flatten (T, N, ...) to (T*N, ...).
    obs = batch["obs"].flatten(0, 1)
    mask = batch["mask"].flatten(0, 1)
    state = batch["state"].flatten(0, 1) if "state" in batch else None
    action = batch["action"].flatten()
    old_log_prob = batch["log_prob"].flatten()
    returns = returns.flatten()
    adv = advantages.flatten()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    eps = cfg["clip_eps"]
    for _ in range(cfg["n_epochs"]):
        dist, value = policy(obs, mask, state)  # same masks as when the actions were taken
        ratio = torch.exp(dist.log_prob(action) - old_log_prob)
        # The clamp stops one update from changing the policy too much.
        policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
        value_loss = ((value - returns) ** 2).mean()
        entropy = dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - cfg["ent_coef"] * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }


def _save_resume(path, cfg, policy, optimizer, env, update, best, windows_at_target):
    """Write everything needed to continue training. Written to a temp file and renamed,
    so a run killed mid-save never leaves a broken resume.pt behind."""
    curriculum = getattr(env, "curriculum", None)
    env_rng = getattr(env, "rng", None)
    bundle = {
        "config": cfg,
        "update": update,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best": list(best),
        "windows_at_target": windows_at_target,
        "curriculum": (
            {"index": curriculum.index, "recent": list(curriculum.recent)} if curriculum else None
        ),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "env": env_rng.bit_generator.state if env_rng is not None else None,
        },
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(bundle, tmp)
    os.replace(tmp, path)


def _check_resume_config(saved, cfg):
    """Refuse to resume with a different setup (only train.total_steps may change)."""

    def strip(c):
        c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in c.items()}
        c.get("train", {}).pop("total_steps", None)
        return c

    if strip(saved) != strip(cfg):
        raise ValueError(
            "the config differs from the one this run started with (only train.total_steps "
            "may change when resuming)"
        )


def train(cfg, resume=None):
    """Tie it together: make the env and policy, then alternate collecting and updating.

    resume: path to a checkpoints/resume.pt to continue from (see Resuming at the top).
    """
    settings = {**DEFAULTS, **(cfg.get("train") or {})}
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    random.seed(seed)  # sim.apply_actions breaks move conflicts with Python's `random`
    np.random.seed(seed)

    if SwarmEnv is None:
        raise RuntimeError("swarm.env.SwarmEnv is not available yet; write env.py first")
    env = SwarmEnv(cfg)
    obs, _ = env.reset(seed=seed)
    obs_size = env.observation_space(env.possible_agents[0])["observation"].shape[0]

    if settings["critic"] == "central":
        state_size = len(env.state())
        policy = CentralCriticPolicy(obs_size, state_size, settings["hidden"])
        print(f"central critic: actor input {obs_size}, critic input {obs_size + state_size}")
    elif settings["critic"] == "shared":
        policy = Policy(obs_size, settings["hidden"])
    else:
        raise ValueError(f"train.critic must be 'shared' or 'central', got {settings['critic']!r}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings["lr"])

    n_updates = settings["total_steps"] // settings["rollout_steps"]
    ckpt_dir = Path(settings["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = (-1.0, -np.inf)  # (success rate, reward/step) of the best window so far
    windows_at_target = 0
    recent_successes = []
    curriculum = getattr(env, "curriculum", None)

    start = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)  # our own file
        _check_resume_config(ckpt["config"], cfg)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["update"]
        best = tuple(ckpt["best"])
        windows_at_target = ckpt["windows_at_target"]
        if curriculum and ckpt["curriculum"]:
            curriculum.index = ckpt["curriculum"]["index"]
            curriculum.recent.clear()
            curriculum.recent.extend(ckpt["curriculum"]["recent"])
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if rng["env"] is not None and getattr(env, "rng", None) is not None:
            env.rng.bit_generator.state = rng["env"]
        obs, _ = env.reset()  # a fresh episode: the one in progress wasn't saved
        print(f"resumed from {resume} at step {start * settings['rollout_steps']}")

    update = start
    for update in range(start + 1, n_updates + 1):
        level_before = curriculum.level if curriculum else None
        batch, obs, successes = collect_rollout(env, policy, obs, settings["rollout_steps"])
        if curriculum and curriculum.level != level_before:
            torch.save(policy.state_dict(), ckpt_dir / f"level_{level_before}.pt")
            print(f"curriculum: passed level {level_before}, now level {curriculum.level}")
        advantages, returns = compute_advantages(batch, settings["gamma"], settings["gae_lambda"])
        losses = ppo_update(policy, optimizer, batch, advantages, returns, settings)
        recent_successes += successes

        if update % settings["log_every"] == 0 or update == n_updates:
            steps = update * settings["rollout_steps"]
            reward = batch["reward"].mean().item()
            rate = np.mean(recent_successes) if recent_successes else float("nan")
            level = ""
            if curriculum:
                level = f" | level {curriculum.level} ({curriculum.success_rate:.2f} at level)"
            print(
                f"steps {steps:>8}{level} | reward/step {reward:.3f} | "
                f"success {rate:.2f} ({len(recent_successes)} eps) | "
                f"pi {losses['policy_loss']:.3f} v {losses['value_loss']:.3f} "
                f"ent {losses['entropy']:.2f}"
            )
            recent_successes = []

            torch.save(policy.state_dict(), ckpt_dir / "latest.pt")
            # Before the final level, success is measured on easier problems; don't let those
            # count as "best" or toward early stopping.
            final = curriculum is None or curriculum.at_final
            if curriculum:
                rate = curriculum.success_rate  # leave out the easier replayed episodes
            if final and not np.isnan(rate) and (rate, reward) > best:  # ties: reward/step
                best = (rate, reward)
                torch.save(policy.state_dict(), ckpt_dir / "best.pt")

            target = settings["stop_success"]
            hit = final and target and rate >= target
            windows_at_target = windows_at_target + 1 if hit else 0
            _save_resume(
                ckpt_dir / "resume.pt",
                cfg,
                policy,
                optimizer,
                env,
                update,
                best,
                windows_at_target,
            )
            if target and windows_at_target >= settings["stop_patience"]:
                print(
                    f"early stop: success >= {target} for {windows_at_target} windows in a row "
                    f"(at {steps} steps)"
                )
                break

    torch.save(policy.state_dict(), "policy.pt")
    torch.save(actor_weights(policy), "actor.pt")
    if best[0] >= 0:
        print(f"best: success {best[0]:.2f}, reward/step {best[1]:.3f} -> {ckpt_dir}/best.pt")
    env.close()
    return policy


def main():
    """Run with: python -m swarm.train configs/debug_tiny.yaml [--resume [path]]"""
    parser = argparse.ArgumentParser(description="Train the swarm policy (masked IPPO).")
    parser.add_argument("config", help="YAML config, e.g. configs/debug_tiny.yaml")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="continue from a resume.pt (default: <checkpoint_dir>/resume.pt)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    resume = args.resume
    if resume == "auto":
        ckpt_dir = (cfg.get("train") or {}).get("checkpoint_dir", DEFAULTS["checkpoint_dir"])
        resume = Path(ckpt_dir) / "resume.pt"
    if resume is not None and not Path(resume).exists():
        raise SystemExit(f"no resume file at {resume}")
    train(cfg, resume=resume)


if __name__ == "__main__":
    main()
