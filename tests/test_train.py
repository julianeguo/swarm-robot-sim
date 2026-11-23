# Checks for train.py.
#
# Key idea: these tests do NOT use the real env.py. They use FakeEnv (below), a tiny env
# that follows the same contract written at the top of train.py. That way:
#   * train.py can be tested before env.py defines SwarmEnv,
#   * a failure here points at train.py, not at the simulator,
#   * FakeEnv doubles as a runnable example of what env.py must return.
# The few tests that touch sim.py check the pieces train.py shares with it (action count,
# action mask), so the two can't drift apart.

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip this file cleanly if torch isn't installed

from swarm import sim, train  # noqa: E402  (must come after the importorskip above)
from swarm.shapes import make_shape, shape_radius  # noqa: E402

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# --- A tiny stand-in env -----------------------------------------------------------------


class FakeEnv:
    """Smallest env that follows train.py's contract, with a task that's easy to learn.

    * 2 modules: possible_agents = ["module_0", "module_1"]
    * observation: one-hot of length 7 saying which action is "correct" for this module
      this step (picked at random each step, never action 6)
    * action_mask: all True except action 6, which is always illegal (to test masking)
    * reward: the same for both modules = fraction of modules that picked their correct action
    * episode: ends (truncation) after 5 steps
    * infos[agent]["success"] at the end: True if every module was correct on the last step

    A random policy gets ~1/6 reward per step. A trained one should get close to 1.0.
    """

    possible_agents = ["module_0", "module_1"]
    episode_length = 5

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.correct = None

    def _observe(self):
        self.correct = self.rng.integers(0, 6, size=len(self.possible_agents))
        obs = {}
        for n, agent in enumerate(self.possible_agents):
            one_hot = np.zeros(7, dtype=np.float32)
            one_hot[self.correct[n]] = 1.0
            mask = np.ones(7, dtype=bool)
            mask[6] = False
            obs[agent] = {"observation": one_hot, "action_mask": mask}
        return obs

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._observe(), {a: {} for a in self.possible_agents}

    def step(self, actions):
        agents = self.possible_agents
        hits = [actions[a] == self.correct[n] for n, a in enumerate(agents)]
        reward = float(np.mean(hits))
        self.t += 1
        end = self.t >= self.episode_length
        infos = {a: ({"success": all(hits)} if end else {}) for a in agents}
        return (
            self._observe(),
            {a: reward for a in agents},
            {a: False for a in agents},
            {a: end for a in agents},
            infos,
        )

    def observation_space(self, agent):
        # Stand-in for a gymnasium Dict space: train.py only reads ["observation"].shape.
        return {"observation": SimpleNamespace(shape=(7,))}

    def close(self):
        pass


class TerminatingFakeEnv(FakeEnv):
    """FakeEnv that ends episodes by termination instead of truncation.

    The real env terminates as soon as the target is complete (sim.completion() == 1.0), so
    train.py must treat termination the same as hitting max_steps.
    """

    def step(self, actions):
        obs, rewards, terms, truncs, infos = super().step(actions)
        return obs, rewards, truncs, terms, infos  # swap: the episode end is a termination


class GlobalRandomFakeEnv(FakeEnv):
    """FakeEnv whose randomness comes from Python's global `random`, like sim.apply_actions
    (which breaks move conflicts with random.shuffle). Only train()'s own seeding makes it
    repeatable: reset(seed=...) deliberately ignores the seed.
    """

    def reset(self, seed=None):
        return super().reset(seed=None)

    def _observe(self):
        self.correct = np.array([random.randrange(6) for _ in self.possible_agents])
        obs = {}
        for n, agent in enumerate(self.possible_agents):
            one_hot = np.zeros(7, dtype=np.float32)
            one_hot[self.correct[n]] = 1.0
            mask = np.ones(7, dtype=bool)
            mask[6] = False
            obs[agent] = {"observation": one_hot, "action_mask": mask}
        return obs


def greedy_reward_per_step(policy, n_steps=100):
    """Play FakeEnv always taking the most likely action. Return the mean reward per step."""
    env = FakeEnv()
    obs, _ = env.reset(seed=1)
    rewards = []
    for _ in range(n_steps):
        o, m = train._stack_obs(obs, env.possible_agents)
        with torch.no_grad():
            dist, _ = policy(o, m)
        action = dist.probs.argmax(-1)
        obs, r, _, trunc, _ = env.step(
            {a: int(action[n]) for n, a in enumerate(env.possible_agents)}
        )
        rewards.append(r["module_0"])
        if trunc["module_0"]:
            obs, _ = env.reset()
    return float(np.mean(rewards))


# --- load_config ---------------------------------------------------------------------------


def test_load_config_merges_inherited_file():
    """debug_tiny.yaml inherits default.yaml: its own values win, the rest come from default."""
    cfg = train.load_config(CONFIGS / "debug_tiny.yaml")
    # Overridden by debug_tiny.
    assert cfg["episode"]["n_modules"] == 4
    assert cfg["grid"]["height"] == 8
    assert cfg["curriculum"]["enabled"] is False
    # Inherited from default.
    assert cfg["seed"] == 0
    assert cfg["reward"]["completion_bonus"] == 1.0
    # Nested merge kept the siblings that only default sets in `curriculum:`.
    assert cfg["curriculum"]["levels"] == [1, 2, 4, 8]
    # The `inherits` key itself is gone.
    assert "inherits" not in cfg


# --- Policy --------------------------------------------------------------------------------


def test_policy_output_shapes():
    """For a batch of B observations: one distribution over 7 actions and one value each."""
    policy = train.Policy(obs_size=10, hidden=16)
    dist, value = policy(torch.randn(5, 10), torch.ones(5, 7, dtype=torch.bool))
    assert dist.probs.shape == (5, train.N_ACTIONS)
    assert value.shape == (5,)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(5))


def test_policy_never_picks_masked_actions():
    """Illegal actions get probability ~0 and are never sampled."""
    torch.manual_seed(0)
    policy = train.Policy(obs_size=10, hidden=16)
    obs = torch.randn(1000, 10)

    mask = torch.zeros(1000, 7, dtype=torch.bool)
    mask[:, [0, 3]] = True
    dist, _ = policy(obs, mask)
    assert set(dist.sample().tolist()) <= {0, 3}
    assert (dist.probs[:, [1, 2, 4, 5, 6]] < 1e-6).all()

    # Edge case: only "stay" (0) is legal -> always 0, and nothing turns into NaN.
    only_stay = torch.zeros(1000, 7, dtype=torch.bool)
    only_stay[:, 0] = True
    dist, value = policy(obs, only_stay)
    assert (dist.sample() == 0).all()
    assert not torch.isnan(dist.probs).any()
    assert not torch.isnan(value).any()


# --- Agreement with sim.py -----------------------------------------------------------------


def test_n_actions_comes_from_sim():
    """train.py and sim.py share one action count: stay + 6 moves."""
    assert train.N_ACTIONS is sim.N_ACTIONS
    assert train.N_ACTIONS == 7


def test_policy_actions_are_accepted_by_sim():
    """Sampling with sim.legal_action_mask() only gives actions apply_actions accepts.

    apply_actions raises ValueError on any illegal action, so this runs many steps of real
    sim play through the policy's masking and checks nothing ever slips through.
    """
    torch.manual_seed(0)
    random.seed(0)
    start, target = make_shape("circle", 12), make_shape("rectangle", 12)
    n_steps = 30
    swarm = sim.SwarmSim(shape_radius(start) + shape_radius(target) + n_steps + 1)
    swarm.reset(start, target)
    policy = train.Policy(obs_size=10, hidden=16)

    for _ in range(n_steps):
        mask = torch.from_numpy(swarm.legal_action_mask())
        assert mask.shape == (swarm.n, train.N_ACTIONS)
        with torch.no_grad():
            dist, _ = policy(torch.randn(swarm.n, 10), mask)
        swarm.apply_actions(dist.sample().numpy())  # raises if any action was illegal


# --- compute_advantages --------------------------------------------------------------------


def _one_module_batch(done):
    return {
        "reward": torch.tensor([[1.0], [1.0]]),
        "value": torch.tensor([[0.5], [0.5]]),
        "done": torch.tensor(done),
        "last_value": torch.tensor([0.5]),
    }


def test_advantages_match_hand_calculation():
    """A 2-step, 1-module example small enough to work out on paper (gamma=0.9, lambda=1)."""
    # delta1 = 1 + 0.9*0.5 - 0.5 = 0.95   -> adv1 = 0.95
    # delta0 = 1 + 0.9*0.5 - 0.5 = 0.95   -> adv0 = 0.95 + 0.9*0.95 = 1.805
    batch = _one_module_batch(done=[[0.0], [0.0]])
    advantages, returns = train.compute_advantages(batch, gamma=0.9, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.tensor([[1.805], [0.95]]))
    assert torch.allclose(returns, advantages + batch["value"])


def test_advantages_stop_at_episode_end():
    """After done, the next episode's value must not leak back into this one."""
    batch = _one_module_batch(done=[[1.0], [0.0]])  # episode ended after step 0
    advantages, _ = train.compute_advantages(batch, gamma=0.9, gae_lambda=1.0)
    assert advantages[0, 0].item() == pytest.approx(1.0 - 0.5)  # reward0 - value0 only


def test_advantages_are_per_module():
    """Modules with the same reward but different values get different advantages."""
    batch = {
        "reward": torch.ones(3, 2),
        "value": torch.tensor([[0.0, 1.0]] * 3),
        "done": torch.zeros(3, 2),
        "last_value": torch.tensor([0.0, 1.0]),
    }
    advantages, _ = train.compute_advantages(batch, gamma=0.9, gae_lambda=0.95)
    assert advantages.shape == (3, 2)
    assert not torch.allclose(advantages[:, 0], advantages[:, 1])


# --- collect_rollout -----------------------------------------------------------------------


def test_collect_rollout_shapes_and_episodes():
    """12 steps with FakeEnv (5-step episodes): right shapes, 2 finished episodes."""
    torch.manual_seed(0)
    env = FakeEnv()
    obs, _ = env.reset(seed=0)
    policy = train.Policy(obs_size=7, hidden=16)

    batch, obs, successes = train.collect_rollout(env, policy, obs, n_steps=12)

    assert batch["obs"].shape == (12, 2, 7)
    assert batch["mask"].shape == (12, 2, 7)
    for key in ["action", "log_prob", "value", "reward", "done"]:
        assert batch[key].shape == (12, 2), key
    assert batch["last_value"].shape == (2,)

    # Episodes end after steps 4 and 9 (0-based), for both modules together.
    assert batch["done"][:, 0].nonzero().flatten().tolist() == [4, 9]
    assert torch.equal(batch["done"][:, 0], batch["done"][:, 1])
    assert len(successes) == 2

    # Every stored action was legal under its stored mask (so never action 6).
    chosen_legal = batch["mask"].gather(-1, batch["action"].unsqueeze(-1))
    assert chosen_legal.all()
    # Team reward: both modules got the same reward at every step.
    assert torch.equal(batch["reward"][:, 0], batch["reward"][:, 1])
    # The returned obs is the env's live observation, so the next rollout can continue.
    assert set(obs) == set(env.possible_agents)


def test_collect_rollout_treats_termination_as_episode_end():
    """Termination (target completed) ends an episode just like truncation (max_steps)."""
    torch.manual_seed(0)
    env = TerminatingFakeEnv()
    obs, _ = env.reset(seed=0)
    policy = train.Policy(obs_size=7, hidden=16)

    batch, _, successes = train.collect_rollout(env, policy, obs, n_steps=12)

    assert batch["done"][:, 0].nonzero().flatten().tolist() == [4, 9]
    assert len(successes) == 2


# --- ppo_update ----------------------------------------------------------------------------


def test_ppo_update_moves_policy_toward_good_actions():
    """If action 2 always had positive advantage (and action 1 negative), p(2) should go up."""
    torch.manual_seed(0)
    policy = train.Policy(obs_size=7, hidden=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    T, N = 50, 2
    obs = torch.randn(7).expand(T, N, 7).clone()  # the same situation every sample
    mask = torch.ones(T, N, 7, dtype=torch.bool)
    action = torch.tensor([2, 1]).repeat(T, 1)  # module 0 picked 2, module 1 picked 1
    advantages = torch.tensor([1.0, -1.0]).repeat(T, 1)  # 2 was good, 1 was bad

    with torch.no_grad():
        dist, _ = policy(obs.flatten(0, 1), mask.flatten(0, 1))
        p2_before = dist.probs[0, 2].item()
        log_prob = dist.log_prob(action.flatten()).view(T, N)

    batch = {"obs": obs, "mask": mask, "action": action, "log_prob": log_prob}
    settings = {**train.DEFAULTS, "n_epochs": 4}
    losses = train.ppo_update(policy, optimizer, batch, advantages, torch.zeros(T, N), settings)

    assert all(np.isfinite(v) for v in losses.values())
    with torch.no_grad():
        dist, _ = policy(obs[0, :1], mask[0, :1])
    assert dist.probs[0, 2].item() > p2_before


# --- train (end to end) --------------------------------------------------------------------


def test_train_without_env_gives_clear_error(monkeypatch):
    """Until env.py defines SwarmEnv, train() should say so instead of failing obscurely."""
    monkeypatch.setattr(train, "SwarmEnv", None)
    with pytest.raises(RuntimeError, match="env.py"):
        train.train({"seed": 0})


def test_train_learns_fake_task(monkeypatch, tmp_path, capsys):
    """The whole loop runs and actually learns: the shortest proof that the pieces fit."""
    monkeypatch.setattr(train, "SwarmEnv", lambda cfg: FakeEnv())  # no real env needed
    monkeypatch.chdir(tmp_path)  # policy.pt lands here

    cfg = {
        "seed": 0,
        "train": {"total_steps": 6000, "rollout_steps": 100, "lr": 3e-3, "log_every": 20},
    }
    policy = train.train(cfg)

    assert (tmp_path / "policy.pt").exists()
    assert "success" in capsys.readouterr().out  # progress lines were printed
    # A random policy scores ~0.17 per step on FakeEnv.
    assert greedy_reward_per_step(policy) > 0.5

    # The saved file holds the same trained weights.
    reloaded = train.Policy(obs_size=7, hidden=train.DEFAULTS["hidden"])
    reloaded.load_state_dict(torch.load(tmp_path / "policy.pt"))
    assert greedy_reward_per_step(reloaded) == greedy_reward_per_step(policy)


def test_train_seeds_python_random(monkeypatch, tmp_path):
    """Same seed -> same trained weights, even when the env draws from Python's `random`.

    sim.apply_actions breaks move conflicts with random.shuffle, so train() must seed
    `random` itself; torch.manual_seed alone would leave runs unrepeatable.
    """
    monkeypatch.setattr(train, "SwarmEnv", lambda cfg: GlobalRandomFakeEnv())
    monkeypatch.chdir(tmp_path)
    cfg = {"seed": 3, "train": {"total_steps": 200, "rollout_steps": 50, "log_every": 100}}

    first = train.train(cfg).state_dict()
    random.random()  # disturb the global state between runs; train() must reset it
    second = train.train(cfg).state_dict()

    assert all(torch.equal(first[k], second[k]) for k in first)


def test_checkpoints_and_early_stop(monkeypatch, tmp_path, capsys):
    """Checkpoints are written, and stop_success ends training before total_steps."""
    monkeypatch.setattr(train, "SwarmEnv", lambda cfg: FakeEnv())
    monkeypatch.chdir(tmp_path)
    cfg = {
        "seed": 0,
        "train": {
            "total_steps": 200_000,
            "rollout_steps": 100,
            "lr": 3e-3,
            "log_every": 20,
            "stop_success": 0.9,
            "stop_patience": 2,
        },
    }
    train.train(cfg)
    out = capsys.readouterr().out
    assert "early stop" in out
    assert "steps   200000" not in out  # stopped well before the budget
    for name in ["checkpoints/latest.pt", "checkpoints/best.pt", "policy.pt"]:
        assert (tmp_path / name).exists(), name
