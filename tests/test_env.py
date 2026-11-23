# Checks for env.py: the PettingZoo contract that train.py relies on, how start and target
# are picked, the reward formula, and when episodes end.

import copy

import numpy as np
import pytest

pytest.importorskip("pettingzoo")  # skip this file cleanly if the `rl` extras aren't installed

from pettingzoo.test import parallel_api_test  # noqa: E402

from swarm import hexgrid  # noqa: E402
from swarm.env import SwarmEnv  # noqa: E402
from swarm.shapes import make_shape  # noqa: E402

# Same values as configs/debug_tiny.yaml, written out so this file doesn't need torch.
CFG = {
    "seed": 0,
    "episode": {
        "n_modules": 4,
        "max_steps": 40,
        "target_walk_steps": 3,
        "start_shapes": ["circle", "rectangle", "star"],
    },
    "reward": {
        "fill_scale": 1.0,
        "time_cost_total": 0.5,
        "completion_bonus": 1.0,
        "isolation_penalty": 0.0,
    },
}


def _cfg(**episode_or_reward):
    """CFG with some episode/reward values replaced, e.g. _cfg(n_modules=9, fill_scale=2)."""
    cfg = copy.deepcopy(CFG)
    for key, value in episode_or_reward.items():
        section = "reward" if key in cfg["reward"] else "episode"
        cfg[section][key] = value
    return cfg


def _cell_set(cells):
    return {tuple(c) for c in np.asarray(cells).tolist()}


def _stay(env):
    return {a: 0 for a in env.agents}


def _random_actions(env, obs, rng):
    return {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}


# --- PettingZoo contract -------------------------------------------------------------------


def test_pettingzoo_parallel_api():
    parallel_api_test(SwarmEnv(CFG), num_cycles=100)


def test_reset_observation_format():
    env = SwarmEnv(CFG)
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(infos) == set(env.possible_agents)
    assert env.possible_agents == ["module_0", "module_1", "module_2", "module_3"]
    size = env.observation_space("module_0")["observation"].shape[0]
    assert size == 6 + 2 + len(hexgrid.window_offsets(env.r))
    for n, a in enumerate(env.possible_agents):
        assert obs[a]["observation"].shape == (size,)
        assert obs[a]["observation"].dtype == np.float32
        assert obs[a]["action_mask"].shape == (7,)
        assert env.observation_space(a).contains(obs[a])
        # The layout train.py was told about: 6 neighbour bits, (i, j) / R, target bitmap.
        o = obs[a]["observation"]
        assert (o[:6] == env.sim.neighbor_occupancy()[n]).all()
        assert np.allclose(o[6:8], env.sim.positions[n] / env.r)
        assert o[8:].sum() == env.n  # all N target cells fit in the crop


def test_target_bitmap_is_the_target_and_shared():
    env = SwarmEnv(CFG)
    obs, _ = env.reset(seed=3)
    bits = obs["module_0"]["observation"][8:]
    in_crop = {tuple(c) for c, b in zip(env.crop.tolist(), bits, strict=True) if b}
    assert in_crop == _cell_set(env.target_cells)
    for a in env.possible_agents:  # every robot sees the same target
        assert (obs[a]["observation"][8:] == bits).all()


# --- start and target ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_target_is_a_short_walk_from_the_start(seed):
    env = SwarmEnv(CFG)
    env.reset(seed=seed)
    start, target = _cell_set(env.start_cells), _cell_set(env.target_cells)
    assert len(target) == env.n
    assert start != target  # never solved before it begins
    assert len(start - target) <= env.k  # each walk move changes at most one cell
    assert (hexgrid.hex_distance(env.target_cells) <= env.r).all()  # fits the crop
    assert _cell_set(env.sim.positions) == start  # robots are back on the start
    assert env.sim.completion() < 1.0


def test_start_shapes_config_is_used():
    env = SwarmEnv(_cfg(start_shapes=["star"]))
    for seed in range(5):
        env.reset(seed=seed)
        assert env.start_name == "star"
        assert _cell_set(env.start_cells) == _cell_set(make_shape("star", 4))


def test_same_seed_same_episode():
    a, b = SwarmEnv(CFG), SwarmEnv(CFG)
    a.reset(seed=7)
    b.reset(seed=7)
    assert a.start_name == b.start_name
    assert (a.target_cells == b.target_cells).all()


def test_bad_config_raises():
    with pytest.raises(ValueError, match="start_shapes"):
        SwarmEnv(_cfg(start_shapes=["triangle"]))
    with pytest.raises(ValueError, match="target_walk_steps"):
        SwarmEnv(_cfg(target_walk_steps=0))
    with pytest.raises(ValueError, match="n_modules"):
        SwarmEnv(_cfg(n_modules=1))


# --- reward and episode end ----------------------------------------------------------------


def test_reward_formula_during_random_play():
    # Every step: delta(filled) * fill_scale / N - time cost - isolation term (+ bonus at the end).
    env = SwarmEnv(_cfg(n_modules=9, fill_scale=2.0, isolation_penalty=0.3))
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    filled = env.sim.filled()
    for _ in range(env.max_steps):
        obs, rew, term, trunc, info = env.step(_random_actions(env, obs, rng))
        r = next(iter(rew.values()))
        assert len(set(rew.values())) == 1  # one team reward
        isolated = int((~env.sim.neighbor_occupancy().any(axis=1)).sum())
        expected = (env.sim.filled() - filled) * 2.0 / 9 - 0.5 / env.max_steps - 0.3 * isolated / 9
        if env.sim.filled() == 9:
            expected += 1.0
        assert r == pytest.approx(expected)
        filled = env.sim.filled()
        if not env.agents:
            break


def test_truncates_at_max_steps():
    env = SwarmEnv(CFG)
    env.reset(seed=0)
    for t in range(1, env.max_steps + 1):
        _, rew, term, trunc, info = env.step(_stay(env))
        assert rew["module_0"] == pytest.approx(-0.5 / env.max_steps)  # staying: time cost only
        assert not any(term.values())
        assert all(trunc.values()) == (t == env.max_steps)
    assert info["module_0"]["success"] is False
    assert env.agents == []


def test_undoing_the_walk_terminates_with_bonus():
    # K=1: the walk moved one robot from cell A to cell B. Moving it back the other way
    # (from A to B, since the robots are back on the start) completes the target.
    env = SwarmEnv(_cfg(target_walk_steps=1))
    env.reset(seed=0)
    (a_cell,) = _cell_set(env.start_cells) - _cell_set(env.target_cells)
    (b_cell,) = _cell_set(env.target_cells) - _cell_set(env.start_cells)
    robot = [tuple(p) for p in env.sim.positions.tolist()].index(a_cell)
    d = hexgrid.direction_index(np.subtract(b_cell, a_cell))
    actions = _stay(env)
    actions[env.possible_agents[robot]] = d + 1

    _, rew, term, trunc, info = env.step(actions)
    assert all(term.values()) and not any(trunc.values())
    assert info["module_0"]["success"] is True
    assert info["module_0"]["completion"] == 1.0
    # One more cell filled, minus the time cost, plus the completion bonus.
    assert rew["module_0"] == pytest.approx(1 / env.n - 0.5 / env.max_steps + 1.0)
    assert env.agents == []


def test_isolation_penalty():
    # Two robots side by side, each pivoting around the other, both moving away at once:
    # (0,0) -> (0,1) and (1,0) -> (1,-1). They end up 2 apart, both with no neighbours.
    env = SwarmEnv(_cfg(n_modules=2, start_shapes=["circle"], isolation_penalty=0.4))
    env.reset(seed=0)
    assert env.sim.positions.tolist() == [[0, 0], [1, 0]]
    filled = env.sim.filled()
    _, rew, _, _, info = env.step({"module_0": 2, "module_1": 5})
    assert info["module_0"]["isolated"] == 2
    expected = (env.sim.filled() - filled) / 2 - 0.5 / env.max_steps - 0.4 * 2 / 2
    if env.sim.filled() == 2:
        expected += 1.0
    assert rew["module_0"] == pytest.approx(expected)


# --- with train.py -------------------------------------------------------------------------


def test_fixed_pair_reuses_the_first_pair():
    env = SwarmEnv(_cfg(fixed_pair=True))
    env.reset(seed=0)
    start, target = env.start_cells.copy(), env.target_cells.copy()
    for seed in [None, 1, 2, None]:
        env.reset(seed=seed)
        np.testing.assert_array_equal(env.start_cells, start)
        np.testing.assert_array_equal(env.target_cells, target)

    # Without the flag, the pair changes between episodes.
    env = SwarmEnv(_cfg())
    env.reset(seed=0)
    pairs = set()
    for _ in range(10):
        env.reset()
        pairs.add(tuple(map(tuple, env.target_cells.tolist())))
    assert len(pairs) > 1


def test_keep_connected_targets_are_connected():
    env = SwarmEnv(_cfg(n_modules=9, target_walk_steps=8, keep_connected=True))
    for seed in range(20):
        env.reset(seed=seed)
        assert hexgrid.is_connected(env.target_cells)


def test_freeze_ends_episode_with_penalty():
    cfg = _cfg(n_modules=6, keep_connected=True, end_on_freeze=True)
    cfg["reward"]["freeze_penalty"] = 0.5
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    ring = hexgrid.DIRS.copy()  # locked under keep_connected: nobody can move
    target = np.vstack([ring[:5], [[0, 0]]])  # differs from the ring, so not complete
    env.sim.reset(ring, target)
    env._filled = env.sim.filled()

    _, rewards, terms, truncs, infos = env.step({a: 0 for a in env.possible_agents})
    info = infos["module_0"]
    assert all(terms.values()) and not any(truncs.values())
    assert info["frozen"] and info["success"] is False
    time_cost = 0.5 / env.max_steps
    expected = -time_cost - time_cost * (env.max_steps - 1) - 0.5
    assert rewards["module_0"] == pytest.approx(expected)
    assert env.agents == []


def test_no_freeze_ending_when_off():
    env = SwarmEnv(_cfg(n_modules=6, keep_connected=True))
    env.reset(seed=0)
    ring = hexgrid.DIRS.copy()
    env.sim.reset(ring, np.vstack([ring[:5], [[0, 0]]]))
    env._filled = env.sim.filled()
    _, _, terms, truncs, infos = env.step({a: 0 for a in env.possible_agents})
    assert not any(terms.values()) and not infos["module_0"]["frozen"]


def test_reverse_curriculum_promotes_and_replays():
    from swarm.env import ReverseCurriculum

    cur = ReverseCurriculum([4, 1, 2], promote_success_rate=0.5, window_episodes=4)
    assert cur.levels == [1, 2, 4] and cur.level == 1
    for success in [True, False, False]:
        cur.record(1, success)
    assert cur.level == 1  # window not full yet
    cur.record(2, True)  # other levels don't count
    cur.record(1, True)
    assert cur.level == 2 and len(cur.recent) == 0  # 2/4 >= 0.5: promoted, window reset
    rng = np.random.default_rng(0)
    assert {cur.sample(rng) for _ in range(200)} == {1, 2}  # mostly 2, some replays of 1
    for _ in range(8):
        cur.record(2, True)
    assert cur.level == 4 and cur.at_final
    for _ in range(8):
        cur.record(4, True)
    assert cur.level == 4 and cur.success_rate == 1.0  # tracked, but no level above


def _cell_set(cells):
    return set(map(tuple, np.asarray(cells).tolist()))


def test_curriculum_level_one_is_one_move_from_target():
    cfg = _cfg(n_modules=9, target_walk_steps=8, keep_connected=True)
    cfg["curriculum"] = {"enabled": True, "levels": [1, 2, 4, 8, 16]}
    env = SwarmEnv(cfg)
    assert env.curriculum.levels == [1, 2, 4, 8]  # 16 > K is dropped
    obs_size = env.observation_space("module_0")["observation"].shape[0]
    for seed in range(20):
        _, infos = env.reset(seed=seed)
        assert infos["module_0"]["level"] == 1
        start, target = _cell_set(env.start_cells), _cell_set(env.target_cells)
        assert len(start - target) == 1  # exactly one robot is off the target
        assert len(env._observe()["module_0"]["observation"]) == obs_size


def test_curriculum_promotes_during_play():
    cfg = _cfg(n_modules=4, keep_connected=True)
    cfg["curriculum"] = {
        "enabled": True,
        "levels": [1, 2],
        "window_episodes": 3,
        "promote_success_rate": 0.0,
    }
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    for _ in range(4):  # finish a few episodes by running out of time
        while env.agents:
            env.step({a: 0 for a in env.possible_agents})
        env.reset()
    assert env.curriculum.level == 2


def test_train_runs_on_real_env(monkeypatch, tmp_path):
    """A few PPO updates on the real env: the pieces fit together end to end."""
    pytest.importorskip("torch")
    from swarm import train

    monkeypatch.chdir(tmp_path)  # policy.pt lands here
    cfg = {**CFG, "train": {"total_steps": 400, "rollout_steps": 100, "log_every": 1}}
    train.train(cfg)
    assert (tmp_path / "policy.pt").exists()
