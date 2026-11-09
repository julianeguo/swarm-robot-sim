import numpy as np
import pytest

from swarm import config as config_module
from swarm.config import Config, EpisodeConfig, GridConfig, RewardConfig
from swarm.reward import REWARDS, TeamFillReward, build_reward
from swarm.sim import moves
from swarm.sim.episodes import make_episode
from swarm.sim.simulator import Simulator

from .helpers import state_from_cells

GRID = (6, 6)
MAX_STEPS = 100


def reward_for(**overrides) -> TeamFillReward:
    return TeamFillReward(RewardConfig(**overrides), max_steps=MAX_STEPS)


def transition(start_cells, target_cells, actions, max_steps=MAX_STEPS):
    """Apply one step and return (prev_state, new_state, info)."""
    state = state_from_cells(start_cells, GRID, target=target_cells)
    new_state, info = moves.apply_actions(state, np.array(actions), np.random.default_rng(0))
    info = {**info, "newly_filled": new_state.n_filled - state.n_filled}
    return state, new_state, info


# -- registry ----------------------------------------------------------


def test_team_fill_is_registered():
    assert REWARDS["team_fill"] is TeamFillReward


def test_build_reward_from_config():
    cfg = config_module.load("configs/default.yaml")
    reward = build_reward(cfg.reward, cfg.episode.max_steps)
    assert isinstance(reward, TeamFillReward)
    assert reward.step_penalty == pytest.approx(
        cfg.reward.time_cost_total / cfg.episode.max_steps
    )


def test_unknown_reward_name_raises():
    with pytest.raises(KeyError, match="unknown reward 'nope'"):
        build_reward(RewardConfig(name="nope"), max_steps=10)


# -- the components ----------------------------------------------------


def test_no_progress_costs_only_time():
    reward = reward_for()
    prev, new, info = transition([(2, 2), (3, 2), (3, 3)], [(0, 0)], [0, 0, 0])
    assert reward(prev, new, info) == pytest.approx(-reward.step_penalty)


def test_filling_a_cell_pays_one_over_n():
    """Scaled by swarm size so a full solve is worth fill_scale regardless of N."""
    reward = reward_for()
    prev, new, info = transition(
        [(2, 2), (3, 2), (3, 3)], [(2, 3), (3, 2), (3, 3)], [moves.RIGHT, 0, 0]
    )
    assert info["newly_filled"] == 1
    # One fill out of three modules, one completion bonus, one step of time.
    expected = 1 / 3 - reward.step_penalty + 1.0
    assert reward(prev, new, info) == pytest.approx(expected)


def test_losing_a_cell_costs_one_over_n():
    reward = reward_for()
    prev, new, info = transition(
        [(2, 2), (3, 2), (3, 3)], [(2, 2), (3, 2), (3, 3)], [moves.RIGHT, 0, 0]
    )
    assert info["newly_filled"] == -1
    assert reward(prev, new, info) == pytest.approx(-1 / 3 - reward.step_penalty)


def test_completion_bonus_fires_only_on_the_completing_step():
    reward = reward_for()
    prev, new, info = transition(
        [(2, 2), (3, 2), (3, 3)], [(2, 3), (3, 2), (3, 3)], [moves.RIGHT, 0, 0]
    )
    assert new.is_complete
    first = reward(prev, new, info)

    # Same state again, already complete beforehand: no second bonus.
    repeat = reward(new, new, {**info, "newly_filled": 0})
    assert first - repeat == pytest.approx(1 / 3 + 1.0)
    assert repeat == pytest.approx(-reward.step_penalty)


def test_fill_scale_and_bonus_are_configurable():
    reward = reward_for(fill_scale=3.0, completion_bonus=5.0)
    prev, new, info = transition(
        [(2, 2), (3, 2), (3, 3)], [(2, 3), (3, 2), (3, 3)], [moves.RIGHT, 0, 0]
    )
    assert reward(prev, new, info) == pytest.approx(3.0 / 3 - reward.step_penalty + 5.0)


def test_zero_time_cost_is_allowed():
    reward = reward_for(time_cost_total=0.0)
    prev, new, info = transition([(2, 2), (3, 2), (3, 3)], [(0, 0)], [0, 0, 0])
    assert reward(prev, new, info) == 0.0


# -- scaling -----------------------------------------------------------


def test_step_penalty_is_a_share_of_the_episode_budget():
    """A flat per-step cost would swamp the fill term as max_steps grows.

    Total fill reward across an episode can only ever reach fill_scale, so the
    time cost has to be pinned to the episode length rather than the step.
    """
    short = TeamFillReward(RewardConfig(), max_steps=50)
    long = TeamFillReward(RewardConfig(), max_steps=500)
    assert short.step_penalty == 10 * long.step_penalty
    assert short.budget()["max_time_cost"] == pytest.approx(long.budget()["max_time_cost"])


def test_time_cost_stays_below_the_task_reward():
    reward = reward_for()
    budget = reward.budget()
    assert budget["max_time_cost"] < budget["max_fill"]
    assert budget["max_time_cost"] < budget["completion_bonus"]


def test_reward_scale_is_independent_of_swarm_size():
    """Total fill reward for a full solve is fill_scale at any N."""
    reward = reward_for(time_cost_total=0.0)
    for n_modules in (2, 5, 20):
        total = sum(
            reward.config.fill_scale * 1 / n_modules for _ in range(n_modules)
        )
        assert total == pytest.approx(reward.config.fill_scale)


# -- the telescoping property ------------------------------------------


def test_fill_term_telescopes_over_an_episode():
    """Total fill reward depends only on start and end occupancy, never the path.

    This is what stops a module farming reward by stepping on and off a target
    cell, so any reward variant should be checked against it.
    """
    cfg = Config(
        grid=GridConfig(height=12, width=12),
        episode=EpisodeConfig(n_modules=8, max_steps=60, target_walk_steps=8),
    )
    reward = TeamFillReward(RewardConfig(time_cost_total=0.0, completion_bonus=0.0), 60)

    rng = np.random.default_rng(0)
    for _ in range(10):
        spec = make_episode(cfg, rng)
        simulator = Simulator(seed=0)
        state = simulator.reset(spec)
        start_filled = state.n_filled

        total = 0.0
        while not simulator.done:
            actions = moves.sample_legal_actions(simulator.legal_action_mask(), rng)
            previous = state
            state, info = simulator.step(actions)
            total += reward(previous, state, info)

        expected = (state.n_filled - start_filled) / state.n_modules
        assert total == pytest.approx(expected, abs=1e-6)


def test_moving_on_and_off_a_target_nets_zero():
    reward = TeamFillReward(RewardConfig(time_cost_total=0.0, completion_bonus=0.0), MAX_STEPS)
    cells = [(2, 2), (3, 2), (3, 3)]
    target = [(2, 2), (3, 2), (3, 3)]

    state = state_from_cells(cells, GRID, target=target)
    rng = np.random.default_rng(0)

    off, info_off = moves.apply_actions(state, np.array([moves.RIGHT, 0, 0]), rng)
    info_off = {**info_off, "newly_filled": off.n_filled - state.n_filled}
    first = reward(state, off, info_off)

    back, info_back = moves.apply_actions(off, np.array([moves.LEFT, 0, 0]), rng)
    info_back = {**info_back, "newly_filled": back.n_filled - off.n_filled}
    second = reward(off, back, info_back)

    assert np.array_equal(back.occupancy, state.occupancy)
    assert first + second == pytest.approx(0.0)


def test_reset_does_not_change_scoring():
    """The reward is a function of the transition, so reset carries no state."""
    reward = reward_for()
    prev, new, info = transition(
        [(2, 2), (3, 2), (3, 3)], [(2, 3), (3, 2), (3, 3)], [moves.RIGHT, 0, 0]
    )
    before = reward(prev, new, info)
    reward.reset(prev)
    assert reward(prev, new, info) == pytest.approx(before)
