import numpy as np
import pytest

from swarm.config import Config, EpisodeConfig, GridConfig
from swarm.sim import moves
from swarm.sim.connectivity import is_connected
from swarm.sim.episodes import EpisodeSpec, make_episode, make_library_episode
from swarm.sim.simulator import Simulator


def tiny_config(**overrides) -> Config:
    base = {
        "grid": GridConfig(height=10, width=10),
        "episode": EpisodeConfig(n_modules=5, max_steps=20, target_walk_steps=5),
    }
    return Config(**{**base, **overrides})


def spec_from_cells(cells, target_cells, grid_shape=(6, 6), max_steps=10) -> EpisodeSpec:
    target = np.zeros(grid_shape, dtype=bool)
    for row, col in target_cells:
        target[row, col] = True
    return EpisodeSpec(
        start_positions=np.array(cells, dtype=np.int32),
        target=target,
        grid_shape=grid_shape,
        max_steps=max_steps,
    )


# -- lifecycle ---------------------------------------------------------


def test_requires_reset_before_use():
    simulator = Simulator(seed=0)
    with pytest.raises(RuntimeError, match="reset\\(\\) must be called"):
        _ = simulator.state
    with pytest.raises(RuntimeError, match="reset\\(\\) must be called"):
        simulator.legal_action_mask()


def test_reset_returns_the_specs_initial_state():
    spec = spec_from_cells([(1, 1), (1, 2)], [(1, 2), (3, 3)])
    simulator = Simulator(seed=0)
    state = simulator.reset(spec)
    assert state.step == 0
    assert state.n_modules == 2
    assert state.n_targets == 2
    assert state.n_filled == 1


def test_reset_clears_the_previous_episode():
    simulator = Simulator(seed=0)
    first = spec_from_cells([(1, 1), (1, 2)], [(1, 1), (1, 2)])
    simulator.reset(first)
    simulator.step(np.array([moves.STAY, moves.STAY]))
    assert simulator.state.step == 1

    second = spec_from_cells([(4, 4)], [(4, 4)])
    state = simulator.reset(second)
    assert state.step == 0
    assert state.n_modules == 1


def test_step_advances_and_reports_progress():
    # An L whose corner module only has to step right to complete the target.
    # A smaller swarm would not do: a 2-module pair is rigid, since either
    # module moving breaks the only contact holding the swarm together.
    spec = spec_from_cells([(2, 2), (3, 2), (3, 3)], [(2, 3), (3, 2), (3, 3)])
    simulator = Simulator(seed=0)
    simulator.reset(spec)

    state, info = simulator.step(np.array([moves.RIGHT, moves.STAY, moves.STAY]))
    assert info["step"] == 1
    assert info["newly_filled"] == 1
    assert info["n_filled"] == 3
    assert info["complete"] is True
    assert info["fill_fraction"] == 1.0
    assert state.is_complete


def test_step_info_carries_the_move_accounting():
    spec = spec_from_cells([(2, 2), (3, 2), (3, 3)], [(2, 2), (3, 2), (3, 3)])
    simulator = Simulator(seed=0)
    simulator.reset(spec)
    _, info = simulator.step(np.array([moves.STAY, moves.STAY, moves.STAY]))
    for key in ("n_attempted", "n_applied", "n_rejected", "rejection_rate", "applied", "order"):
        assert key in info


def test_newly_filled_can_be_negative():
    """Moving off a target cell has to register as negative progress.

    The reward is built on this delta, and its telescoping property depends on
    losses being reported as faithfully as gains.
    """
    spec = spec_from_cells([(2, 2), (3, 2), (3, 3)], [(2, 2), (3, 2), (3, 3)])
    simulator = Simulator(seed=0)
    simulator.reset(spec)
    _, info = simulator.step(np.array([moves.RIGHT, moves.STAY, moves.STAY]))
    assert info["newly_filled"] == -1
    assert info["n_filled"] == 2


# -- termination -------------------------------------------------------


def test_terminates_on_completion():
    spec = spec_from_cells([(2, 2), (3, 2), (3, 3)], [(2, 3), (3, 2), (3, 3)], max_steps=50)
    simulator = Simulator(seed=0)
    simulator.reset(spec)
    assert not simulator.done

    simulator.step(np.array([moves.RIGHT, moves.STAY, moves.STAY]))
    assert simulator.terminated
    assert not simulator.truncated
    assert simulator.done


def test_truncates_at_max_steps():
    spec = spec_from_cells([(2, 2)], [(5, 5)], max_steps=3)
    simulator = Simulator(seed=0)
    simulator.reset(spec)
    for _ in range(3):
        simulator.step(np.array([moves.STAY]))
    assert simulator.truncated
    assert not simulator.terminated
    assert simulator.done


def test_termination_wins_over_truncation():
    """Completing on the final allowed step is a success, not a timeout."""
    spec = spec_from_cells([(2, 2), (3, 2), (3, 3)], [(2, 3), (3, 2), (3, 3)], max_steps=1)
    simulator = Simulator(seed=0)
    simulator.reset(spec)
    _, info = simulator.step(np.array([moves.RIGHT, moves.STAY, moves.STAY]))
    assert info["terminated"] is True
    assert info["truncated"] is False


def test_already_complete_episode_is_done_at_reset():
    spec = spec_from_cells([(2, 2), (2, 3)], [(2, 2), (2, 3)])
    simulator = Simulator(seed=0)
    simulator.reset(spec)
    assert simulator.terminated
    assert simulator.done


# -- the mask and the rules --------------------------------------------


def test_mask_comes_straight_from_the_rules():
    spec = spec_from_cells([(2, 2), (3, 2), (3, 3)], [(2, 2)])
    simulator = Simulator(seed=0)
    state = simulator.reset(spec)
    assert np.array_equal(simulator.legal_action_mask(), moves.legal_action_mask(state))


def test_rigid_start_shape_can_never_progress():
    """square -> line is unsolvable: a solid block has no legal moves at all."""
    cfg = tiny_config(episode=EpisodeConfig(n_modules=4, max_steps=25))
    spec = make_library_episode(cfg, "square", "line_h")
    simulator = Simulator(seed=0)
    start = simulator.reset(spec)
    start_filled = start.n_filled

    mask = simulator.legal_action_mask()
    assert not mask[:, 1:].any(), "a solid block should have no legal moves"

    rng = np.random.default_rng(0)
    while not simulator.done:
        simulator.step(moves.sample_legal_actions(simulator.legal_action_mask(), rng))

    assert simulator.truncated
    assert simulator.state.n_filled == start_filled


# -- invariants over full episodes -------------------------------------


def test_invariants_hold_across_many_episodes():
    cfg = tiny_config()
    rng = np.random.default_rng(0)
    simulator = Simulator(seed=0, validate=True)

    for _ in range(40):
        spec = make_episode(cfg, rng)
        state = simulator.reset(spec)
        target = state.target
        while not simulator.done:
            actions = moves.sample_legal_actions(simulator.legal_action_mask(), rng)
            state, _ = simulator.step(actions)
            state.validate()
            assert is_connected(state.occupancy)
            assert state.n_modules == cfg.episode.n_modules
            assert state.target is target
        assert state.step <= spec.max_steps


def test_episodes_are_reproducible_from_a_seed():
    def rollout(seed: int):
        cfg = tiny_config()
        rng = np.random.default_rng(seed)
        simulator = Simulator(seed=seed)
        spec = make_episode(cfg, rng)
        simulator.reset(spec)
        action_rng = np.random.default_rng(seed)
        while not simulator.done:
            actions = moves.sample_legal_actions(simulator.legal_action_mask(), action_rng)
            simulator.step(actions)
        return simulator.state.positions, simulator.state.step

    assert np.array_equal(rollout(3)[0], rollout(3)[0])
    assert rollout(3)[1] == rollout(3)[1]


def test_simulator_does_not_mutate_the_spec():
    cfg = tiny_config()
    spec = make_episode(cfg, np.random.default_rng(0))
    start_before = spec.start_positions.copy()
    target_before = spec.target.copy()

    simulator = Simulator(seed=0)
    simulator.reset(spec)
    rng = np.random.default_rng(0)
    while not simulator.done:
        simulator.step(moves.sample_legal_actions(simulator.legal_action_mask(), rng))

    assert np.array_equal(spec.start_positions, start_before)
    assert np.array_equal(spec.target, target_before)
