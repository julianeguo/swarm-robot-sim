"""Long random-legal rollouts, asserting every simulator invariant each step.

This is the fuzzer that has to catch rule-level mistakes the hand-built
fixtures miss.
"""

import numpy as np

from swarm.sim import moves
from swarm.sim.connectivity import is_connected, n_components

from .helpers import random_state


def test_invariants_hold_over_a_long_rollout():
    rng = np.random.default_rng(0)
    n_modules, grid_shape = 12, (12, 12)
    state = random_state(rng, n_modules=n_modules, grid_shape=grid_shape)
    target = state.target

    for _ in range(500):
        mask = moves.legal_action_mask(state)
        actions = moves.sample_legal_actions(mask, rng)
        state, info = moves.apply_actions(state, actions, rng)

        state.validate()  # count, bounds, uniqueness, grid/position agreement
        assert is_connected(state.occupancy)
        assert n_components(state.occupancy) == 1
        assert state.n_modules == n_modules
        assert state.target is target, "the target shape must not mutate mid-episode"
        assert 0 <= info["n_applied"] <= info["n_attempted"]


def test_invariants_hold_across_many_swarm_sizes():
    rng = np.random.default_rng(1)
    for n_modules in (1, 2, 3, 5, 9, 20):
        state = random_state(rng, n_modules=n_modules, grid_shape=(10, 10))
        for _ in range(40):
            mask = moves.legal_action_mask(state)
            actions = moves.sample_legal_actions(mask, rng)
            state, _ = moves.apply_actions(state, actions, rng, validate=True)
            assert state.n_modules == n_modules


def test_connectivity_holds_after_every_intermediate_move():
    """Moves are applied one at a time, so each intermediate state must be valid too.

    Checking only the end-of-step state would miss a rule that lets the swarm
    pass through a disconnected configuration, which a physical lattice cannot
    do.
    """
    rng = np.random.default_rng(2)
    state = random_state(rng, n_modules=10, grid_shape=(10, 10))

    for _ in range(100):
        mask = moves.legal_action_mask(state)
        actions = moves.sample_legal_actions(mask, rng)

        # Replay the step module by module in the order apply_actions used.
        next_state, info = moves.apply_actions(state, actions, rng)
        replay = state.copy()
        for module in info["order"]:
            action = int(actions[module])
            if action == moves.STAY or not info["applied"][module]:
                continue
            row, col = int(replay.positions[module, 0]), int(replay.positions[module, 1])
            dr, dc = moves.DELTAS[action]
            replay.occupancy[row, col] = False
            replay.occupancy[row + int(dr), col + int(dc)] = True
            replay.positions[module] = (row + int(dr), col + int(dc))
            assert is_connected(replay.occupancy), "disconnected mid-step"

        assert np.array_equal(replay.occupancy, next_state.occupancy)
        state = next_state


def test_rollouts_are_reproducible_from_a_seed():
    def rollout(seed: int):
        rng = np.random.default_rng(seed)
        state = random_state(rng, n_modules=8, grid_shape=(10, 10))
        for _ in range(50):
            mask = moves.legal_action_mask(state)
            actions = moves.sample_legal_actions(mask, rng)
            state, _ = moves.apply_actions(state, actions, rng)
        return state.positions

    assert np.array_equal(rollout(5), rollout(5))
    assert not np.array_equal(rollout(5), rollout(6))


def test_random_policy_rejection_rate_is_reported_and_bounded():
    """Tracks how often a masked-legal move is rejected at apply time.

    Simultaneous action selection means this is never zero; it is a diagnostic
    to watch, not something to assert away.
    """
    rng = np.random.default_rng(3)
    state = random_state(rng, n_modules=16, grid_shape=(14, 14))
    attempted = applied = 0

    for _ in range(200):
        mask = moves.legal_action_mask(state)
        actions = moves.sample_legal_actions(mask, rng)
        state, info = moves.apply_actions(state, actions, rng)
        attempted += info["n_attempted"]
        applied += info["n_applied"]

    assert attempted > 0
    rejection_rate = (attempted - applied) / attempted
    assert 0.0 <= rejection_rate < 0.9
