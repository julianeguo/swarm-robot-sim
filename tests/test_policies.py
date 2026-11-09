import numpy as np
import pytest

from swarm.config import Config, EpisodeConfig, GridConfig
from swarm.rl.policies import GreedyPolicy, Policy, RandomLegalPolicy, StayPolicy
from swarm.sim import moves
from swarm.sim.episodes import make_episode
from swarm.sim.simulator import Simulator

from .helpers import random_state, state_from_cells


def tiny_config() -> Config:
    return Config(
        grid=GridConfig(height=10, width=10),
        episode=EpisodeConfig(n_modules=5, max_steps=20, target_walk_steps=5),
    )


def test_baselines_satisfy_the_policy_protocol():
    assert isinstance(RandomLegalPolicy(seed=0), Policy)
    assert isinstance(StayPolicy(), Policy)


def test_policies_only_return_legal_actions():
    """Every policy has to respect the mask, learned or not."""
    rng = np.random.default_rng(0)
    for policy in (RandomLegalPolicy(seed=0), StayPolicy()):
        for _ in range(50):
            state = random_state(rng, n_modules=int(rng.integers(1, 12)))
            mask = moves.legal_action_mask(state)
            actions = policy.act(None, mask)
            assert actions.shape == (state.n_modules,)
            assert mask[np.arange(state.n_modules), actions].all(), policy.name


def test_stay_policy_never_moves():
    mask = np.ones((4, moves.N_ACTIONS), dtype=bool)
    assert (StayPolicy().act(None, mask) == moves.STAY).all()


def test_stay_policy_leaves_the_swarm_untouched():
    cfg = tiny_config()
    spec = make_episode(cfg, np.random.default_rng(0))
    simulator = Simulator(seed=0)
    start = simulator.reset(spec).positions.copy()

    policy = StayPolicy()
    while not simulator.done:
        simulator.step(policy.act(None, simulator.legal_action_mask()))

    assert np.array_equal(simulator.state.positions, start)
    assert simulator.truncated


def test_random_policy_is_reproducible():
    mask = np.ones((6, moves.N_ACTIONS), dtype=bool)
    first = RandomLegalPolicy(seed=5)
    second = RandomLegalPolicy(seed=5)
    for _ in range(10):
        assert np.array_equal(first.act(None, mask), second.act(None, mask))


def test_random_policy_reset_reseeds():
    mask = np.ones((6, moves.N_ACTIONS), dtype=bool)
    policy = RandomLegalPolicy(seed=1)
    first = policy.act(None, mask)
    policy.reset(seed=1)
    assert np.array_equal(policy.act(None, mask), first)


def test_random_policy_actually_moves_things():
    """Counts accepted moves rather than comparing start to end positions.

    Mobility is low enough that a small swarm cycles through only a handful of
    reachable configurations, so it can easily sit back on its start cells at an
    arbitrary step count even though it moved the whole way.
    """
    cfg = tiny_config()
    spec = make_episode(cfg, np.random.default_rng(0))
    simulator = Simulator(seed=0)
    start = simulator.reset(spec).positions.copy()

    policy = RandomLegalPolicy(seed=0)
    applied = 0
    left_start = False
    while not simulator.done:
        _, info = simulator.step(policy.act(None, simulator.legal_action_mask()))
        applied += info["n_applied"]
        left_start |= not np.array_equal(simulator.state.positions, start)

    assert applied > 0
    assert left_start


def test_policies_are_torch_free():
    """The baselines and the Policy protocol must not drag in the RL stack."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; sys.path.insert(0, "src"); import swarm.rl.policies; '
            'print("torch" if "torch" in sys.modules else "")',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# -- the greedy baseline -----------------------------------------------


def observation_for(state, builder=None):
    from swarm.config import ObservationConfig
    from swarm.obs.local_window import ObservationBuilder

    builder = builder or ObservationBuilder(ObservationConfig(), state.grid_shape)
    return builder.build(state)


def run_baseline(policy, cfg, episodes=40, seed=0):
    """Roll a policy over several episodes and return its success rate."""
    from swarm.config import ObservationConfig
    from swarm.obs.local_window import ObservationBuilder

    rng = np.random.default_rng(seed)
    builder = ObservationBuilder(ObservationConfig(), cfg.grid.shape)
    simulator = Simulator(seed=seed)
    successes = 0

    for _ in range(episodes):
        spec = make_episode(cfg, rng)
        state = simulator.reset(spec)
        builder.reset(state.n_modules, rng)
        while not simulator.done:
            mask = simulator.legal_action_mask()
            state, _ = simulator.step(policy.act(builder.build(state, mask), mask))
        successes += simulator.terminated

    return successes / episodes


def test_greedy_satisfies_the_policy_protocol():
    assert isinstance(GreedyPolicy(seed=0), Policy)


def test_greedy_needs_an_observation():
    with pytest.raises(ValueError, match="needs an observation"):
        GreedyPolicy(seed=0).act(None, np.ones((2, moves.N_ACTIONS), dtype=bool))


def test_greedy_only_returns_legal_actions():
    rng = np.random.default_rng(0)
    for epsilon in (0.0, 0.3, 1.0):
        policy = GreedyPolicy(seed=0, epsilon=epsilon)
        for _ in range(30):
            state = random_state(rng, n_modules=int(rng.integers(1, 10)))
            state.target[:] = False
            state.target[0, 0] = True
            mask = moves.legal_action_mask(state)
            actions = policy.act(observation_for(state), mask)
            assert mask[np.arange(state.n_modules), actions].all(), f"epsilon={epsilon}"


def test_greedy_steps_toward_an_unfilled_target():
    # An L whose corner module needs one step right to finish the shape.
    state = state_from_cells([(2, 2), (3, 2), (3, 3)], (8, 8), target=[(2, 3), (3, 2), (3, 3)])
    actions = GreedyPolicy(seed=0).act(observation_for(state), moves.legal_action_mask(state))
    assert actions[0] == moves.RIGHT


def test_greedy_holds_a_module_that_is_already_on_target():
    """Leaving a filled cell undoes progress the team is scored on."""
    state = state_from_cells([(2, 2), (3, 2), (3, 3)], (8, 8), target=[(2, 2), (3, 2), (3, 3)])
    actions = GreedyPolicy(seed=0).act(observation_for(state), moves.legal_action_mask(state))
    assert (actions == moves.STAY).all()


def test_greedy_stays_when_the_target_is_complete():
    state = state_from_cells([(2, 2), (3, 2)], (8, 8), target=[(2, 2), (3, 2)])
    obs = observation_for(state)
    assert obs["aux"]["n_unfilled"] == 0
    actions = GreedyPolicy(seed=0).act(obs, moves.legal_action_mask(state))
    assert (actions == moves.STAY).all()


def test_greedy_deadlocks_without_an_escape():
    """Pure greedy freezes: the move a module wants is usually illegal.

    Recorded because it is the reason epsilon exists, and because the naive
    heuristic looks reasonable right up until it is measured.
    """
    cfg = Config(
        grid=GridConfig(height=16, width=16),
        episode=EpisodeConfig(n_modules=9, max_steps=120, target_walk_steps=8),
    )
    from swarm.config import ObservationConfig
    from swarm.obs.local_window import ObservationBuilder

    rng = np.random.default_rng(0)
    builder = ObservationBuilder(ObservationConfig(), cfg.grid.shape)
    simulator = Simulator(seed=0)
    policy = GreedyPolicy(seed=0, epsilon=0.0)

    frozen = total = 0
    for _ in range(10):
        spec = make_episode(cfg, rng)
        state = simulator.reset(spec)
        while not simulator.done:
            mask = simulator.legal_action_mask()
            state, info = simulator.step(policy.act(builder.build(state, mask), mask))
            frozen += info["n_applied"] == 0
            total += 1

    assert frozen / total > 0.8, "pure greedy is expected to be frozen on most steps"


def test_epsilon_unfreezes_greedy():
    cfg = Config(
        grid=GridConfig(height=16, width=16),
        episode=EpisodeConfig(n_modules=9, max_steps=200, target_walk_steps=8),
    )
    pure = run_baseline(GreedyPolicy(seed=0, epsilon=0.0), cfg)
    escaping = run_baseline(GreedyPolicy(seed=0, epsilon=0.3), cfg)
    assert escaping > pure + 0.3, f"epsilon=0.3 ({escaping}) should clear pure greedy ({pure})"


def test_greedy_with_escape_beats_random():
    """The bar the learned policy has to clear."""
    cfg = Config(
        grid=GridConfig(height=16, width=16),
        episode=EpisodeConfig(n_modules=9, max_steps=200, target_walk_steps=8),
    )
    greedy = run_baseline(GreedyPolicy(seed=0, epsilon=0.3), cfg)
    random_legal = run_baseline(RandomLegalPolicy(seed=0), cfg)
    assert greedy > random_legal


def test_epsilon_one_is_equivalent_to_random():
    """A sanity check on the escape: all-exploration greedy is the random policy."""
    cfg = Config(
        grid=GridConfig(height=16, width=16),
        episode=EpisodeConfig(n_modules=9, max_steps=200, target_walk_steps=8),
    )
    saturated = run_baseline(GreedyPolicy(seed=0, epsilon=1.0), cfg)
    random_legal = run_baseline(RandomLegalPolicy(seed=0), cfg)
    assert abs(saturated - random_legal) < 0.25


def test_strict_greedy_matches_plain_greedy_in_practice():
    """Both freeze for the same reason: a sideways step scores worse than staying."""
    cfg = Config(
        grid=GridConfig(height=16, width=16),
        episode=EpisodeConfig(n_modules=9, max_steps=120, target_walk_steps=8),
    )
    strict = run_baseline(GreedyPolicy(seed=0, strict=True), cfg, episodes=20)
    plain = run_baseline(GreedyPolicy(seed=0, strict=False), cfg, episodes=20)
    assert abs(strict - plain) < 0.2
