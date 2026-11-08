import numpy as np

from swarm.config import Config, EpisodeConfig, GridConfig
from swarm.rl.policies import Policy, RandomLegalPolicy, StayPolicy
from swarm.sim import moves
from swarm.sim.episodes import make_episode
from swarm.sim.simulator import Simulator

from .helpers import random_state


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
