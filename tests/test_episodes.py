import numpy as np
import pytest

from swarm import config as config_module
from swarm.config import Config, CurriculumConfig, EpisodeConfig, GridConfig
from swarm.sim.connectivity import is_connected
from swarm.sim.episodes import (
    Curriculum,
    EpisodeSpec,
    make_episode,
    make_library_episode,
    make_unpaired_episode,
    random_walk_target,
)
from swarm.sim.shapes import library_shape, random_connected_shape


def tiny_config(**overrides) -> Config:
    base = {
        "grid": GridConfig(height=10, width=10),
        "episode": EpisodeConfig(n_modules=6, max_steps=50, target_walk_steps=6),
    }
    return Config(**{**base, **overrides})


# -- the solvability guarantee -----------------------------------------


def test_target_is_the_start_shape_after_k_moves():
    """The walk is the solvability guarantee, so the target must be reachable.

    Replaying the generating walk from the same seed has to land exactly on the
    target mask.
    """
    rng_seed = 11
    start = random_connected_shape(np.random.default_rng(0), 8, (12, 12), origin=(6, 6))

    first, applied_a = random_walk_target(start, (12, 12), 10, np.random.default_rng(rng_seed))
    second, applied_b = random_walk_target(start, (12, 12), 10, np.random.default_rng(rng_seed))

    assert np.array_equal(first, second)
    assert applied_a == applied_b


def test_target_preserves_module_count_and_connectivity():
    rng = np.random.default_rng(1)
    for _ in range(50):
        start = random_connected_shape(rng, 9, (12, 12), origin=(6, 6))
        target, _ = random_walk_target(start, (12, 12), 12, rng)
        assert int(target.sum()) == 9, "a walk must not create or destroy modules"
        assert is_connected(target)


def test_zero_walk_steps_gives_the_start_shape():
    start = random_connected_shape(np.random.default_rng(0), 5, (10, 10), origin=(5, 5))
    target, applied = random_walk_target(start, (10, 10), 0, np.random.default_rng(0))
    assert applied == 0
    occupancy = np.zeros((10, 10), dtype=bool)
    occupancy[start[:, 0], start[:, 1]] = True
    assert np.array_equal(target, occupancy)


def test_rigid_start_shape_cannot_walk():
    """A solid block has no legal moves, so its walk applies nothing."""
    block = library_shape("square", 4, (10, 10))
    target, applied = random_walk_target(block, (10, 10), 25, np.random.default_rng(0))
    assert applied == 0
    occupancy = np.zeros((10, 10), dtype=bool)
    occupancy[block[:, 0], block[:, 1]] = True
    assert np.array_equal(target, occupancy)


# -- episode construction ----------------------------------------------


def test_make_episode_is_well_formed():
    cfg = tiny_config()
    rng = np.random.default_rng(0)
    for _ in range(30):
        spec = make_episode(cfg, rng)
        assert spec.n_modules == cfg.episode.n_modules
        assert spec.n_targets == cfg.episode.n_modules
        assert spec.grid_shape == cfg.grid.shape
        assert spec.max_steps == cfg.episode.max_steps
        assert spec.solvable_in == cfg.episode.target_walk_steps
        assert spec.source == "random_walk"

        state = spec.initial_state()
        state.validate()
        assert is_connected(state.occupancy)
        assert is_connected(spec.target)


def test_make_episode_avoids_degenerate_episodes():
    """An episode whose target is already satisfied teaches nothing.

    It happens when the start shape is rigid or when a short walk wanders back
    to where it began, so generation resamples instead of emitting it.
    """
    cfg = tiny_config()
    rng = np.random.default_rng(2)
    specs = [make_episode(cfg, rng) for _ in range(100)]
    degenerate = sum(spec.is_degenerate for spec in specs)
    assert degenerate == 0, f"{degenerate}/100 episodes started already solved"


def test_make_episode_respects_a_walk_steps_override():
    cfg = tiny_config()
    spec = make_episode(cfg, np.random.default_rng(0), walk_steps=3)
    assert spec.walk_steps == 3
    assert spec.solvable_in == 3


def test_make_episode_is_reproducible_from_a_seed():
    cfg = tiny_config()
    first = make_episode(cfg, np.random.default_rng(7))
    second = make_episode(cfg, np.random.default_rng(7))
    assert np.array_equal(first.start_positions, second.start_positions)
    assert np.array_equal(first.target, second.target)


def test_make_episode_tolerates_a_rigid_configuration():
    """With one module nothing can be rigid; with a full grid everything is.

    A 2x2 grid holding 4 modules is a solid block, so generation cannot avoid a
    degenerate episode and must return one rather than loop or raise.
    """
    cfg = Config(
        grid=GridConfig(height=2, width=2),
        episode=EpisodeConfig(n_modules=4, max_steps=10, target_walk_steps=5),
    )
    spec = make_episode(cfg, np.random.default_rng(0))
    assert spec.walk_applied == 0
    assert spec.is_degenerate
    assert spec.metadata["attempts"] == 8


def test_start_shapes_are_seeded_away_from_the_edges():
    """Shapes grown from a corner are pressed against the walls and lose mobility."""
    cfg = tiny_config(episode=EpisodeConfig(n_modules=4, max_steps=50, target_walk_steps=4))
    rng = np.random.default_rng(0)
    starts = [make_episode(cfg, rng).start_positions for _ in range(50)]
    seeds_touching_edge = sum(
        bool((s.min() == 0) or (s[:, 0].max() == 9) or (s[:, 1].max() == 9)) for s in starts
    )
    assert seeds_touching_edge < len(starts) // 2


# -- evaluation episode sources ----------------------------------------


def test_library_episode_is_deterministic_and_unguaranteed():
    cfg = tiny_config(episode=EpisodeConfig(n_modules=4, max_steps=50))
    spec = make_library_episode(cfg, "square", "line_h")
    assert spec.source == "library"
    assert spec.solvable_in is None, "library pairs carry no solvability guarantee"
    assert spec.metadata == {"start_shape": "square", "target_shape": "line_h"}

    again = make_library_episode(cfg, "square", "line_h")
    assert np.array_equal(spec.start_positions, again.start_positions)
    assert np.array_equal(spec.target, again.target)


def test_library_episode_honours_a_target_anchor():
    cfg = tiny_config(episode=EpisodeConfig(n_modules=3, max_steps=50))
    spec = make_library_episode(cfg, "line_h", "line_v", target_anchor=(0, 0))
    rows, cols = np.nonzero(spec.target)
    assert rows.min() == 0 and cols.min() == 0


def test_unpaired_episode_is_well_formed():
    cfg = tiny_config()
    rng = np.random.default_rng(0)
    spec = make_unpaired_episode(cfg, rng)
    assert spec.source == "unpaired"
    assert spec.solvable_in is None
    assert spec.n_targets == cfg.episode.n_modules
    assert is_connected(spec.target)
    spec.initial_state().validate()


def test_episode_spec_detects_a_degenerate_target():
    start = np.array([(1, 1), (1, 2)], dtype=np.int32)
    target = np.zeros((6, 6), dtype=bool)
    target[1, 1] = target[1, 2] = True
    spec = EpisodeSpec(start, target, (6, 6), max_steps=10)
    assert spec.is_degenerate

    shifted = np.zeros((6, 6), dtype=bool)
    shifted[2, 1] = shifted[2, 2] = True
    assert not EpisodeSpec(start, shifted, (6, 6), max_steps=10).is_degenerate


# -- curriculum --------------------------------------------------------


def curriculum(**overrides) -> Curriculum:
    base = {"enabled": True, "levels": (2, 4, 8), "promote_success_rate": 0.8, "window_episodes": 5}
    return Curriculum(CurriculumConfig(**{**base, **overrides}), fallback_walk_steps=99)


def test_curriculum_starts_at_the_first_level():
    c = curriculum()
    assert c.walk_steps == 2
    assert c.level_index == 0


def test_curriculum_promotes_once_the_window_is_good_enough():
    c = curriculum()
    promoted = [c.record(True) for _ in range(5)]
    assert promoted == [False, False, False, False, True]
    assert c.walk_steps == 4
    assert c.promotions == 1


def test_curriculum_does_not_promote_below_the_threshold():
    c = curriculum()
    outcomes = [True, True, True, False, False]  # 60% < 80%
    assert not any(c.record(o) for o in outcomes)
    assert c.walk_steps == 2


def test_curriculum_window_is_cleared_on_promotion():
    """A new level must earn its own evidence, not inherit the old level's."""
    c = curriculum()
    for _ in range(5):
        c.record(True)
    assert c.success_rate == 0.0
    assert len(c._window) == 0


def test_curriculum_stops_at_the_final_level():
    c = curriculum()
    for _ in range(100):
        c.record(True)
    assert c.at_final_level
    assert c.walk_steps == 8


def test_disabled_curriculum_uses_the_configured_walk_steps():
    c = curriculum(enabled=False)
    assert c.walk_steps == 99
    assert not c.record(True)
    assert c.episodes_seen == 1


def test_curriculum_stats_are_loggable():
    c = curriculum()
    c.record(True)
    stats = c.stats()
    assert stats["curriculum/level"] == 0
    assert stats["curriculum/walk_steps"] == 2
    assert stats["curriculum/episodes_seen"] == 1
    assert 0.0 <= stats["curriculum/success_rate"] <= 1.0


def test_curriculum_tracks_the_default_config():
    cfg = config_module.load("configs/default.yaml")
    c = Curriculum(cfg.curriculum, cfg.episode.target_walk_steps)
    assert c.walk_steps == cfg.curriculum.levels[0]


@pytest.mark.parametrize("n_modules", [1, 2, 3])
def test_tiny_swarms_still_produce_valid_episodes(n_modules):
    cfg = tiny_config(
        episode=EpisodeConfig(n_modules=n_modules, max_steps=20, target_walk_steps=4)
    )
    spec = make_episode(cfg, np.random.default_rng(0))
    spec.initial_state().validate()
    assert spec.n_targets == n_modules
