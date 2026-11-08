import pytest
import yaml

from swarm import config as config_module
from swarm.config import Config, ConfigError

CONFIG_DIR = "configs"


def test_default_config_loads_and_validates():
    cfg = config_module.load(f"{CONFIG_DIR}/default.yaml")
    assert cfg.grid.shape == (cfg.grid.height, cfg.grid.width)
    assert cfg.episode.n_modules >= 1
    assert cfg.observation.k_nearest_targets == 3
    assert cfg.observation.window % 2 == 1


def test_debug_config_inherits_and_overrides():
    default = config_module.load(f"{CONFIG_DIR}/default.yaml")
    tiny = config_module.load(f"{CONFIG_DIR}/debug_tiny.yaml")

    assert tiny.grid.shape == (8, 8)
    assert tiny.episode.n_modules == 4
    assert not tiny.curriculum.enabled
    # Inherited unchanged from the parent file.
    assert tiny.observation.k_nearest_targets == default.observation.k_nearest_targets
    assert tiny.reward.name == default.reward.name


def test_step_penalty_scales_with_episode_length():
    """The time cost is a share of the episode budget, not a flat constant.

    The fill term can only sum to fill_scale over an episode, so a flat penalty
    would come to dominate the signal as max_steps grows.
    """
    cfg = config_module.load(f"{CONFIG_DIR}/default.yaml")
    penalty = cfg.reward.step_penalty(cfg.episode.max_steps)
    assert penalty * cfg.episode.max_steps == pytest.approx(cfg.reward.time_cost_total)
    assert cfg.reward.step_penalty(1000) < cfg.reward.step_penalty(100)


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown keys in section 'grid'"):
        config_module.from_dict({"grid": {"heigth": 10}})


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown top-level config keys"):
        config_module.from_dict({"gird": {}})


@pytest.mark.parametrize(
    "raw,match",
    [
        ({"grid": {"height": 1}}, "at least 2x2"),
        ({"episode": {"n_modules": 0}}, "n_modules must be >= 1"),
        ({"episode": {"max_steps": 0}}, "max_steps must be >= 1"),
        ({"episode": {"target_walk_steps": -1}}, "target_walk_steps must be >= 0"),
        ({"observation": {"window": 4}}, "must be odd"),
        ({"observation": {"window": 1}}, "must be odd and >= 3"),
        ({"observation": {"k_nearest_targets": 0}}, "k_nearest_targets must be >= 1"),
        ({"reward": {"time_cost_total": -1}}, "time_cost_total must be >= 0"),
        ({"curriculum": {"promote_success_rate": 0}}, "promote_success_rate"),
        ({"curriculum": {"levels": []}}, "levels must be non-empty"),
        ({"curriculum": {"window_episodes": 0}}, "window_episodes must be >= 1"),
    ],
)
def test_validation_rejects_bad_values(raw, match):
    with pytest.raises(ConfigError, match=match):
        config_module.from_dict(raw)


def test_module_count_cannot_exceed_grid():
    with pytest.raises(ConfigError, match="exceeds grid cells"):
        config_module.from_dict({"grid": {"height": 3, "width": 3}, "episode": {"n_modules": 10}})


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="config file not found"):
        config_module.load(f"{CONFIG_DIR}/does_not_exist.yaml")


def test_circular_inheritance_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"inherits": "b.yaml"}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"inherits": "a.yaml"}))
    with pytest.raises(ConfigError, match="circular config inheritance"):
        config_module.load(tmp_path / "a.yaml")


def test_defaults_are_valid_on_their_own():
    Config().validate()
