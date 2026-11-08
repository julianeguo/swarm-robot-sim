"""Configuration loading and validation.

A config is a YAML file of nested sections that maps onto the dataclasses below.
A file may declare ``inherits: <other.yaml>`` (resolved relative to its own
directory) and override only the keys it changes. Validation happens at load
time so that a bad config fails before a training run starts rather than
hundreds of steps in.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is structurally or semantically invalid."""


@dataclass(frozen=True)
class GridConfig:
    height: int = 24
    width: int = 24

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def n_cells(self) -> int:
        return self.height * self.width

    def validate(self) -> None:
        if self.height < 2 or self.width < 2:
            raise ConfigError(f"grid must be at least 2x2, got {self.height}x{self.width}")


@dataclass(frozen=True)
class EpisodeConfig:
    n_modules: int = 16
    max_steps: int = 200
    target_walk_steps: int = 8

    def validate(self) -> None:
        if self.n_modules < 1:
            raise ConfigError(f"episode.n_modules must be >= 1, got {self.n_modules}")
        if self.max_steps < 1:
            raise ConfigError(f"episode.max_steps must be >= 1, got {self.max_steps}")
        if self.target_walk_steps < 0:
            raise ConfigError(
                f"episode.target_walk_steps must be >= 0, got {self.target_walk_steps}"
            )


@dataclass(frozen=True)
class ObservationConfig:
    window: int = 5
    k_nearest_targets: int = 3
    use_relative_position: bool = True
    use_absolute_position: bool = False
    use_module_tag: bool = False
    use_global_map: bool = False
    global_map_size: int = 10

    def validate(self) -> None:
        if self.window < 3 or self.window % 2 == 0:
            raise ConfigError(f"observation.window must be odd and >= 3, got {self.window}")
        if self.k_nearest_targets < 1:
            raise ConfigError(
                f"observation.k_nearest_targets must be >= 1, got {self.k_nearest_targets}"
            )
        if self.use_global_map and self.global_map_size < 2:
            raise ConfigError(
                f"observation.global_map_size must be >= 2, got {self.global_map_size}"
            )


@dataclass(frozen=True)
class RewardConfig:
    name: str = "team_fill"
    fill_scale: float = 1.0
    time_cost_total: float = 0.5
    completion_bonus: float = 1.0

    def validate(self) -> None:
        if self.time_cost_total < 0:
            raise ConfigError(f"reward.time_cost_total must be >= 0, got {self.time_cost_total}")

    def step_penalty(self, max_steps: int) -> float:
        """Per-step time cost.

        Expressed as a fraction of an episode's total budget rather than a flat
        constant: the fill term can only ever sum to ``fill_scale`` over a whole
        episode, so a flat penalty would swamp it as ``max_steps`` grows.
        """
        return self.time_cost_total / max_steps


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = True
    levels: tuple[int, ...] = (2, 4, 8, 16, 32)
    promote_success_rate: float = 0.8
    window_episodes: int = 100

    def validate(self) -> None:
        if self.enabled and not self.levels:
            raise ConfigError("curriculum.levels must be non-empty when curriculum is enabled")
        if any(level < 0 for level in self.levels):
            raise ConfigError(f"curriculum.levels must all be >= 0, got {self.levels}")
        if not 0.0 < self.promote_success_rate <= 1.0:
            raise ConfigError(
                f"curriculum.promote_success_rate must be in (0, 1], "
                f"got {self.promote_success_rate}"
            )
        if self.window_episodes < 1:
            raise ConfigError(
                f"curriculum.window_episodes must be >= 1, got {self.window_episodes}"
            )


@dataclass(frozen=True)
class Config:
    seed: int = 0
    grid: GridConfig = field(default_factory=GridConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    train: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for f in fields(self):
            section = getattr(self, f.name)
            if is_dataclass(section):
                section.validate()
        if self.episode.n_modules > self.grid.n_cells:
            raise ConfigError(
                f"episode.n_modules ({self.episode.n_modules}) exceeds grid cells "
                f"({self.grid.n_cells})"
            )


_SECTIONS = {
    "grid": GridConfig,
    "episode": EpisodeConfig,
    "observation": ObservationConfig,
    "reward": RewardConfig,
    "curriculum": CurriculumConfig,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ConfigError(f"circular config inheritance: {chain}")
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {path}")

    parent_name = raw.pop("inherits", None)
    if parent_name is None:
        return raw
    parent = _load_raw(path.parent / parent_name, (*_seen, path))
    return _deep_merge(parent, raw)


def from_dict(raw: dict[str, Any]) -> Config:
    """Build and validate a Config from an already-merged mapping."""
    raw = dict(raw)
    kwargs: dict[str, Any] = {}

    for name, cls in _SECTIONS.items():
        section = raw.pop(name, {}) or {}
        if not isinstance(section, dict):
            raise ConfigError(f"config section '{name}' must be a mapping, got {type(section)}")
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ConfigError(f"unknown keys in section '{name}': {sorted(unknown)}")
        if cls is CurriculumConfig and "levels" in section:
            section = {**section, "levels": tuple(section["levels"])}
        kwargs[name] = cls(**section)

    if "seed" in raw:
        kwargs["seed"] = int(raw.pop("seed"))
    kwargs["train"] = raw.pop("train", {}) or {}

    if raw:
        raise ConfigError(f"unknown top-level config keys: {sorted(raw)}")

    config = Config(**kwargs)
    config.validate()
    return config


def load(path: str | Path) -> Config:
    """Load, merge, and validate a config file."""
    return from_dict(_load_raw(Path(path)))
