"""Episode construction and the curriculum schedule.

The target shape is the start shape after K random legal moves under the same
dynamics the policy will face. That is what makes an episode solvable by
construction: two independently sampled connected shapes of equal size are not
guaranteed to be connectable while the swarm must stay connected throughout, so
sampling them independently would put an unknown ceiling on success rate that
looks exactly like a learning failure. K doubles as the difficulty dial and as
an upper bound on the optimal number of steps.

Independently sampled start/target pairs are still available through
:func:`make_unpaired_episode` for held-out evaluation, where "unsolvable" is a
legitimate outcome to report.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from swarm.config import Config, CurriculumConfig
from swarm.sim import moves
from swarm.sim.shapes import library_shape, place, random_connected_shape
from swarm.sim.state import SwarmState


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything needed to start one episode, plus how it was generated."""

    start_positions: np.ndarray  # (N, 2) int32
    target: np.ndarray  # (H, W) bool
    grid_shape: tuple[int, int]
    max_steps: int
    walk_steps: int = 0  # K requested during generation
    walk_applied: int = 0  # moves that were actually applied
    solvable_in: int | None = None  # known step bound, or None if not guaranteed
    seed: int | None = None
    source: str = "random_walk"
    metadata: dict = field(default_factory=dict)

    @property
    def n_modules(self) -> int:
        return int(self.start_positions.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.target.sum())

    @property
    def is_degenerate(self) -> bool:
        """True when the start shape already satisfies the target.

        Happens when the generating walk could not move anything -- a rigid
        start shape such as a line or a solid block has no legal moves at all.
        """
        occupancy = np.zeros(self.grid_shape, dtype=bool)
        occupancy[self.start_positions[:, 0], self.start_positions[:, 1]] = True
        return bool((occupancy & self.target).sum() == self.n_targets)

    def initial_state(self) -> SwarmState:
        return SwarmState.from_positions(self.start_positions, self.grid_shape, self.target)


def random_walk_target(
    start_positions: np.ndarray,
    grid_shape: tuple[int, int],
    walk_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Walk the swarm forward ``walk_steps`` random legal steps.

    Returns the resulting target mask and the number of individual module moves
    that were actually applied. A rigid shape yields zero applied moves and a
    target identical to the start.
    """
    state = SwarmState.from_positions(start_positions, grid_shape)
    applied = 0

    for _ in range(walk_steps):
        mask = moves.legal_action_mask(state)
        actions = moves.sample_legal_actions(mask, rng)
        state, info = moves.apply_actions(state, actions, rng)
        applied += info["n_applied"]

    return state.occupancy.copy(), applied


def _sample_start(
    rng: np.random.Generator,
    n_modules: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Sample a start shape, seeded away from the grid edges.

    A shape grown from a corner is pressed against the walls, which cuts its
    mobility and makes a degenerate episode much more likely, so the growth seed
    is drawn from the interior when the grid is large enough to allow it.
    """
    height, width = grid_shape
    margin = int(np.ceil(np.sqrt(n_modules)))
    row_margin = margin if 2 * margin < height else 0
    col_margin = margin if 2 * margin < width else 0

    origin = (
        int(rng.integers(row_margin, height - row_margin)),
        int(rng.integers(col_margin, width - col_margin)),
    )
    return random_connected_shape(rng, n_modules, grid_shape, origin=origin)


def make_episode(
    config: Config,
    rng: np.random.Generator,
    walk_steps: int | None = None,
    max_attempts: int = 8,
) -> EpisodeSpec:
    """Build a solvable-by-construction episode.

    Resamples whenever the walk leaves the swarm covering its own target, which
    happens either because the start shape was rigid (``walk_applied == 0``) or
    because a short random walk wandered back to where it started. After
    ``max_attempts`` the degenerate episode is returned rather than raising, and
    ``walk_applied`` records which case it was.
    """
    n_modules = config.episode.n_modules
    grid_shape = config.grid.shape
    if walk_steps is None:
        walk_steps = config.episode.target_walk_steps

    start = target = None
    applied = 0
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        start = _sample_start(rng, n_modules, grid_shape)
        target, applied = random_walk_target(start, grid_shape, walk_steps, rng)
        if walk_steps == 0:
            break
        start_occupancy = np.zeros(grid_shape, dtype=bool)
        start_occupancy[start[:, 0], start[:, 1]] = True
        if applied > 0 and not np.array_equal(start_occupancy, target):
            break

    return EpisodeSpec(
        start_positions=start,
        target=target,
        grid_shape=grid_shape,
        max_steps=config.episode.max_steps,
        walk_steps=walk_steps,
        walk_applied=applied,
        solvable_in=walk_steps,
        source="random_walk",
        metadata={"attempts": attempts},
    )


def make_library_episode(
    config: Config,
    start_shape: str,
    target_shape: str,
    target_anchor: tuple[int, int] | None = None,
) -> EpisodeSpec:
    """Build an episode from two named library shapes.

    Deterministic and generally *not* known to be solvable -- this is for the
    held-out evaluation set, where failing to solve a given pair is a result
    rather than a bug.
    """
    n_modules = config.episode.n_modules
    grid_shape = config.grid.shape

    start = library_shape(start_shape, n_modules, grid_shape)
    target_cells = library_shape(target_shape, n_modules, grid_shape)
    if target_anchor is not None:
        target_cells = place(target_cells, grid_shape, target_anchor)

    target = np.zeros(grid_shape, dtype=bool)
    target[target_cells[:, 0], target_cells[:, 1]] = True

    return EpisodeSpec(
        start_positions=start,
        target=target,
        grid_shape=grid_shape,
        max_steps=config.episode.max_steps,
        solvable_in=None,
        source="library",
        metadata={"start_shape": start_shape, "target_shape": target_shape},
    )


def make_unpaired_episode(config: Config, rng: np.random.Generator) -> EpisodeSpec:
    """Sample start and target shapes independently.

    Harder than :func:`make_episode` and with no solvability guarantee; kept for
    evaluation so that generalization beyond walk-generated targets can be
    measured honestly.
    """
    n_modules = config.episode.n_modules
    grid_shape = config.grid.shape

    start = _sample_start(rng, n_modules, grid_shape)
    target_cells = _sample_start(rng, n_modules, grid_shape)
    target = np.zeros(grid_shape, dtype=bool)
    target[target_cells[:, 0], target_cells[:, 1]] = True

    return EpisodeSpec(
        start_positions=start,
        target=target,
        grid_shape=grid_shape,
        max_steps=config.episode.max_steps,
        solvable_in=None,
        source="unpaired",
    )


class Curriculum:
    """Advances ``target_walk_steps`` once a rolling success rate is reached.

    Promotion clears the window, so the next level has to earn its own
    evidence rather than coasting on the previous level's successes.
    """

    def __init__(self, config: CurriculumConfig, fallback_walk_steps: int):
        self.config = config
        self.fallback_walk_steps = fallback_walk_steps
        self.level_index = 0
        self.episodes_seen = 0
        self.promotions = 0
        self._window: deque[bool] = deque(maxlen=config.window_episodes)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def walk_steps(self) -> int:
        """The K to generate the next episode with."""
        if not self.config.enabled:
            return self.fallback_walk_steps
        return int(self.config.levels[self.level_index])

    @property
    def at_final_level(self) -> bool:
        return self.level_index >= len(self.config.levels) - 1

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._window)) if self._window else 0.0

    def record(self, success: bool) -> bool:
        """Record an episode outcome. Returns True if this triggered a promotion."""
        self.episodes_seen += 1
        if not self.config.enabled:
            return False

        self._window.append(bool(success))
        if len(self._window) < self.config.window_episodes:
            return False
        if self.success_rate < self.config.promote_success_rate:
            return False
        if self.at_final_level:
            return False

        self.level_index += 1
        self.promotions += 1
        self._window.clear()
        return True

    def stats(self) -> dict:
        return {
            "curriculum/level": self.level_index,
            "curriculum/walk_steps": self.walk_steps,
            "curriculum/success_rate": self.success_rate,
            "curriculum/window_size": len(self._window),
            "curriculum/promotions": self.promotions,
            "curriculum/episodes_seen": self.episodes_seen,
        }
