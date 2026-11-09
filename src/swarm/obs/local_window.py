"""Per-module observations.

Each module sees a small egocentric window, the k nearest unfilled target cells,
whether it is standing on a target cell, and its own action mask. The output is
a dict of batched arrays with the module axis first, so nothing is stored per
module as a Python object.

The k in "k nearest" matters: with only the single nearest unfilled cell, every
module near the same gap chases it, and because one policy is shared by all
modules there is nothing in the observation to break the tie. Reporting several
candidates at least gives the policy the material to spread out.

The ``aux`` entry carries raw integer deltas for the greedy baseline and for
debugging. It is deliberately *not* part of the policy input: it is unnormalized
and unbounded, and feeding it to a network would be a second, redundant encoding
of the same information already in ``vector``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm.config import ObservationConfig
from swarm.sim import moves
from swarm.sim.state import SwarmState

N_CHANNELS = 3
CHANNEL_OCCUPANCY, CHANNEL_TARGET, CHANNEL_OUT_OF_BOUNDS = range(N_CHANNELS)
CHANNEL_NAMES = ("occupancy", "target", "out_of_bounds")

# Per candidate target cell: a direction split across the two axes, the
# normalized distance, and a validity flag for when fewer than k cells remain.
SLOT_FIELDS = ("delta_row", "delta_col", "distance", "valid")


@dataclass(frozen=True)
class ObservationSpec:
    """Shapes and field layout of an observation, derived from the config.

    The environment builds its observation spaces from this, the network reads
    its input sizes from it, and tests assert against it, so the layout is
    stated once rather than implied in three places.
    """

    window: int
    n_channels: int
    vector_fields: tuple[tuple[str, int], ...]

    @property
    def window_shape(self) -> tuple[int, int, int]:
        return (self.n_channels, self.window, self.window)

    @property
    def vector_size(self) -> int:
        return sum(size for _, size in self.vector_fields)

    @property
    def n_actions(self) -> int:
        return moves.N_ACTIONS

    def field_slice(self, name: str) -> slice:
        offset = 0
        for field_name, size in self.vector_fields:
            if field_name == name:
                return slice(offset, offset + size)
            offset += size
        raise KeyError(f"unknown observation field '{name}'; have {self.field_names}")

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.vector_fields)


def build_spec(config: ObservationConfig) -> ObservationSpec:
    fields: list[tuple[str, int]] = [
        (f"target_{index}", len(SLOT_FIELDS)) for index in range(config.k_nearest_targets)
    ]
    fields.append(("on_target", 1))
    if config.use_relative_position:
        fields.append(("relative_position", 2))
    if config.use_absolute_position:
        fields.append(("absolute_position", 2))
    if config.use_module_tag:
        fields.append(("module_tag", 1))
    if config.use_global_map:
        fields.append(("global_map", config.global_map_size**2))
    return ObservationSpec(
        window=config.window, n_channels=N_CHANNELS, vector_fields=tuple(fields)
    )


def _windows(state: SwarmState, window: int) -> np.ndarray:
    """(N, C, window, window) egocentric crops, zero-padded past the grid edge.

    Out-of-bounds cells get their own channel: a wall and an empty cell are
    different things to a module deciding where it can go.
    """
    radius = window // 2
    pad = ((radius, radius), (radius, radius))

    occupancy = np.pad(state.occupancy, pad, constant_values=False)
    target = np.pad(state.target, pad, constant_values=False)
    in_bounds = np.pad(np.ones(state.grid_shape, dtype=bool), pad, constant_values=False)

    offsets = np.arange(window)
    rows = state.positions[:, 0, None, None] + offsets[None, :, None]
    cols = state.positions[:, 1, None, None] + offsets[None, None, :]

    stacked = np.empty((state.n_modules, N_CHANNELS, window, window), dtype=np.float32)
    stacked[:, CHANNEL_OCCUPANCY] = occupancy[rows, cols]
    stacked[:, CHANNEL_TARGET] = target[rows, cols]
    stacked[:, CHANNEL_OUT_OF_BOUNDS] = ~in_bounds[rows, cols]
    return stacked


def _nearest_unfilled(state: SwarmState, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k nearest unfilled target cells per module.

    Returns raw ``(N, k, 2)`` integer deltas and an ``(N, k)`` validity mask.
    Distance is Manhattan, which is the true lower bound on move count under
    4-neighbour movement, so the ordering means something rather than being a
    Euclidean approximation of it.
    """
    n_modules = state.n_modules
    deltas = np.zeros((n_modules, k, 2), dtype=np.int32)
    valid = np.zeros((n_modules, k), dtype=bool)

    unfilled_rows, unfilled_cols = np.nonzero(state.unfilled_mask)
    n_unfilled = unfilled_rows.shape[0]
    if n_unfilled == 0:
        return deltas, valid

    # (N, U) Manhattan distances from every module to every unfilled target.
    row_delta = unfilled_rows[None, :] - state.positions[:, 0, None]
    col_delta = unfilled_cols[None, :] - state.positions[:, 1, None]
    distance = np.abs(row_delta) + np.abs(col_delta)

    take = min(k, n_unfilled)
    nearest = np.argpartition(distance, take - 1, axis=1)[:, :take]
    # argpartition leaves the chosen block unordered; sort so slot 0 is nearest.
    order = np.argsort(np.take_along_axis(distance, nearest, axis=1), axis=1)
    nearest = np.take_along_axis(nearest, order, axis=1)

    module_index = np.arange(n_modules)[:, None]
    deltas[:, :take, 0] = row_delta[module_index, nearest]
    deltas[:, :take, 1] = col_delta[module_index, nearest]
    valid[:, :take] = True
    return deltas, valid


def _pooled_global_map(state: SwarmState, size: int) -> np.ndarray:
    """Coarse occupancy-minus-target map, average-pooled to ``size`` x ``size``.

    Lets a module sense that its own region is overcrowded while another is
    short of modules. Identical for every module, so it is broadcast rather
    than computed per module.
    """
    height, width = state.grid_shape
    row_edges = np.linspace(0, height, size + 1).astype(int)
    col_edges = np.linspace(0, width, size + 1).astype(int)

    difference = state.occupancy.astype(np.float32) - state.target.astype(np.float32)
    pooled = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            block = difference[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            pooled[i, j] = block.mean() if block.size else 0.0
    return pooled.reshape(-1)


@dataclass
class ObservationBuilder:
    """Turns a state into per-module observation arrays.

    Holds the per-episode module tags when they are enabled; otherwise it is
    stateless and ``build`` is a pure function of the state.
    """

    config: ObservationConfig
    grid_shape: tuple[int, int]
    spec: ObservationSpec = field(init=False)
    _tags: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.spec = build_spec(self.config)

    def reset(self, n_modules: int, rng: np.random.Generator | None = None) -> None:
        """Draw fresh module tags for a new episode."""
        if not self.config.use_module_tag:
            self._tags = None
            return
        generator = rng if rng is not None else np.random.default_rng()
        self._tags = generator.random(n_modules).astype(np.float32)

    def build(self, state: SwarmState, mask: np.ndarray | None = None) -> dict[str, Any]:
        """Build the observation bundle for every module.

        ``mask`` may be passed in when the caller already computed it, which
        avoids running the connectivity checks twice per step.
        """
        if mask is None:
            mask = moves.legal_action_mask(state)

        n_modules = state.n_modules
        height, width = state.grid_shape
        k = self.config.k_nearest_targets

        on_target = state.on_target()
        deltas, valid = _nearest_unfilled(state, k)
        distance = np.abs(deltas).sum(axis=2).astype(np.float32)

        # Split the direction across the two axes so the pair is bounded in
        # [-1, 1] and |row| + |col| == 1 for any valid slot.
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = np.where(
                distance[:, :, None] > 0, deltas / distance[:, :, None], 0.0
            ).astype(np.float32)

        scale = np.float32(height + width)
        slots = np.concatenate(
            [
                direction,
                (distance / scale)[:, :, None],
                valid.astype(np.float32)[:, :, None],
            ],
            axis=2,
        )

        pieces: list[np.ndarray] = [slots.reshape(n_modules, -1)]
        pieces.append(on_target.astype(np.float32)[:, None])

        if self.config.use_relative_position:
            centroid = state.positions.mean(axis=0)
            relative = (state.positions - centroid) / np.array([height, width], dtype=np.float32)
            pieces.append(relative.astype(np.float32))

        if self.config.use_absolute_position:
            absolute = state.positions / np.array([height, width], dtype=np.float32)
            pieces.append(absolute.astype(np.float32))

        if self.config.use_module_tag:
            if self._tags is None or self._tags.shape[0] != n_modules:
                raise RuntimeError("reset() must be called before building tagged observations")
            pieces.append(self._tags[:, None])

        if self.config.use_global_map:
            pooled = _pooled_global_map(state, self.config.global_map_size)
            pieces.append(np.broadcast_to(pooled, (n_modules, pooled.shape[0])))

        vector = np.concatenate(pieces, axis=1).astype(np.float32)
        if vector.shape[1] != self.spec.vector_size:
            raise AssertionError(
                f"vector width {vector.shape[1]} does not match spec {self.spec.vector_size}"
            )

        return {
            "window": _windows(state, self.config.window),
            "vector": vector,
            "mask": mask,
            "aux": {
                "nearest_delta": deltas,
                "nearest_valid": valid,
                "on_target": on_target,
                "n_unfilled": int(state.unfilled_mask.sum()),
            },
        }
