"""Episode lifecycle around the movement rules.

The simulator owns the current state, the move-resolution RNG, and termination.
It deliberately knows nothing about observations, rewards, or agents: those are
layered on by :mod:`swarm.obs`, :mod:`swarm.reward`, and :mod:`swarm.env`, which
keeps the rules testable and batchable without the RL stack installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from swarm.sim import moves
from swarm.sim.episodes import EpisodeSpec
from swarm.sim.state import SwarmState


class Simulator:
    """Steps a swarm through one episode at a time."""

    def __init__(self, seed: int | None = None, validate: bool = False):
        """``validate`` re-checks every state invariant each step; use it in tests."""
        self.rng = np.random.default_rng(seed)
        self.validate = validate
        self._spec: EpisodeSpec | None = None
        self._state: SwarmState | None = None

    # -- lifecycle -------------------------------------------------------

    def reset(self, spec: EpisodeSpec, seed: int | None = None) -> SwarmState:
        """Start a new episode. ``seed`` reseeds move-order resolution."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._spec = spec
        self._state = spec.initial_state()
        if self.validate:
            self._state.validate()
        return self._state

    def step(self, actions: np.ndarray) -> tuple[SwarmState, dict[str, Any]]:
        """Apply one simultaneous action per module and report what happened."""
        previous = self.state
        new_state, move_info = moves.apply_actions(
            previous, actions, self.rng, validate=self.validate
        )
        self._state = new_state

        n_filled = new_state.n_filled
        info: dict[str, Any] = {
            **move_info,
            "step": new_state.step,
            "n_filled": n_filled,
            "newly_filled": n_filled - previous.n_filled,
            "n_targets": new_state.n_targets,
            "fill_fraction": new_state.fill_fraction,
            "complete": new_state.is_complete,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        return new_state, info

    # -- accessors -------------------------------------------------------

    @property
    def spec(self) -> EpisodeSpec:
        if self._spec is None:
            raise RuntimeError("reset() must be called before using the simulator")
        return self._spec

    @property
    def state(self) -> SwarmState:
        if self._state is None:
            raise RuntimeError("reset() must be called before using the simulator")
        return self._state

    def legal_action_mask(self) -> np.ndarray:
        """(N, 5) bool mask for the current state, straight from the rules."""
        return moves.legal_action_mask(self.state)

    @property
    def terminated(self) -> bool:
        """The target shape is complete."""
        return self.state.is_complete

    @property
    def truncated(self) -> bool:
        """The step budget ran out without completing the target."""
        return not self.terminated and self.state.step >= self.spec.max_steps

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
