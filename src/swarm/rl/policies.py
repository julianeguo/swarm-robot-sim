"""Policy interface and the non-learned baselines.

Every policy -- random, greedy, or a trained checkpoint -- satisfies the same
interface, so the rollout, evaluation, and viewer scripts work with any of them
and only training needs torch. This module is deliberately torch-free; the
checkpoint policy imports it lazily.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from swarm.sim import moves


@runtime_checkable
class Policy(Protocol):
    """Chooses one action per module.

    ``observation`` is the per-module observation bundle, or None for policies
    that need nothing beyond the action mask. ``mask`` is the ``(N, 5)`` legal
    action mask from the rules, and a policy must only ever return actions the
    mask permits.
    """

    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: Any, mask: np.ndarray) -> np.ndarray: ...


class RandomLegalPolicy:
    """Picks uniformly among each module's legal actions.

    The floor every other policy has to beat. Note that it is not a neutral
    baseline for congestion: uniform random choices collide far more often than
    a coordinated policy, so its move-rejection rate is an upper bound rather
    than a typical value.
    """

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observation: Any, mask: np.ndarray) -> np.ndarray:
        return moves.sample_legal_actions(mask, self.rng)


class StayPolicy:
    """Never moves. Used to pin the no-op baseline in reward and metric tests."""

    name = "stay"

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, observation: Any, mask: np.ndarray) -> np.ndarray:
        return np.full(mask.shape[0], moves.STAY, dtype=np.int64)
