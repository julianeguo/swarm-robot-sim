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


class GreedyPolicy:
    """Each module steps toward its nearest unfilled target cell.

    The handcrafted reference the learned policy has to beat. It reads raw
    deltas from the observation's ``aux`` entry rather than reconstructing them
    from the normalized vector, since it is a baseline and not a network.

    Among its legal actions, a module picks the one that minimizes Manhattan
    distance from the destination to its goal cell, breaking ties randomly. The
    tie-break only matters when the goal is off-axis in both directions and two
    moves are equally good; it is not a way out of a blocked position, because
    under Manhattan distance a sideways step always scores strictly worse than
    staying put. ``strict=True`` restricts the choice to the one or two aligned
    moves, which is behaviourally almost the same thing.

    ``epsilon`` is the deadlock escape, and it is not optional in practice.
    Pure greedy applies zero moves on 99% of steps: with only ~45% of modules
    mobile at any instant, the move a module wants is usually illegal, every
    alternative scores worse than staying, and the swarm freezes within a step
    or two. A random legal action with probability ``epsilon`` trades efficiency
    for the ability to leave such configurations at all.

    Its remaining weakness is the one a shared policy also has to solve: nothing
    coordinates which module claims which cell, so several modules chase the
    same gap. Measuring that gap is the point of having this baseline.
    """

    name = "greedy"

    def __init__(self, seed: int | None = None, epsilon: float = 0.0, strict: bool = False):
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.strict = strict

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observation: Any, mask: np.ndarray) -> np.ndarray:
        if observation is None:
            raise ValueError("GreedyPolicy needs an observation; got None")

        deltas = observation["aux"]["nearest_delta"][:, 0, :]  # nearest slot only
        valid = observation["aux"]["nearest_valid"][:, 0]
        on_target = observation["aux"]["on_target"]

        n_modules = mask.shape[0]
        actions = np.full(n_modules, moves.STAY, dtype=np.int64)
        explore = self.rng.random(n_modules) < self.epsilon

        for module in range(n_modules):
            if explore[module]:
                legal = np.flatnonzero(mask[module])
                actions[module] = int(self.rng.choice(legal))
                continue
            # A module already on a target cell holds its place; leaving would
            # undo filled progress the team is scored on.
            if on_target[module] or not valid[module]:
                continue
            actions[module] = self._choose(deltas[module], mask[module])

        return actions

    def _choose(self, delta: np.ndarray, module_mask: np.ndarray) -> int:
        """Best legal action for one module, given the delta to its goal cell."""
        if self.strict:
            for action in self._aligned_actions(delta):
                if module_mask[action]:
                    return action
            return moves.STAY

        # Distance from each candidate destination to the goal cell. Staying is
        # scored too, so a module only moves when moving is at least as good.
        best_actions: list[int] = []
        best_distance = None
        for action in range(moves.N_ACTIONS):
            if not module_mask[action]:
                continue
            step = moves.DELTAS[action]
            remaining = abs(int(delta[0]) - int(step[0])) + abs(int(delta[1]) - int(step[1]))
            if best_distance is None or remaining < best_distance:
                best_distance = remaining
                best_actions = [action]
            elif remaining == best_distance:
                best_actions.append(action)

        if not best_actions:
            return moves.STAY
        return int(self.rng.choice(best_actions))

    def _aligned_actions(self, delta: np.ndarray) -> list[int]:
        """The one or two moves that point straight at the goal, closest axis first."""
        row_delta, col_delta = int(delta[0]), int(delta[1])
        vertical = moves.DOWN if row_delta > 0 else moves.UP
        horizontal = moves.RIGHT if col_delta > 0 else moves.LEFT

        if abs(row_delta) > abs(col_delta):
            candidates = [vertical, horizontal]
        elif abs(col_delta) > abs(row_delta):
            candidates = [horizontal, vertical]
        else:
            candidates = [vertical, horizontal]
            if self.rng.random() < 0.5:
                candidates.reverse()

        if row_delta == 0:
            candidates = [action for action in candidates if action != vertical]
        if col_delta == 0:
            candidates = [action for action in candidates if action != horizontal]
        return candidates
