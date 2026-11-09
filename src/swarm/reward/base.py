"""Reward interface and registry.

A reward is a pure function of two consecutive states plus the simulator's step
info. Keeping it that narrow is what lets variants be swapped by name from a
config without touching the simulator or the environment.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from swarm.config import RewardConfig
from swarm.sim.state import SwarmState


@runtime_checkable
class Reward(Protocol):
    """Scores one transition.

    The return value is a single scalar shared by every module: the swarm is
    rewarded as a team, so credit assignment is left entirely to the learner.
    """

    name: str

    def reset(self, state: SwarmState) -> None: ...

    def __call__(
        self, prev_state: SwarmState, new_state: SwarmState, info: dict[str, Any]
    ) -> float: ...


REWARDS: dict[str, type] = {}


def register(reward_cls: type) -> type:
    """Register a reward class under its ``name`` attribute."""
    REWARDS[reward_cls.name] = reward_cls
    return reward_cls


def build_reward(config: RewardConfig, max_steps: int) -> Reward:
    """Construct the reward named in the config.

    ``max_steps`` is required because the time cost is specified as a share of
    the episode budget rather than as a flat per-step constant.
    """
    if config.name not in REWARDS:
        raise KeyError(f"unknown reward '{config.name}'; registered: {sorted(REWARDS)}")
    return REWARDS[config.name](config, max_steps)
