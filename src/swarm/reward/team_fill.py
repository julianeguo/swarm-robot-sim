"""The shared team reward: progress on filling the target, minus time.

    reward = fill_scale * delta(filled target cells) / n_modules
             - time_cost_total / max_steps
             + completion_bonus   on the step the shape is completed

Two properties worth keeping in mind when adding variants:

The fill term is a potential function of the number of filled target cells, so
it telescopes over an episode. Total fill reward depends only on start and end
occupancy, never on the path, which means a module cannot farm reward by
stepping on and off a target cell. Any variant that breaks this invites exactly
that exploit.

The time cost is a share of the episode budget, not a flat constant. The fill
term can only ever sum to ``fill_scale`` across a whole episode, so a flat
per-step penalty would come to dominate the task signal as ``max_steps`` grows
and make finishing early more valuable than solving the shape.
"""

from __future__ import annotations

from typing import Any

from swarm.config import RewardConfig
from swarm.reward.base import register
from swarm.sim.state import SwarmState


@register
class TeamFillReward:
    name = "team_fill"

    def __init__(self, config: RewardConfig, max_steps: int):
        self.config = config
        self.max_steps = max_steps
        self.step_penalty = config.step_penalty(max_steps)

    def reset(self, state: SwarmState) -> None:
        """Nothing to carry between steps: the reward is a function of the transition."""
        return None

    def __call__(
        self, prev_state: SwarmState, new_state: SwarmState, info: dict[str, Any]
    ) -> float:
        n_modules = max(1, new_state.n_modules)
        newly_filled = info["newly_filled"]

        reward = self.config.fill_scale * newly_filled / n_modules
        reward -= self.step_penalty

        # Guarded rather than relying on the episode ending here, so a variant
        # that keeps stepping after completion cannot collect the bonus twice.
        if new_state.is_complete and not prev_state.is_complete:
            reward += self.config.completion_bonus

        return float(reward)

    def budget(self) -> dict[str, float]:
        """The reward each component can contribute over a full episode.

        Used in tests and when retuning: the point of the scaling is that the
        time cost stays well under the task reward.
        """
        return {
            "max_fill": self.config.fill_scale,
            "completion_bonus": self.config.completion_bonus,
            "max_time_cost": self.step_penalty * self.max_steps,
        }
