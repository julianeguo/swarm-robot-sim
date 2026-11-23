"""
A thin PettingZoo ParallelEnv wrapper around sim.py.

sim.py is only the physics (cells, legal moves, conflicts). This file turns it into an RL
problem: named agents, episodes, observations, rewards, and when an episode ends. "Parallel"
means every robot acts at once each step, matching sim.apply_actions.

On reset it picks the start and target shapes:
  start  = shapes.make_shape(<a name from episode.start_shapes>, N)
  target = the start after K random legal moves in the sim (K = episode.target_walk_steps),
           one robot per move, so every episode is solvable in at most K moves and K sets the
           difficulty. A walk that ends back on the start shape is thrown away and redone.
Both are kept on the env (start_cells, target_cells) for logging and rendering.

It builds each robot's observation:
  6 neighbour bits (hexgrid.DIRS order): is a robot in that neighbouring cell?
  + 1 on-target bit: is this robot's own cell a target cell?
  + 6 neighbour-target bits (hexgrid.DIRS order): is that neighbouring cell a target cell?
  + 2 numbers: position (i, j), divided by R
  + the target bitmap around the origin, radius R, flattened in hexgrid.window_offsets order
  (R = largest start shape radius + K, so the crop always holds the whole target)
The 7 target bits repeat information that position + bitmap already contain, but in a form
the network can use directly. Without them it has to learn to look its own position up in
the bitmap, which it doesn't manage, and training collapses to "never move".
passes along the action masks from sim.legal_action_mask(), and gives each robot a reward
after every step. The team part is the same for every robot:
  change in sim.filled() * fill_scale / N
  - time_cost_total / max_steps
  - isolation_penalty * (robots with no neighbours) / N
  + completion_bonus on the step the target is completed
The robot's own part is different per robot:
  + self_fill_scale if this robot stepped onto a target cell this step,
    - self_fill_scale if it stepped off one, 0 otherwise
With only the team part, one robot's good move is drowned out by what everyone else did
that step, and with 8+ robots training collapses to "never move". The own part tells each
robot directly whether its own move helped.
An episode terminates when sim.completion() == 1.0 and truncates at episode.max_steps.
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from swarm import hexgrid, shapes
from swarm.sim import N_ACTIONS, SwarmSim

MAX_WALK_TRIES = 100  # redo a random walk that ends back on the start this many times at most


class SwarmEnv(ParallelEnv):
    metadata = {"name": "swarm_v0", "render_modes": []}

    def __init__(self, cfg):
        """Build the env from a loaded config (see configs/default.yaml).

        All sizes are fixed here, once, so the observation has the same length every episode.
        """
        episode = cfg["episode"]
        reward = cfg.get("reward") or {}
        self.n = int(episode["n_modules"])
        self.max_steps = int(episode["max_steps"])
        self.k = int(episode["target_walk_steps"])
        self.start_shapes = list(episode.get("start_shapes", shapes.SHAPE_NAMES))
        if self.n < 2:
            raise ValueError("n_modules must be >= 2: a lone robot has no pivot and can't move")
        if self.k < 1:
            raise ValueError("target_walk_steps must be >= 1, or the target is the start")
        unknown = set(self.start_shapes) - set(shapes.SHAPE_NAMES)
        if not self.start_shapes or unknown:
            raise ValueError(f"start_shapes must be a non-empty subset of {shapes.SHAPE_NAMES}")

        self.fill_scale = float(reward.get("fill_scale", 1.0))
        self.time_cost = float(reward.get("time_cost_total", 0.0)) / self.max_steps
        self.completion_bonus = float(reward.get("completion_bonus", 0.0))
        self.isolation_penalty = float(reward.get("isolation_penalty", 0.0))
        self.self_fill_scale = float(reward.get("self_fill_scale", 0.0))

        # Sizes. A walk move takes one robot one step, so the target lies within
        # (largest start radius + K) of the origin. That is R, the target crop radius.
        # During the episode each robot moves at most max_steps more, and the neighbour
        # lookup reads 1 cell further, which gives the board radius M.
        shape_r = max(shapes.shape_radius(shapes.make_shape(s, self.n)) for s in self.start_shapes)
        self.r = shape_r + self.k
        self.sim = SwarmSim(hexgrid.board_radius(self.r, self.max_steps, 1))
        self.crop = hexgrid.window_offsets(self.r)  # (C, 2) target-bitmap cells, fixed order
        crop_idx = hexgrid.to_index(self.crop, self.sim.m)
        self._crop_rows, self._crop_cols = crop_idx[:, 0], crop_idx[:, 1]

        self.possible_agents = [f"module_{n}" for n in range(self.n)]
        self.agents = []
        obs_size = 6 + 1 + 6 + 2 + len(self.crop)
        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, (obs_size,), np.float32),
                "action_mask": spaces.MultiBinary(N_ACTIONS),
            }
        )
        self._action_space = spaces.Discrete(N_ACTIONS)

        self.rng = np.random.default_rng(cfg.get("seed"))
        self.t = 0
        self._filled = 0
        self.start_name = None
        self.start_cells = None
        self.target_cells = None

    # PettingZoo requires the same space object on every call.
    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    # --- episode -------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Pick a start shape, make a target by random walk, and place the robots."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.start_name = self.start_shapes[self.rng.integers(len(self.start_shapes))]
        self.start_cells = shapes.make_shape(self.start_name, self.n)
        self.target_cells = self._random_walk_target(self.start_cells)
        self.sim.reset(self.start_cells, self.target_cells)

        self.t = 0
        self._filled = self.sim.filled()
        self.agents = self.possible_agents[:]
        # The target never changes during an episode, so its crop is read once here.
        self._target_bits = self.sim.target[self._crop_rows, self._crop_cols].astype(np.float32)

        info = {"completion": self.sim.completion(), "start_shape": self.start_name}
        return self._observe(), {a: dict(info) for a in self.agents}

    def step(self, actions):
        """Apply one action per robot, all at once, and return the PettingZoo 5-tuple."""
        on_target_before = self._is_target(self.sim.positions)
        self.sim.apply_actions([actions[a] for a in self.possible_agents])
        self.t += 1
        own = self.self_fill_scale * (self._is_target(self.sim.positions) - on_target_before)

        filled = self.sim.filled()
        isolated = int((~self.sim.neighbor_occupancy().any(axis=1)).sum())
        done = filled == self.n
        reward = (
            (filled - self._filled) * self.fill_scale / self.n
            - self.time_cost
            - self.isolation_penalty * isolated / self.n
            + (self.completion_bonus if done else 0.0)
        )
        self._filled = filled
        truncated = not done and self.t >= self.max_steps

        info = {"completion": filled / self.n, "isolated": isolated}
        if done or truncated:
            info["success"] = done
        agents = self.agents
        if done or truncated:
            self.agents = []  # PettingZoo: no live agents once the episode is over

        return (
            self._observe(),
            {a: reward + float(own[n]) for n, a in enumerate(self.possible_agents) if a in agents},
            {a: done for a in agents},
            {a: truncated for a in agents},
            {a: dict(info) for a in agents},
        )

    def close(self):
        pass

    # --- helpers -------------------------------------------------------------------------

    def _observe(self):
        """Build {agent: {"observation", "action_mask"}} for every robot from the sim."""
        neighbours = self.sim.neighbor_occupancy().astype(np.float32)  # (N, 6)
        on_target = self._is_target(self.sim.positions)[:, None]  # (N, 1)
        nb_target = self._is_target(hexgrid.neighbors(self.sim.positions))  # (N, 6)
        position = self.sim.positions.astype(np.float32) / self.r  # (N, 2)
        target = np.broadcast_to(self._target_bits, (self.n, len(self._target_bits)))
        obs = np.concatenate([neighbours, on_target, nb_target, position, target], axis=1)
        mask = self.sim.legal_action_mask().astype(np.int8)  # gymnasium masks are int8
        return {
            a: {"observation": obs[n], "action_mask": mask[n]}
            for n, a in enumerate(self.possible_agents)
        }

    def _is_target(self, cells):
        """1.0 where a cell (..., 2) is a target cell, else 0.0, as float32 (...,)."""
        idx = hexgrid.to_index(cells, self.sim.m)
        return self.sim.target[idx[..., 0], idx[..., 1]].astype(np.float32)

    def _random_walk_target(self, start):
        """Return the start cells after K random legal moves, one robot moving per move.

        Each move is legal on the board as it is (nothing else moves at the same time), so it
        can be undone, and the target is reachable from the start in at most K moves.
        If the walk ends with every start cell still filled (it undid itself), redo it.
        The sim is used as scratch space here; reset() places the robots again afterwards.
        """
        for _ in range(MAX_WALK_TRIES):
            self.sim.reset(start, start)
            for _ in range(self.k):
                mask = self.sim.legal_action_mask()
                movable = np.flatnonzero(mask[:, 1:].any(axis=1))
                if len(movable) == 0:
                    break
                robot = self.rng.choice(movable)
                actions = np.zeros(self.n, dtype=np.int64)
                actions[robot] = 1 + self.rng.choice(np.flatnonzero(mask[robot, 1:]))
                self.sim.apply_actions(actions)
            if self.sim.filled() < self.n:  # target == start means filled() == N
                return self.sim.positions.copy()
        raise RuntimeError(f"random walk kept returning to the start ({MAX_WALK_TRIES} tries)")
