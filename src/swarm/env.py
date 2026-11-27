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
With episode.fixed_pair: true, the pair made on the first reset is reused for every later
episode (a sanity check: can the policy learn even one fixed problem?).

It builds each robot's observation, in one of two layouts (observation.type in the config):
  "global" (default, the original):
    6 neighbour bits (hexgrid.DIRS order)
    + 2 numbers: position (i, j), divided by R
    + the target bitmap around the origin, radius R, flattened in hexgrid.window_offsets order
    (R = largest start shape radius + K, so the crop always holds the whole target)
    Size depends on N and K.
  "local" (robot-centred):
    6 neighbour bits
    + the target bits in a window of radius observation.window_radius (default 3) centred
      on the robot, in hexgrid.window_offsets order (the centre entry = "my cell is a target")
    Size 6 + 1 + 3w(w+1) (43 for w = 3), the same for any N and K. Information model: each
    robot stores the target (given at the start) and knows its own swarm-frame position, and
    looks the window up itself. The position is needed for that lookup but isn't an input.
passes along the action masks from sim.legal_action_mask(), and gives a team reward after
every step:
  change in sim.filled() * fill_scale / N
  - time_cost_total / max_steps
  - isolation_penalty * (robots with no neighbours) / N
  + completion_bonus on the step the target is completed
  - on a freeze (see below): the time cost of the remaining steps + freeze_penalty
An episode terminates when sim.completion() == 1.0 and truncates at episode.max_steps.

Reverse curriculum (curriculum block, enabled: true):
  The random walk start -> target is recorded. Instead of starting each episode at the
  beginning of the walk, start it j moves before the end, so at most j correct moves remain.
  j (the level) starts at levels[0] and moves up to the next level once the success rate over
  the last window_episodes episodes at the current level reaches promote_success_rate. A
  fraction replay_easier of episodes uses an earlier level, so earlier skills aren't forgotten.
  The target, and so the observation size, is the same at every level.

Global state for a centralized critic (training only; robots never see it):
  env.state() returns one vector shared by all robots:
    [steps left / max_steps, fraction of target filled, fraction of robots that can move,
     fraction of robots outside the crop] + occupancy of the target-crop cells (where every
     robot is, in the same cell order as the target bitmap)
  train.py feeds it to the critic only when train.critic is "central".

Options (episode block):
  allow_squeeze:  allow a move with both flanking cells occupied (no sweep check; the original
                  rule, before 2026-11-27). Off by default (see sim.py).
  keep_connected: sim's connectivity-preserving movement rules (see sim.py). Targets are
                  made under the same rules, so they stay solvable.
  end_on_freeze:  also terminate (with success False) when no robot has a legal move. The
                  board can never change again, so the rest of the episode would teach nothing.
                  The penalty charges the time cost the episode would still have paid, so
                  freezing on purpose is never cheaper than carrying on.
"""

from collections import deque

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from swarm import hexgrid, shapes
from swarm.sim import N_ACTIONS, SwarmSim

MAX_WALK_TRIES = 100  # redo a random walk that ends back on the start this many times at most


class ReverseCurriculum:
    """Which level (moves left before the target) each episode starts at, and promotion."""

    def __init__(self, levels, promote_success_rate=0.8, window_episodes=100, replay_easier=0.2):
        self.levels = sorted(int(j) for j in levels)
        if not self.levels or self.levels[0] < 1:
            raise ValueError("curriculum levels must be a non-empty list of ints >= 1")
        self.promote_success_rate = float(promote_success_rate)
        self.replay_easier = float(replay_easier)
        self.index = 0
        self.recent = deque(maxlen=int(window_episodes))

    @property
    def level(self):
        return self.levels[self.index]

    @property
    def at_final(self):
        return self.index == len(self.levels) - 1

    def sample(self, rng):
        """Level for the next episode: usually the current one, sometimes an earlier one."""
        if self.index > 0 and rng.random() < self.replay_easier:
            return self.levels[rng.integers(self.index)]
        return self.level

    @property
    def success_rate(self):
        """Success over the recent episodes at the current level (nan if none yet)."""
        return float(np.mean(self.recent)) if self.recent else float("nan")

    def record(self, level, success):
        """Count an episode's result toward promotion (only episodes at the current level)."""
        if level != self.level:
            return
        self.recent.append(bool(success))
        full = len(self.recent) == self.recent.maxlen
        if not self.at_final and full and self.success_rate >= self.promote_success_rate:
            self.index += 1
            self.recent.clear()


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
        self.fixed_pair = bool(episode.get("fixed_pair", False))
        self.keep_connected = bool(episode.get("keep_connected", False))
        self.allow_squeeze = bool(episode.get("allow_squeeze", False))
        self.end_on_freeze = bool(episode.get("end_on_freeze", False))
        self._fixed = None  # (start_name, start_cells, target_cells) once made, if fixed_pair
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
        self.freeze_penalty = float(reward.get("freeze_penalty", 0.0))

        # Sizes. A walk move takes one robot one step, so the target lies within
        # (largest start radius + K) of the origin. That is R, the target crop radius.
        # During the episode each robot moves at most max_steps more, and the neighbour
        # lookup reads 1 cell further, which gives the board radius M.
        shape_r = max(shapes.shape_radius(shapes.make_shape(s, self.n)) for s in self.start_shapes)
        self.r = shape_r + self.k
        obs_cfg = cfg.get("observation") or {}
        self.obs_type = obs_cfg.get("type", "global")
        if self.obs_type not in ("global", "local"):
            raise ValueError(f"observation.type must be 'global' or 'local', got {self.obs_type!r}")
        self.window_radius = int(obs_cfg.get("window_radius", 3))
        if self.obs_type == "local" and self.window_radius < 1:
            raise ValueError("observation.window_radius must be >= 1")
        # The board must leave room for the farthest cell any robot reads.
        reach = self.window_radius if self.obs_type == "local" else 1
        self.sim = SwarmSim(
            hexgrid.board_radius(self.r, self.max_steps, reach),
            keep_connected=self.keep_connected,
            allow_squeeze=self.allow_squeeze,
        )
        self.window = hexgrid.window_offsets(self.window_radius)  # (W, 2), local obs only
        self.crop = hexgrid.window_offsets(self.r)  # (C, 2) target-bitmap cells, fixed order
        crop_idx = hexgrid.to_index(self.crop, self.sim.m)
        self._crop_rows, self._crop_cols = crop_idx[:, 0], crop_idx[:, 1]

        self.possible_agents = [f"module_{n}" for n in range(self.n)]
        self.agents = []
        if self.obs_type == "local":
            obs_size = 6 + len(self.window)
        else:
            obs_size = 6 + 2 + len(self.crop)
        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, (obs_size,), np.float32),
                "action_mask": spaces.MultiBinary(N_ACTIONS),
            }
        )
        self._action_space = spaces.Discrete(N_ACTIONS)
        self.state_space = spaces.Box(0.0, 1.0, (4 + len(self.crop),), np.float32)

        self.rng = np.random.default_rng(cfg.get("seed"))
        cur = cfg.get("curriculum") or {}
        self.curriculum = None
        if cur.get("enabled"):
            if cur.get("type", "reverse") != "reverse":
                raise ValueError("only curriculum type 'reverse' is implemented")
            levels = [j for j in cur.get("levels", [1, 2, 4, 8]) if j <= self.k]
            self.curriculum = ReverseCurriculum(
                levels or [self.k],
                cur.get("promote_success_rate", 0.8),
                cur.get("window_episodes", 100),
                cur.get("replay_easier", 0.2),
            )
        self.level = None  # this episode's curriculum level (None without a curriculum)
        self._walk = None  # positions after each move of the last random walk, start first
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
        """Pick a start shape, make a target by random walk, and place the robots.

        With fixed_pair, only the first reset picks a pair (using its seed); later resets
        reuse it, whatever seed they get.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._fixed is not None:
            self.start_name, self.start_cells, self.target_cells, walk = self._fixed
        else:
            self.start_name = self.start_shapes[self.rng.integers(len(self.start_shapes))]
            self.start_cells = shapes.make_shape(self.start_name, self.n)
            self.target_cells = self._random_walk_target(self.start_cells)
            walk = self._walk
            if self.fixed_pair:
                self._fixed = (self.start_name, self.start_cells, self.target_cells, walk)
        if self.curriculum is not None:
            # Start j moves before the end of the walk (skipping back past any point that
            # already equals the target, in case the walk undid itself near the end).
            self.level = self.curriculum.sample(self.rng)
            target_key = _cells_key(self.target_cells)
            i = max(len(walk) - 1 - self.level, 0)
            while i > 0 and _cells_key(walk[i]) == target_key:
                i -= 1
            self.start_cells = walk[i]
        self.sim.reset(self.start_cells, self.target_cells)

        self.t = 0
        self._filled = self.sim.filled()
        self.agents = self.possible_agents[:]
        # The target never changes during an episode, so its crop is read once here.
        self._target_bits = self.sim.target[self._crop_rows, self._crop_cols].astype(np.float32)

        info = {
            "completion": self.sim.completion(),
            "start_shape": self.start_name,
            "level": self.level,
        }
        return self._observe(), {a: dict(info) for a in self.agents}

    def step(self, actions):
        """Apply one action per robot, all at once, and return the PettingZoo 5-tuple."""
        self.sim.apply_actions([actions[a] for a in self.possible_agents])
        self.t += 1

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

        obs = self._observe()
        frozen = (
            self.end_on_freeze
            and not done
            and not any(o["action_mask"][1:].any() for o in obs.values())
        )
        if frozen:
            reward -= self.time_cost * (self.max_steps - self.t) + self.freeze_penalty
        terminated = done or frozen
        truncated = not terminated and self.t >= self.max_steps

        info = {
            "completion": filled / self.n,
            "isolated": isolated,
            "frozen": frozen,
            "level": self.level,
        }
        if terminated or truncated:
            info["success"] = done
            if self.curriculum is not None:
                self.curriculum.record(self.level, done)
        agents = self.agents
        if terminated or truncated:
            self.agents = []  # PettingZoo: no live agents once the episode is over

        return (
            obs,
            {a: reward for a in agents},
            {a: terminated for a in agents},
            {a: truncated for a in agents},
            {a: dict(info) for a in agents},
        )

    def close(self):
        pass

    def state(self):
        """The global state vector for a centralized critic (see the top of this file).

        Training-only information: the actor must never be given this.
        """
        occ = self.sim.occupied[self._crop_rows, self._crop_cols].astype(np.float32)
        n = max(self.sim.n, 1)
        can_move = self.sim.legal_action_mask()[:, 1:].any(axis=1).sum() if self.sim.n else 0
        scalars = [
            (self.max_steps - self.t) / self.max_steps,
            self.sim.filled() / n,
            can_move / n,
            1.0 - occ.sum() / n,  # robots outside the crop
        ]
        return np.concatenate([np.asarray(scalars, dtype=np.float32), occ])

    # --- helpers -------------------------------------------------------------------------

    def _observe(self):
        """Build {agent: {"observation", "action_mask"}} for every robot from the sim."""
        neighbours = self.sim.neighbor_occupancy().astype(np.float32)  # (N, 6)
        if self.obs_type == "local":
            cells = self.sim.positions[:, None, :] + self.window[None, :, :]  # (N, W, 2)
            idx = hexgrid.to_index(cells, self.sim.m)
            window = self.sim.target[idx[..., 0], idx[..., 1]].astype(np.float32)  # (N, W)
            obs = np.concatenate([neighbours, window], axis=1)
        else:
            position = self.sim.positions.astype(np.float32) / self.r  # (N, 2)
            target = np.broadcast_to(self._target_bits, (self.n, len(self._target_bits)))
            obs = np.concatenate([neighbours, position, target], axis=1)
        mask = self.sim.legal_action_mask().astype(np.int8)  # gymnasium masks are int8
        return {
            a: {"observation": obs[n], "action_mask": mask[n]}
            for n, a in enumerate(self.possible_agents)
        }

    def _random_walk_target(self, start):
        """Return the start cells after K random legal moves, one robot moving per move.

        Each move is legal on the board as it is (nothing else moves at the same time), so it
        can be undone, and the target is reachable from the start in at most K moves.
        If the walk ends with every start cell still filled (it undid itself), redo it.
        The sim is used as scratch space here; reset() places the robots again afterwards.
        Every point along the walk (start first) is kept in self._walk for the curriculum.
        """
        for _ in range(MAX_WALK_TRIES):
            self.sim.reset(start, start)
            self._walk = [self.sim.positions.copy()]
            for _ in range(self.k):
                mask = self.sim.legal_action_mask()
                movable = np.flatnonzero(mask[:, 1:].any(axis=1))
                if len(movable) == 0:
                    break
                robot = self.rng.choice(movable)
                actions = np.zeros(self.n, dtype=np.int64)
                actions[robot] = 1 + self.rng.choice(np.flatnonzero(mask[robot, 1:]))
                self.sim.apply_actions(actions)
                self._walk.append(self.sim.positions.copy())
            if self.sim.filled() < self.n:  # target == start means filled() == N
                return self.sim.positions.copy()
        raise RuntimeError(f"random walk kept returning to the start ({MAX_WALK_TRIES} tries)")


def _cells_key(cells):
    """The set of occupied cells, ignoring which robot is where (for comparing shapes)."""
    return frozenset(map(tuple, np.asarray(cells).tolist()))
