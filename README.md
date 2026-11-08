# swarm-robot-sim

Swarm robot 2D simulation with discrete positions, and a decentralized
controller trained with multi-agent deep RL so a lattice of connected modules
can self-reconfigure from a starting shape into a target shape.

One policy is shared by every module (parameter sharing / IPPO), each module
acts on a local observation plus an action mask, and the team shares a single
reward.

## Status

| Milestone | Contents | State |
|---|---|---|
| M0 | Package skeleton, config loading and validation, ASCII renderer | done |
| M1 | State record, connectivity oracle, movement rules, invariant fuzzer | done |
| M2 | Shape library, random-walk target generation, curriculum, `Simulator`, random baseline | next |
| M3 | Observations, reward, greedy baseline | |
| M4 | pygame renderer and episode recording | |
| M5 | PettingZoo `ParallelEnv` wrapper, `parallel_api_test` | |
| M6 | Masked IPPO training loop; overfit one tiny episode first | |
| M7 | Curriculum scheduling, checkpointing, held-out evaluation | |
| M8 | Articulation-point fast path, vectorized observations, scale-up | |

## Repository layout

```
swarm-robot-sim/
├── pyproject.toml              # deps + pytest/ruff config; src layout
├── Makefile                    # make test / lint / fmt / demo
├── configs/
│   ├── default.yaml            # baseline: grid, episode, observation, reward, curriculum
│   └── debug_tiny.yaml         # 8x8, N=4 — used by tests and overfit runs
├── src/swarm/
│   ├── config.py               # YAML -> validated dataclasses, with `inherits:` merging
│   ├── sim/                    # no RL, no PettingZoo, no torch (enforced by a test)
│   │   ├── state.py            # SwarmState: occupancy grid, module positions, target, step
│   │   ├── connectivity.py     # 4-connectivity flood fill; brute-force reference oracle
│   │   ├── moves.py            # the movement rules: legal_action_mask(), apply_actions()
│   │   ├── shapes.py           # connected shape samplers and the fixed shape library
│   │   ├── episodes.py         # EpisodeSpec, start/target generation, curriculum schedule
│   │   └── simulator.py        # Simulator.reset() / step(actions) -> (state, info)
│   ├── obs/
│   │   └── local_window.py     # state -> per-module observation arrays
│   ├── reward/
│   │   ├── base.py             # Reward protocol and registry
│   │   └── team_fill.py        # shared team reward; variants live beside it
│   ├── env/
│   │   ├── parallel_env.py     # PettingZoo ParallelEnv — translation layer only
│   │   └── spaces.py           # observation/action spaces built from config
│   ├── rl/
│   │   ├── networks.py         # shared actor-critic
│   │   ├── ppo.py              # masked IPPO training loop
│   │   ├── policies.py         # Policy protocol + RandomLegal, Greedy, Checkpoint
│   │   └── metrics.py          # episode stats -> TensorBoard/CSV
│   └── render/
│       ├── ascii.py            # text rendering; always available, used by tests
│       └── pygame_view.py      # optional viewer, imported lazily, off during training
├── scripts/
│   ├── random_walk_demo.py     # M1 deliverable: random-legal walk with invariant checks
│   ├── rollout.py              # run any policy headless, print stats
│   ├── train.py
│   ├── evaluate.py             # trained policy + baselines on held-out shapes
│   ├── play.py                 # pygame viewer for a policy or checkpoint
│   └── profile_sim.py          # steps/sec regression guard
└── tests/
```

Directories exist ahead of the milestone that fills them; the table above says
what is implemented today.

## Quickstart

```bash
pip install -e ".[dev]"      # numpy + pyyaml + pytest
make test
make demo                    # random-legal walk, invariants asserted every step
```

```bash
python3 scripts/random_walk_demo.py \
    --config configs/debug_tiny.yaml --steps 40 --render
```

The `rl` and `viz` extras (torch, gymnasium, pettingzoo, pygame) are installed
at the milestone that first needs them, not before.

## Design notes

**The rules live in one place.** `sim/moves.py` is the only module that decides
what is legal. Observations, the environment wrapper, and the baselines all
call `legal_action_mask()` rather than reimplementing legality, and both the
mask and `apply_actions()` route through a single predicate, so they cannot
drift apart. Swapping in sliding or pivoting movement means replacing one
function body.

**Current rule set.** A module stays put or translates one cell up, down, left,
or right. A move is legal when the destination is in bounds, the destination is
empty, and the swarm is still a single 4-connected group afterwards. One
consequence worth knowing: because a module may only move into a cell that
still touches the rest of the swarm, shapes without a reachable corner are
completely rigid — no module in a straight line or a solid 2x2 block can move
at all. Across random shapes only about 45% of modules have any legal move at a
given instant, against about 85% if destinations are allowed anywhere in the
8-neighbourhood. Tests pin the rigid cases as expected behaviour of this rule
set.

**Simultaneous actions, sequential resolution.** Modules choose at the same
time; moves are applied one at a time in a random order with legality
re-checked against the partially-updated state, so every intermediate
configuration is physically valid. A move that was masked-legal can therefore
still be rejected. Rejected moves become no-ops and the rejection rate is
reported in the step info as a diagnostic.

**Solvable episodes by construction.** The target shape is the start shape after
K random legal moves. Two independently sampled connected shapes of equal size
are not guaranteed to be connectable under a persistent connectivity
constraint, so sampling them independently would put an unknown ceiling on
success rate that is indistinguishable from a learning failure. K is also the
curriculum dial and gives a free upper bound on the optimal step count.
Independently sampled shape pairs are kept as a harder held-out evaluation set.

**Reward scale.** The shared team reward is
`delta(filled target cells) * fill_scale / n_modules`, minus a per-step time
cost, plus a completion bonus. The time cost is specified as a total budget
(`time_cost_total`) divided by `max_steps` rather than as a flat constant: the
fill term can only ever sum to `fill_scale` over a whole episode, so a flat
penalty would swamp the task signal as `max_steps` grows. The fill term is a
potential function, so it telescopes and cannot be farmed by moving a module
on and off a target cell.

**State layout.** Occupancy is a `(H, W)` bool grid and positions are an
`(N, 2)` int array — the same information in two layouts, since one suits
neighbour lookups and the other suits per-module work. `SwarmState.validate()`
keeps them in agreement. Nothing is stored per module as a Python object, so
the simulator can be batched later.

## Tests

```bash
make test        # 90 tests, under a second
```

The ones that matter most:

- **mask matches apply exactly** — for every module and action in randomly
  generated states, the mask says legal if and only if applying that move alone
  actually succeeds. Everything downstream trusts the mask.
- **long-rollout invariant fuzzer** — module count conserved, positions in
  bounds and unique, occupancy grid in sync with positions, and the swarm
  connected after every *intermediate* single-module move, not just at the end
  of a step.
- **rigid shapes** — lines and solid blocks have no legal moves, pinned so a
  future rule change has to be deliberate.
- **import hygiene** — `swarm.sim` imports without torch, pettingzoo,
  gymnasium, or pygame present.
- **reproducibility** — identical seeds give identical shapes and identical
  rollouts.
