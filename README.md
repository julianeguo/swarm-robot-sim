# swarm-robot-sim

Swarm robot 2D simulation with discrete positions, and a decentralized
controller trained with multi-agent deep RL so a lattice of connected modules
can self-reconfigure from a starting shape into a target shape.

One policy is shared by every module (parameter sharing / IPPO), each module
acts on a local observation plus an action mask, and the team shares a single
reward.

## Status
to be updated

## Repository layout

```
swarm-robot-sim/
├── pyproject.toml
├── src/swarm/
│   ├── hexgrid.py
│   ├── shapes.py      # fixed shapes + bitmap conversion
│   ├── sim.py
│   ├── env.py         # obs = 6 neighbor bits + (x,y,z) position + target bitmap
│   ├── train.py       # includes the policy network for now
│   └── render.py      # pygame only
├── scripts/demo_random.py
└── tests/test_sim.py
```

Directories exist ahead of the milestone that fills them; the table above says
what is implemented today.

## Quickstart
to be updated

## Design notes
to be added
