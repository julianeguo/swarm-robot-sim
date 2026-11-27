# Edwin's branch notes (as of 2026-11-22)

Working log for the `edwin` branch: what we changed, what the experiments showed, what's
decided, and what's next. Runs referenced here live in `runs/` (git-ignored, local only).

## How to run

```bash
pip install -e ".[rl,viz,dev]"
python3 -m pytest
python3 -m swarm.train configs/<config>.yaml          # writes policy.pt + checkpoints/ in cwd
python3 scripts/demo_random.py --config configs/<config>.yaml --render          # random moves
python3 scripts/demo_random.py --config configs/<config>.yaml --policy policy.pt --render
```

A policy only loads with a config that gives the same observation size (it depends on N, K and
the start shapes). Use the config the policy was trained with.

## What's on the branch

| Commit / state | Change |
|---|---|
| `45d07dc` | `episode.fixed_pair` (reuse one start/target pair), `configs/debug_fixed.yaml`; train.py checkpoints (`checkpoints/latest.pt`, `best.pt`) and early stopping (`train.stop_success`, `stop_patience`) |
| `907eea3` | `sim.keep_connected`: (1) a robot may move only if its occupied neighbours form one unbroken run around its 6-cell ring; (2) neighbour lock: two adjacent robots can't move in the same step (random winner). Together: the swarm can never split, using only local info. `episode.end_on_freeze` + `reward.freeze_penalty`: end the episode as a failure when no robot can move, charging the remaining time cost + penalty (so freezing is never cheaper than continuing). Both on in `default.yaml`. `configs/exp_small_fixed.yaml`. |
| uncommitted (11-27) | `train.critic: central` (separate actor/critic; critic also gets `env.state()`: steps left, fill fraction, movable fraction, outside-crop fraction + occupancy crop; `actor.pt` saved). `observation.type: local` + `window_radius` (robot-centred target window, 6 + 37 = 43 inputs for any N, K). Resume support (`checkpoints/resume.pt`, `--resume`). Configs `exp_small_*_central.yaml`, `exp_small_curriculum_local.yaml`. 166 tests pass. |
| `c8e0227` | Reverse curriculum (`curriculum.enabled`, `ReverseCurriculum` in env.py): episodes start j moves before the end of the random walk that made the target; levels [1, 2, 4, 8]; promote at 80% over 100 episodes; 20% replay of easier levels; `checkpoints/level_<j>.pt` on promotion. `configs/exp_small_curriculum.yaml`, `configs/exp_small_varied.yaml`. 3 tests. 154 tests pass. |

## Experiment log

| Run | Setup | Result | Takeaway |
|---|---|---|---|
| 1 | N=4 fixed pair, old rules | 100% in ~25k steps | the training loop works |
| 2 | N=9 fixed pair (multi-step), old rules | 0% after 1M | swarm splits on step 1, freezes ~step 12; ~94% of data is frozen |
| 3 | N=9 fixed pair + connectivity rules + freeze ending | **93.5%** after 1M, median 5 steps (`runs/2026-11-22_exp_small_fixed_connected`) | fixing dynamics made multi-step learnable |
| 4 | N=9 varied pairs + reverse curriculum, 2M | stuck at level 1, 10–35% (`runs/2026-11-22_exp_small_curriculum`) | can't generalise |
| 5 | N=9 varied pairs, no curriculum, 2M | 0–15%, ~random (`runs/2026-11-22_exp_small_varied`) | same |
| 6 | 9-robot fixed pair, **central critic** (11-27) | 0% after 1M; collapsed to "always stay" | 1 seed; may be bad luck vs run 3. Not retested yet |
| 7 | varied pairs + curriculum, central critic (11-27) | stuck at level 1, 30–40% | a better critic can't fix what the actor can't see |
| 8 | varied pairs + curriculum, old ("global") input, seeds 1–2 (11-27) | both stuck at level 1 (~30%) | with run 4: 3/3 seeds fail |
| 9 | varied pairs + curriculum, **robot-centred window (r=3)**, seeds 0–2 (11-27) | **3/3 reach the full task**: levels 1→8 by 59–74k steps; ≥90% at final level by 180–310k (`runs/2026-11-27_curriculum_local_seed*`) | **the input format was the bottleneck** |

Evaluation of run 9's `best.pt` (300 fresh episodes per row, seeds never used in training;
`scratchpad/evaluate.py` logic, not yet in the repo):

| Test | Random | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| N=9, K=8 (trained) | 2.7% | 87.0% | 89.3% | 88.7% |
| N=9, K=16 (unseen K) | 4.0% | 89.0% | 86.7% | 89.7% |
| N=4, K=3 (unseen N) | 22.7% | 99.7% | 99.7% | 99.7% |
| N=16, K=8 (unseen N) | 0.3% | 49.3% | 66.7% | 68.0% |
| N=25, K=8 (unseen N) | 0.0% | 6.3% | 21.3% | 29.7% |

Trained policies never froze (random play freezes 24% at N=16). Avg completion stays 0.82–0.99
even at N=25. Median solve 7–19 steps.

Key measurements:
- Old rules, random play: swarm split in 98–100% of episodes (step 1); full freeze in 96–100%,
  with *every* robot isolated. 88% of isolations came from simultaneous pivoting (a robot's
  pivot moved away the same step), 12% from being left behind. One-at-a-time moves can't
  fully freeze the swarm; simultaneous moves can.
- New rules: 0 splits in 1,200 random episodes. Random success N=4: 2.5% → 14.7%;
  N=9: 0% → 3.3%. N=16 still freezes in 21% of random episodes (connected but locked, e.g. a
  6-robot ring; the rule is conservative).
- Random success vs K at N=9 is flat (~3% for K=2..32), so K is a poor difficulty dial; reverse
  curriculum (moves remaining) is better: 16% / 16% / 13% / 3% at levels 1 / 2 / 4 / 8.
- **Diagnosis of runs 4–5:** at level 1, the learned policy almost never moves and behaves the
  same on and off target (P(move) 0.01 vs 0.05). A 10-line hand-written local rule ("if I'm off
  target and a neighbouring empty cell is a target, move there") solves level 1 100%. The info
  is there, but the observation format hides it: "is my cell a target?" needs combining 2
  position numbers with the right one of 331 absolute target-map slots.

## Information model (agreed: "relative to the swarm")

Runtime, each robot has only:
- its 6 neighbour bits (and the legal-move mask derived from them)
- A. its coordinates in the swarm's own frame (via neighbour-to-neighbour propagation from an
  anchor, or known start cell + counting its own moves); no world/GPS position
- B. the target shape, broadcast once at the start, in that swarm frame
- C. a shared sense of direction (direction 0 is the same for everyone)
- D. a short "I'm moving" handshake with direct neighbours (for the neighbour lock)

Not allowed at runtime: other robots' positions, which target cells are filled (beyond
neighbours), world position. Training-only global info (reward, episode endings, curriculum,
and possibly a critic with global features) is fine under centralized training / decentralized
execution, pending professor confirmation.

## Current architecture

- Shared-parameter IPPO. One MLP shared by all robots: obs → 128 → Tanh → 128 → Tanh, then
  actor head (7 logits, illegal masked to −1e9) and critic head (1 value). Actor and critic share
  the body and get the same input; the critic sees no extra info yet.
- Observation (N=9): 6 neighbour bits + position (i, j)/R + absolute target crop of radius
  R = shape radius + K (331 cells) = 339. Size depends on N and K, so policies don't transfer
  across swarm sizes.
- Reward (team, per step): Δfilled·fill_scale/N − time_cost_total/max_steps + completion_bonus
  on completion − (remaining time cost + freeze_penalty) on a freeze. Isolation penalty is moot
  under keep_connected.
- PPO: 256-step rollouts, GAE (γ 0.99, λ 0.95), 4 epochs full batch, clip 0.2, ent 0.01, lr 3e-4.

## Next steps (in order)

1. **Probe test** (supervised, ~1 min): can a small network learn "is my cell a target?" from
   the current input vs from a robot-centred window? Direct evidence for/against the format
   being the bottleneck.
2. **Robot-centred target window** as `observation: local` config switch: target bits in a
   radius-3 patch around the robot (37 cells; centre bit = "my cell is a target"), optionally a
   pointer (direction + distance) to the nearest target cell. No new information (derived from
   A + B), fixed size for any N and K.
3. Rerun curriculum vs baseline with it. Expect level 1 → ~100% if the format was the problem.
4. If it stalls later: critic improvements. Recommended layout: separate actor and critic
   networks; critic gets local obs + global features (steps left, fraction filled) via PettingZoo
   `state()`. Training only. Needs professor OK.
5. Then mixed swarm sizes / curriculum on N, and test at an unseen N to claim scaling.
6. Housekeeping: checkpoint bundle (weights + config + optimizer state + step) with `--resume`.

## Questions for the professor

1. Are assumptions A–D acceptable (especially swarm-frame coordinates and the neighbour
   handshake)?
2. Is training-only global info OK (reward, episode endings, a critic with global features), as
   long as the deployed actor uses only runtime-available inputs?
3. Is a robot-centred target window computed from A + B acceptable as the actor's input?
