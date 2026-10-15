# Run S: snapshot tactic, warm-started from run A

SAC intra-option policy for the **snapshot tactic** (paper Sec 4.2.2, Eq. 40).

Everything except the shaping reward is identical to the angle tactic's run W
([`../angle_tactic_W`](../angle_tactic_W)):
- the same JSBSim F-16 environment, kill rule and terminal outcomes;
- the WGAN-GP curriculum and the opponent ladder ending at the full Shaw-doctrine `BFMAgent`;
- a warm start from run A's epoch-20 actor ([`../angle_tactic_A`](../angle_tactic_A)), which can fly but barely fights;
- the paper's 15M sim steps (epochs 21–267).

Train it with `train_angle_wgan.py --tactic snapshot`. The snapshot reward pays for pointing at the
bandit at any aspect, plus a separate term for sitting behind its 3/9 line: "the capture of the shooting window".

## Headline result

**Training fights against the full BFM (`bfm_angles`):**

| Epochs | Fights | Win | Loss | Draw (timeout) | Crash |
|---|---|---|---|---|---|
| All top-rung epochs (72–267) | 3,914 | **75%** | 14% | 11% | 0.2% |
| Last 20 epochs | 394 | **83%** | 15% | 2% | 0% |
| Paper's snapshot tactic | — | 0.4–0.5 | — | — | — |

The top-rung win rate rose from 62% in the first half of those epochs to 87% in the second. The best policy
(`policy_best.pt`) reached a rolling win rate of **0.97** over 100 fights at epoch 176.

**Frozen-weight flights** (deterministic actions, the same 38 starts as every other run; see
[`../../visualizations`](../../visualizations)):

| Policy | Win | Timeout | Loss | Crash |
|---|---|---|---|---|
| `policy_final.pt` (15M) | **37** | 1 | 0 | 0 |
| `policy_best.pt` (epoch 176) | 32 | 5 | 1 | 0 |

## Read this before quoting the numbers: the floor exploit

**About a third of the snapshot wins exploit the opponent's ground-avoidance rule**, not air-combat skill.

**What the flights show.** All 13 of the final policy's frozen wins that last over 30 s follow the same sequence:
1. From a defensive start the agent extends in a dive with the BFM 1.3 km behind it.
2. It descends below 1,100 m; its median low point is 895 m.
3. Low down, it reverses and kills.

The BFM's median low point in these fights is 1,520 m.

**Why it works.** `BFMAgent._safety` (`f16_bfm_agent.py`) overrides the expert's tactics with a
wings-level maximum pull (`GROUND_AVOID`) whenever its predicted pull-out would end within 600 ft of its
3,000 ft (914 m) hard deck. Our agent is only penalised below 500 m and crashes at 200 m. Dragging the
expert into a dive forces it into a predictable straight pull-up, and the agent turns onto it.

**What this means for the numbers.**
- Excluding these, the final policy wins about **24 of 38** frozen fights cleanly.
- The other two tactics never do this: energy and angle wins all happen above 1,100 m.
- **Fix:** give both aircraft the same floor, for example by lowering `BFMAgent`'s hard deck to the agent's 500 m, then retrain.

## One reward change from the paper

`reward.Constants.snapshot_aa_in_range = True` weights Eq. 40's standalone AA term by φ_range.

- **The failure:** trained literally, the bare φ_AA term paid **+0.5 to +0.9 per step for flying away
  from the bandit** while staying in its rear sector. The agent learned to turn away and dive, and crashes
  rose from 2 to 12 of 20 per epoch by epoch 28. The literal-reward run (`runs/S0_eq40_literal`) was
  stopped there, and its logs are in `training/comparison_eq40_literal/`.
- **The effect:** weighted by φ_range, the term is unchanged where a snapshot happens (φ_range = 1 at 500 m)
  and fades as the fight is abandoned (0.03 at 9 km). With it, crashes peaked at 7 per epoch right after
  the actor unfroze, then fell to 0–2.
- **Restoring the paper's form:** set the flag to `False`.

## How it learned

| Phase | Epochs | What happened |
|---|---|---|
| Critic warm-up and unfreeze | 21–30 | Run A's actor was frozen for the first 20,000 updates. After unfreezing, crashes peaked at 7 per epoch (epoch 25), then fell to 0–2 |
| First wins | 26–45 | Mostly timeouts at first, then wins against the 250 kt straight target |
| Climb the ladder | 46–72 | Alternated between `straight` and `turn4` for 18 epochs, cleared `turn6` in one epoch and `bfm_energy` in seven. **Reached the full BFM at epoch 72** (run W: epoch 93) |
| Full BFM only | 72–267 | Rolling win rate 0.4 → 0.97 (epoch 176), last 20 epochs 83%. Losses stayed at ~15%, the price of an aggressive tactic, as the paper says |

Fights got shorter as the agent improved, so 15M sim steps came to 247 epochs here, against 209 for run W.

## Files

| File | Contents |
|---|---|
| `policy_final.pt` | actor at the end of training: 15.0M sim steps, epoch 267 (`sac.GaussianPolicy(21, 4)` state dict) |
| `policy_best.pt` + `.json` | best actor: epoch 176, rolling win rate 0.97 |
| `state_15M_noreplay.pt` | resumable state: actor, critics, optimizers, α, curriculum, counters, RNG, tactic; the replay buffer is left out |
| `wgan_generator.pt`, `wgan_critic.pt` | curriculum WGAN at the end of training |
| `snapshots/actor_epochNNNN.pt` | actors every 25 epochs |
| `stats.json` | the numbers in the tables above |
| `training/` | per-episode and per-epoch CSVs, console log, `training_diagnostics.png`, and `comparison_eq40_literal/` |
| `flights/` | three late training wins against the full BFM, as Tacview files and 3D plots |

## Using it

```bash
python fly_policy.py --policy models/angle_tactic_S/policy_final.pt --out visualizations/my_S --random-starts 30
python train_angle_wgan.py --run-dir runs/S_continued --resume models/angle_tactic_S/state_15M_noreplay.pt --total-env-steps 20000000 --opponent-curriculum
```

The resume keeps the snapshot tactic automatically, because the checkpoint records it.

## Known limitations

1. **The floor exploit** described above inflates the win rate. Fix the floors and retrain before comparing tactics.
2. **Head-on wins in about 8 s** exploit the BFM's offset entry before the merge. All three tactics do this; see W's README.
3. **Single run, single seed.**
