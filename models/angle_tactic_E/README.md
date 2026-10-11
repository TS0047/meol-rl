# Run E: energy tactic, warm-started from run A

SAC intra-option policy for the **energy tactic** (paper Sec 4.2.3, Eq. 41). It is trained exactly like the
angle tactic's run W ([`../angle_tactic_W`](../angle_tactic_W)) and the snapshot tactic's run S
([`../angle_tactic_S`](../angle_tactic_S)):
- the same environment, kill rule, terminal outcomes, WGAN-GP curriculum and opponent ladder;
- a warm start from run A's actor;
- 15M sim steps (epochs 21–418).

Train it with `train_angle_wgan.py --tactic energy`. The shaping is the paper's Eq. 41 **unchanged**:
nose-on-at-range reward multiplied by the energy ratio E_self / E_adv, plus 0.25·φ_AA.

## Headline result

**Training fights against the full BFM (`bfm_angles`):**

| Epochs | Fights | Win | Loss | Draw (timeout) | Crash |
|---|---|---|---|---|---|
| All top-rung epochs (73–418) | 6,908 | **84%** | 6% | 10% | 0.3% |
| Last 20 epochs | 388 | **89%** | 1% | 10% | 0% |
| Paper's energy tactic | — | 0.5–0.6 | — | — | — |

The top-rung win rate rose from 79% in the first half of those epochs to 89% in the second. The best policy
(`policy_best.pt`) reached a rolling win rate of **0.98** over 100 fights at epoch 276.

**Frozen-weight flights** (deterministic actions, the same 38 starts as every other run; see
[`../../visualizations`](../../visualizations)):

| Policy | Win | Timeout | Loss | Crash |
|---|---|---|---|---|
| `policy_final.pt` (15M) | **30** | 4 | 4 | 0 |
| `policy_best.pt` (epoch 276) | 30 | 0 | 8 | 0 |

**What the flights show:**
- **Every frozen win comes in under 30 s, and all above 1,100 m.** Energy does not use the floor exploit
  that inflates the snapshot tactic's numbers; see S's README.
- **Its losses come from defensive starts.** Where run W extended away into a draw, energy turns and
  fights, and sometimes loses.
- The training-fight win rate (89%) is higher than the frozen one (30/38 = 79%). Training fights use starts
  the curriculum picks plus 30% uniform ones, while the frozen set holds a fixed share of hard defensive starts.

## How it learned

| Phase | Epochs | What happened |
|---|---|---|
| Critic warm-up and unfreeze | 21–25 | Run A's actor was frozen for 20,000 updates. On unfreezing, crashes spiked to 15 of 20 (epoch 23) and recovered to 0–2 within three epochs |
| Stay in the fight | 26–50 | Disengages dominated (as in run W), falling to ~2 per epoch by epoch 50 |
| First wins | 37–53 | Wins against the straight target, rolling win rate 0.05 → 0.24 |
| Climb the ladder | 54–73 | Bounced between `straight` and `turn4` for 16 epochs: it could beat the full BFM before it could beat a steady 4 g turner. Then cleared `turn6` in two epochs and `bfm_energy` in one. **Reached the full BFM at epoch 73** (run W: 93) |
| Full BFM only | 73–418 | Rolling 0.50 → 0.84 by epoch 116 → 0.98 best (epoch 276). Losses fell to ~1% in the last 20 epochs |

Fights got short, so 15M sim steps came to 398 epochs here (run W: 209, run S: 247).

## Files

| File | Contents |
|---|---|
| `policy_final.pt` | actor at the end of training: 15.0M sim steps, epoch 418 (`sac.GaussianPolicy(21, 4)` state dict) |
| `policy_best.pt` + `.json` | best actor: epoch 276, rolling win rate 0.98 |
| `state_15M_noreplay.pt` | resumable state: actor, critics, optimizers, α, curriculum, counters, RNG, tactic; the replay buffer is left out |
| `wgan_generator.pt`, `wgan_critic.pt` | curriculum WGAN at the end of training |
| `snapshots/actor_epochNNNN.pt` | actors every 25 epochs |
| `stats.json` | the numbers in the tables above |
| `training/` | per-episode and per-epoch CSVs, console log, `training_diagnostics.png` |
| `flights/` | three late training wins against the full BFM, as Tacview files and 3D plots |

## Using it

```bash
python fly_policy.py --policy models/angle_tactic_E/policy_final.pt --out visualizations/my_E --random-starts 30
python train_angle_wgan.py --run-dir runs/E_continued --resume models/angle_tactic_E/state_15M_noreplay.pt --total-env-steps 20000000 --opponent-curriculum
```

## Known limitations

1. **Far above the paper's 0.5–0.6.** The opponent is our own reconstruction of the paper's rule-based
   expert, the kill rule differs (a 0.5 s hold in the WEZ), and the observation is extended (21 dims).
   Compare tactics within this repository, not against the paper's absolute numbers.
2. **Head-on wins in about 8 s** exploit the BFM's offset entry before the merge, as in all three tactics.
3. **Single run, single seed.**
