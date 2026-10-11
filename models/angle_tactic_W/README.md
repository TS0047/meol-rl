# Run W — angle tactic, warm-started from run A, all fixes

SAC intra-option policy for the **angle tactic** (Algorithm 1), trained under the WGAN-GP
automatic curriculum (Algorithm 2) and an opponent ladder that ends at the full Shaw-doctrine
`BFMAgent`, the paper's rule-based expert (Sec 5). It was warm-started from run A's epoch-20 actor
([`../angle_tactic_A`](../angle_tactic_A)) and trained on branch `feat/resume-training` with every
fix listed below. It trained for the paper's full budget: **15.0M sim steps, epochs 21–229**. It was
paused by hand at epoch 204 (13.83M) and resumed exactly from its checkpoint on 2026-10-11.

## Headline result

**Training fights against the full BFM (`bfm_angles`):**

| Epochs | Fights | Win | Loss | Draw (timeout) | Crash |
|---|---|---|---|---|---|
| All top-rung epochs (93–228) | 2,720 | **58%** | 7% | 34% | 1% |
| Up to the pause (93–204) | 2,240 | 56% | 8% | 35% | 1% |
| After the resume (205–228) | 480 | **65%** | 4% | 31% | 0.2% |
| Last 20 epochs (209–228) | 400 | **64%** | 3% | 32% | 0.25% |
| Paper's MEOL angle tactic | — | ~60% | — | — | — |

The best policy (`policy_best.pt`) reached a rolling win rate of **0.72** over the last 100 fights at
epoch 202. No later epoch beat it; the resumed segment peaked at 0.71.

**Frozen-weight flights** (deterministic actions, the same 38 starts for each policy; see
[`../../visualizations`](../../visualizations)):

| Policy | Win | Timeout | Loss | Crash |
|---|---|---|---|---|
| `policy_best.pt` (epoch 202) | 23 | 12 | 3 | 0 |
| `snapshots/actor_epoch0204.pt` | 23 | 12 | 3 | 0 |
| `policy_final.pt` (15M) | 23 | 15 | **0** | 0 |

> **About these numbers:**
> - Training fights use stochastic actions and curriculum-chosen starts, which get harder as the agent improves.
> - The frozen-weight sets are only 38 fights each. A larger deterministic evaluation is still the right number for a report.
> - The kill rule differs from the paper's: here a side must hold the other inside its WEZ (150–1000 m, ±15°) for 0.5 s.

## How it learned

| Phase | Epochs | What happened |
|---|---|---|
| Inherited habit | 21–31 | Run A's actor flew inverted on negative g at idle throttle. It survived, but drifted away from even a slow straight target (mostly disengages) |
| Stay in the fight | 32–49 | Disengages fell from ~46% to 31%; first wins (~3%) |
| Finish fights | 50–75 | Wins against the straight target rose to 78% (epoch 75) |
| Climb the ladder | 75–92 | straight → turn4 (epoch 75) → turn6 (89) → bfm_energy (91) → **bfm_angles (92)**, with each rung taking fewer epochs than the last |
| Full BFM only | 93–204 | ~25% at first, ~55% plateau with ~40% draws, then peaks to 80% (epochs 123–127). A brief critic wobble at epochs 134–140 (Q-loss doubled) settled on its own |
| Resumed to 15M | 205–229 | Steady at 60–70% rolling, losses down to 3–4%. Q-loss at its lowest of the run (≈4.3) and α ≈ 0.005. Epoch 229 is a 1,630-step stub that ends exactly at the budget |

`training/training_diagnostics.png` shows all of this. Regenerate it with
`python plot_training.py models/angle_tactic_W/training --shade-until-epoch 20 --shade-label "run A"`.

## The fix chain (why earlier runs failed)

Each run stalled on one specific failure; each fix removed it.

| Run | Failure observed | Fix | Commit |
|---|---|---|---|
| paper setup | crashed 51 of 53 episodes: no attitude in obs, 50 Hz decisions too short-sighted | extended obs, 10 Hz decisions, ground-avoidance penalty, twin Q | `39bd8ba` |
| A | learned to point but never close; draws paid more than wins | wider potential scales + Auto-GCAS-style safety gate; 0.5 s gun kill ends the episode (win +1500 / loss −1000) | `be0a9c4`, `e8b4b47` |
| A | curriculum stuck on tail chases; replay forgot rare crashes | Eq.(9) win-rate labels, sin/cos angles, real WGAN-GP steps, 30% uniform starts; 1M replay | `db57c9e`, `09d6fb2` |
| C | α ran away (×31 in 3 epochs) under the large terminal rewards | SAC reward scale 0.1 | `1368c46` |
| C2 | flew inverted on push (−1.1 g) and could not turn; never won | negative-g penalty | `9626f3d` |
| C2 | no win was reachable against the full BFM, so the reward signal was never seen | opponent ladder, adaptive by win rate | `b532981`, `60169e4` |
| C3 | ran away once losses cost −1000 (disengage was free) | disengaging costs a loss; range 12 km | `6401ca4` |
| D / W | "easy" straight target (380 kt) was faster than the agent | bottom rung at 250 kt | `1fce16d` |
| W | crashed when a log CSV was open in Excel; peak policy overwritten | lock-tolerant logging; best-actor saving | `57acad1`, `7a381e2` |

**Fresh start vs warm start.** A run with the same fixes but a fresh actor (`training/comparison_D_fresh/`)
crashed 5, 5, 5, 9, 13, 15, 18 of 20 in epochs 0–6 (an over-banked "graveyard" spiral) and never won.
It was stopped in favour of this warm-started run. Under maneuver pressure from step 0, a fresh agent
did not learn to fly; run A's survival skill, though inverted, gave W a base it could correct.

## Configuration

| Setting | Value |
|---|---|
| Observation | `extended`, 21 dims |
| Decision rate | `action_repeat=5`: 10 Hz agent on a 50 Hz JSBSim F-16 |
| Actions | raw aileron / rudder / elevator / throttle (paper Eq. 44) |
| SAC | Table 2 hyperparameters, twin Q, `reward_scale=0.1`, 1M replay, one update per decision |
| Curriculum | WGAN-GP, Eq.(9) win-rate band [0.1, 0.9], 10 points × 2 episodes per epoch, 30% uniform |
| Opponents | ladder `straight(250 kt) → turn4 → turn6 → bfm_energy → bfm_angles`, ≥50% up / <10% down, 25% of points always vs `bfm_angles` |
| Hardware | RTX 5060 Laptop GPU, ~560 sim steps/s |

## Files

| File | Contents |
|---|---|
| `policy_best.pt` + `.json` | **best actor**: epoch 202, rolling win rate 0.72 (`sac.GaussianPolicy(21, 4)` state dict) |
| `policy_final.pt` | actor at the end of training (15.0M sim steps, epoch 229) |
| `state_15M_noreplay.pt` | resumable training state at 15M: actor, critics, targets, optimizers, α, curriculum, counters, RNG. The 275 MB replay buffer is left out, so it refills on resume |
| `wgan_generator.pt`, `wgan_critic.pt` | curriculum WGAN at the end of training |
| `snapshots/actor_epochNNNN.pt` | earlier actors: 198 (rolling 0.70), 200, 204 (where training was paused), 205–225 every 5 epochs |
| `training/angle_tactic_wgan_episodes.csv` | every episode, including run A's epochs 0–20: return, outcome, opponent, start condition |
| `training/sac_epochs.csv`, `wgan_epochs.csv` | per-epoch SAC stats (Q-loss, α, entropy, outcomes, rung) and curriculum stats |
| `training/stdout.log` | console log of all of W's segments (it was resumed several times) |
| `training/training_diagnostics.png` | the figure above |
| `training/comparison_D_fresh/` | logs of the failed fresh-start run |
| `flights/*.acmi`, `*_3d.png` | three late training wins against the full BFM: open the ACMI files in Tacview (with attitude) |

## Using it

Fly the best policy against the full BFM:

```python
import torch
from env import AngleTacticEnv
from sac import GaussianPolicy

pi = GaussianPolicy(21, 4)
pi.load_state_dict(torch.load("models/angle_tactic_W/policy_best.pt"))
env = AngleTacticEnv(adversary_type="ladder", opponent_level=4)     # 4 = bfm_angles
obs, _ = env.reset(seed=0, options={"init_condition": (3000.0, 0.0, 0.0)})   # tail chase: (Range m, AA deg, ATA deg)
done = False
while not done:
    with torch.no_grad():
        action = pi.sample(torch.as_tensor(obs).unsqueeze(0))[2].squeeze(0).numpy()   # deterministic
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(info["outcome"])   # "win" after ~58 s from this start
```

Or fly it over a whole set of starts, saving 3D plots, panels and Tacview files:

```bash
python fly_policy.py --policy models/angle_tactic_W/policy_final.pt --out visualizations/my_run --random-starts 30
```

Continue training past 15M. Critics and curriculum are restored; only the replay buffer starts empty.
`--total-env-steps` is the new total, so pick it above 15M:

```bash
python train_angle_wgan.py --run-dir runs/W_continued --resume models/angle_tactic_W/state_15M_noreplay.pt --device cuda --total-env-steps 20000000 --plot-every-n-episodes 50 --opponent-curriculum
```

## Known limitations

1. **Only small deterministic evaluations so far.** There are 38 frozen-weight fights per policy; see the note under the headline.
2. **About a third of fights end in draws, and the draws are a reward loophole.** All 15 timeouts of
   the final policy's 38 frozen flights are the same move, from defensive and beam starts:
   - the agent turns away and extends in a dive, from 4.5 km to 1–2.5 km, at about 320 m/s against the expert's 290;
   - the expert sits on its tail, 1.5–4.5 km behind: outside gun range, but well inside the 12 km disengage limit;
   - it runs about 30 km until the 120 s limit ends the fight.

   A timeout pays 0 and a loss costs −1000, so running away safely is the best-paying answer to a bad start.
   The next experiments: a timeout reward that depends on who holds the positional advantage at the end,
   a shorter disengage range, and a small cost per second of an undecided fight.
3. **Some wins exploit a habit of the opponent.** From a head-on start (3500 m, AA 180°, ATA 0°) the best
   actor wins in ~8 s. `BFMAgent`'s entry offsets its flight path before the merge, so for a moment
   only our nose is on target. That is a legal snapshot in this simulation, but a stronger or less
   predictable opponent would not hand it over.
4. **The other tactics are in sibling folders.** [`../angle_tactic_S`](../angle_tactic_S) (snapshot) and
   [`../angle_tactic_E`](../angle_tactic_E) (energy) were trained the same way. The option-selection layer
   (Algorithm 3) is `train_options.py`.
5. **Single run, single seed.** The training curve swings by ±0.2 over tens of epochs (as the paper's Fig. 4 also shows).
