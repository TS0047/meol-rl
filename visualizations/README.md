# Frozen-policy engagements

Each folder is one trained policy flown by [`fly_policy.py`](../fly_policy.py) with **frozen weights**:
- eval mode and no gradients;
- deterministic actions: tanh of the Gaussian mean, no exploration noise;
- seeded BFM randomness, so every run is repeatable.

Every folder uses the same **38 starts**: 8 named Table 1 geometries plus 30 seeded random ones
(`--random-starts 30 --seed 2026`). The opponent is the full Shaw-doctrine `BFMAgent` (`bfm_angles`).
That makes the folders directly comparable, start by start, through their `summary.csv`.

## All three tactics at a glance

| Folder | Tactic, policy | Win | Timeout | Loss | Median win time |
|---|---|---|---|---|---|
| [`W_final_15M`](W_final_15M) | angle, final (15M) | 23 | 15 | 0 | 12.3 s |
| [`S_final`](S_final) | snapshot, final (15M) | **37** | 1 | 0 | 18.8 s |
| [`S_best`](S_best) | snapshot, best (epoch 176) | 32 | 5 | 1 | 16.6 s |
| [`E_final`](E_final) | energy, final (15M) | 30 | 4 | 4 | 13.2 s |
| [`E_best`](E_best) | energy, best (epoch 276) | 30 | 0 | 8 | 14.4 s |
| [`O_meol_final`](O_meol_final) | **MEOL option layer** over the three final tactics, final | 29 | 9 | **0** | 13.2 s |
| [`O_meol_best`](O_meol_best) | MEOL option layer, best (epoch 323) | 30 | 8 | 0 | 13.3 s |

The `O_meol_*` folders come from `train_options.py --fly`. Each fight has a `_options.png`, the tactic
in use over time, in place of the panels plot.

**What separates them:**
- **Angle** never loses, but escapes every defensive start into a draw. See "What the timeouts are" below.
- **Snapshot** wins almost everything, but 13 of its 37 wins exploit the BFM's ground-avoidance rule.
  It drags the expert into a dive below ~1,100 m, where `BFMAgent._safety` forces a predictable
  wings-level pull-up, then turns onto it. About 24 wins are clean. Details:
  [`models/angle_tactic_S`](../models/angle_tactic_S).
- **Energy** wins every fight it wins in under 30 s and above 1,100 m, with no exploit. It loses some
  defensive starts instead of escaping them.
- **Shared:** all three win the head-on start in about 8 s, through the BFM's offset entry at the merge.
- **MEOL option layer** flies energy most of the time and keeps it from offensive and neutral starts. From
  defensive starts it switches between angle and energy and ends in a draw. It wins every start energy wins
  except the beam start, and **never loses**. See [`models/meol_options`](../models/meol_options).

## Angle tactic (run W) in detail

These three are policies from [`models/angle_tactic_W`](../models/angle_tactic_W).

| Folder | Policy | Win | Timeout | Loss | Crash | Median win time |
|---|---|---|---|---|---|---|
| [`W_best_epoch202`](W_best_epoch202) | `policy_best.pt`: epoch 202, best training rolling win rate (0.72) | 23 | 12 | 3 | 0 | 15.8 s |
| [`my_run`](my_run) | epoch 204 (`snapshots/actor_epoch0204.pt`) | 23 | 12 | 3 | 0 | 13.0 s |
| [`W_final_15M`](W_final_15M) | `policy_final.pt`: end of training, 15M sim steps | **23** | 15 | **0** | 0 | **12.3 s** |

**Between the three policies:**
- They agree on 29 of the 38 starts. All three win the same 19 and time out on the same 10.
- The final policy never loses, and it wins faster. Against the best policy, it turns that policy's 3 losses into wins but 3 of its wins into timeouts. Compare the `summary.csv` files start by start.

### What the timeouts are

All 15 of the final policy's timeouts are the same manoeuvre. From a defensive or beam start the
agent **turns away and extends in a dive**:
- It descends from 4.5 km to 1–2.5 km at about 320 m/s, against the expert's 290 m/s.
- The expert sits on its tail: our track is 170–176° off the line of sight and the expert's is about 180°.
- The range stays at 1.5–4.5 km. That is beyond the 1 km gun envelope and well inside the 12 km disengage limit.
- It runs about 30 km of ground track in 120 s, until the time limit ends the fight.

This is a reward loophole, not a skill. A timeout pays 0 and a loss costs −1000, so a safe extension
is the best-paying answer to a bad start. Two ways to close it:
- a time-out reward that depends on who holds the positional advantage at the end;
- a shorter disengage range.

## Files in each folder

| File | Contents |
|---|---|
| `overview_3d.png` | contact sheet: all 38 trajectories, titled in the outcome's colour |
| `NN_<start>_<outcome>_3d.png` | 3D trajectories. Blue is the agent, red the expert; squares are starts and triangles are ends |
| `NN_<start>_<outcome>_panels.png` | paper Fig. 5/6-style panels: 3D, AA, ATA, HCA, range, closure, energy |
| `NN_<start>_<outcome>.acmi` | the same fight for **Tacview**: two F-16s with attitude, over open ocean at 0°N 0°E |
| `summary.csv`, `summary.md` | start geometry, outcome and duration of every fight |

## Regenerate

```bash
python fly_policy.py --policy models/angle_tactic_W/policy_final.pt --out visualizations/W_final_15M --random-starts 30
```

Older `.acmi` files from before the Tacview fix can be converted in place with `python export.py --upgrade <folder>`.
