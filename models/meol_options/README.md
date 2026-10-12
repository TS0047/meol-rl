# MEOL option layer over the three frozen tactics

The high level of MEOL (Algorithm 3, Eqs. 21–25), trained by [`train_options.py`](../../train_options.py).
It works like this:
- The three tactic policies are loaded **frozen**: angle W, snapshot S and energy E, the `policy_final.pt`
  of each (see `tactics.json`).
- A maximum-entropy option-selection policy π(ω_t | s_t, ω_{t−1}) decides **once per second** which tactic
  flies the aircraft. Seeing the previous tactic folds the paper's termination function into the policy (Eq. 4).
- It is trained with discrete soft actor-critic on the **sparse combat outcome only** (Eq. 42):
  win +1.5, loss / crash / disengage −1, timeout 0. That's the tactics' R_goal in units of 1000.
- The opponent is the full BFM expert (`bfm_angles`). Starts are uniform over Table 1, with no curriculum,
  as in Algorithm 3. The budget is 15M sim steps, as in the paper's Fig. 13.

## Headline result

**Training fights** (stochastic option choice, uniform random starts):

| | Fights | Win | Loss | Draw (timeout) |
|---|---|---|---|---|
| All 386 epochs | 7,720 | 75% | 1.7% | 23% |
| First quarter (0–3.8M steps) | 1,920 | 60% | 0.9% | 39% |
| Last 20 epochs | 400 | **81%** | 2.8% | 16% |
| Paper's MEOL | — | ~0.8 | — | — |

The best option policy reached a rolling win rate of **0.93** at epoch 323 (13.0M steps). The final policy's
rolling rate is 0.79.

**Frozen weights:** argmax option, deterministic tactics, the same 38 starts as every tactic in
[`../../visualizations`](../../visualizations).

| Agent | Win | Draw | Loss |
|---|---|---|---|
| Angle tactic alone | 23 | 15 | 0 |
| Snapshot tactic alone | 37 (about 24 clean) | 1 | 0 |
| Energy tactic alone | 30 | 4 | 4 |
| **MEOL option layer, final** | **29** | 9 | **0** |
| MEOL option layer, best (epoch 323) | 30 | 8 | 0 |

## What it learned

- **Mostly energy, some angle, little snapshot.** In the last 20 epochs it flew energy 61% of the time,
  angle 29% and snapshot 11%. Paid only for wins, it moved away from the tactic that loses most in
  training (snapshot, ~15%).
- **Commit offensively.** From offensive and neutral starts it picks energy and keeps it: 0–5 switches,
  winning in 7–30 s. It wins every start the energy tactic wins alone, except the beam start, which it draws.
- **Never lose defensively.** From the defensive starts where energy alone loses (4 of them) or draws, it
  switches between angle and energy every few seconds (50–70 switches in 120 s). The fight ends in a draw:
  it borrows the angle tactic's safe escape. **The option layer never loses a frozen fight.**
- **No floor exploit.** It does not use snapshot's trick of dragging the expert down to its ground-avoidance
  limit. Snapshot flies only 3–9% of those defensive fights, so they end in draws, not wins.

**Against the paper:**
- The paper reports MEOL beating each single tactic by 0.2–0.35.
- Here it **matches the best single tactic's wins** (energy: 29–30 of 38) and **removes its losses**
  (4 → 0), but does not add wins.
- Two reasons:
  - The energy tactic alone is already much stronger than the paper's (0.89 vs 0.5–0.6), which leaves little headroom.
  - Under R_goal a draw costs 0 and a loss −1, so the selector's best answer to a bad start is a safe draw.
    That's the same loophole as the angle tactic's.

## How it trained (`training/training_curves.png`)

| Steps | What happened |
|---|---|
| 0–5M | α decayed slowly at the paper's option-α learning rate (3e-5), so entropy stayed near its 1.1 maximum: near-random tactic mixing, 35–55 switches per fight, win rate 0.4–0.65 |
| ~5.2M | Once α was small enough for preferences to form, the win rate **jumped from ~0.55 to ~0.8** within a few epochs, and energy's share started to grow |
| 5–9M | Entropy fell to its 0.5 target, switches dropped to 10–20 per fight, and snapshot's share shrank |
| 9–15M | Steady at 0.75–0.9 (best 0.93), entropy held at 0.5 by α ≈ 5e-4 |

The paper's 3e-5 α learning rate dominates the timeline: for its first third, the run learns little. A 10×
larger α learning rate would likely reach the same plateau in a fraction of the steps.

## Files

| File | Contents |
|---|---|
| `options_final.pt` | option-selection policy at 15M steps (`train_options.OptionAgent.pi` state dict) |
| `options_best.pt` + `.json` | best rolling win rate: 0.93 at epoch 323 |
| `state_15M_noreplay.pt` | resumable state: policy, twin Q and targets, optimizers, α, counters, RNG, config; the replay buffer is left out |
| `tactics.json` | the frozen tactic files it was trained over |
| `stats.json` | the numbers above |
| `training/` | per-fight and per-epoch CSVs (outcome, tactic shares, switches), console log, `training_curves.png` |

## Using it

Fly it frozen over the standard 38 starts. Each fight is saved as a 3D plot, a tactic-over-time plot and a Tacview file:

```bash
python train_options.py --fly models/meol_options/options_final.pt --out visualizations/my_options --random-starts 30
```

The default `--angle/--snapshot/--energy` paths are the three `models/angle_tactic_*/policy_final.pt`, which this layer was trained over.

## Known limitations

1. **The draw loophole** carries over from R_goal: defensive starts end in safe draws. A timeout reward that
   depends on who holds the advantage at the end would push the selector to convert them.
2. **Chattering when indifferent.** In defensive fights both tactics lead to a draw worth ~0, so the argmax
   flips between them dozens of times. A small switching cost, or a minimum option duration, would smooth this.
3. **The snapshot tactic it selects from carries the floor exploit.** The option layer barely uses it, but
   retraining snapshot with symmetric floors would make all three tactics clean.
4. **Single run, single seed.**
