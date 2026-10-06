# MEOL angle tactic: SAC + WGAN curriculum on a JSBSim F-16

Reproduction of the **angle-tactic** low-level policy from Li et al., *Hierarchical Reinforcement Learning
with Automatic Curriculum Generation for UCAV Tactical Decision-Making in Autonomous Air Combat*,
**Drones 2025, 9, 384** (MEOL): within-visual-range 1v1 gun combat between two JSBSim F-16s, a SAC
intra-option policy (Algorithm 1) trained under a WGAN-GP automatic curriculum (Algorithm 2) against a
Shaw-doctrine rule-based expert.

**Result so far** ([`models/angle_tactic_W`](models/angle_tactic_W)): against the full rule-based expert the
trained policy wins **61%** of fights over its last 20 training epochs (414 fights; paper ≈ 0.6), loses 7%,
and draws the rest. That's at 13.8M of the paper's 15M sim steps. A deterministic evaluation is still to be run.

## Layout

| File | Role |
|---|---|
| `env.py` | Gymnasium environment: two JSBSim F-16s at 50 Hz, Eq.(8) geometry, 0.5 s gun-kill WEZ, observation, action repeat |
| `reward.py` | Eqs.(28)–(41) shaping and regularisation, plus ground-safety gate and negative-g penalty |
| `sac.py` | twin-Q SAC (Eqs. 14–20), reward scale, full save/load |
| `wgan.py` | WGAN-GP curriculum over (Range, AA, ATA), Eq.(9) win-rate labelling |
| `adversary.py` | `BFMAgent` adapter and the opponent ladder |
| `f16_bfm_agent.py`, `dogfight_sim.py` | Shaw-doctrine rule-based BFM pilot and a standalone two-agent dogfight sim |
| `train_angle_wgan.py` | training loop: curriculum, opponent ladder, checkpoints, exact resume |
| `models/` | trained runs with weights, logs and a README each |

## Train

```bash
python train_angle_wgan.py --device cuda --opponent-curriculum --run-dir runs/my_run --plot-every-n-episodes 50
```

Useful flags:

| Flag | Effect |
|---|---|
| `--resume <checkpoint or model folder>` | exact resume from a checkpoint file, or actor warm start from a model folder |
| `--obs-mode paper --action-repeat 1 --no-terminate-on-kill` | the paper's literal interface |
| `--uniform-frac 1.0` | the no-curriculum ablation (Sec 5.4) |

`python train_angle_wgan.py -h` lists everything.

## Models

| Run | What it is |
|---|---|
| [`angle_tactic_A`](models/angle_tactic_A) | first run after the crash fixes: survives and defends, rarely attacks (3–5% wins) |
| [`angle_tactic_W`](models/angle_tactic_W) | warm-started from A with every fix: climbs the full opponent ladder, ~60% wins vs the expert |

The W README explains the chain of failures and fixes between the two, and what differs from the paper.
