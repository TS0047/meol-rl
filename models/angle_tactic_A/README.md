# Run A — angle-tactic policy (crash fixes, paper reward scales)

SAC intra-option policy for the **angle tactic** (Algorithm 1), trained under the
WGAN-GP automatic curriculum (Algorithm 2) against the Shaw rule-based BFM adversary.
Checkpoint taken at the end of curriculum epoch 20, after which training was stopped.

## What this run is

The first training run after the stability fixes in commit `39bd8ba`:

| Setting | Value |
|---|---|
| Observation | `obs_mode="extended"`, 21 dims (Eq. 43 + attitude, body rates, body-frame bandit LOS/heading, vertical speed) |
| Decision rate | `action_repeat=5`, so the agent acts at 10 Hz on a 50 Hz JSBSim sim |
| Reward | Eq. 38 angle tactic, paper-scale potentials (`c_r=400 m`, `c_AA=c_ATA=30°`), time-to-deck `P_ground` (`k_ground=2`), `CRASH_PENALTY=1000` |
| SAC | Table 2 hyperparameters, twin Q with min-backup |
| Curriculum | WGAN-GP, 10 points/epoch × 2 episodes, reward-percentile feasible band |
| Episode | ≤ 6000 sim steps (120 s), BFM adversary in "angles" style |
| Budget | 2,328,933 sim steps (20 epochs), seed 0, CPU, ~75 min wall clock |

Equivalent command (it ran through a logging wrapper that only adds the files in `training/`):

```bash
python train_angle_wgan.py --total-env-steps 3000000 --plot-every-n-episodes 25
```

## Files

| File | Contents |
|---|---|
| `policy.pt` | `sac.GaussianPolicy(21, 4)` state dict: the trained actor |
| `wgan_generator.pt`, `wgan_critic.pt` | curriculum WGAN at epoch 20 |
| `training/angle_tactic_wgan_episodes.csv` | every episode: return, outcome, rolling win rate, start condition (the last 3 rows are from epoch 21, after the checkpoint) |
| `training/wgan_epochs.csv`, `training/curriculum_epochs.jsonl` | per-epoch curriculum state: critic/generator losses, feasible buffer, generator coverage and drift |
| `training/sac_updates.csv` | Q-loss, policy loss, α and mean Q every 2000 updates |
| `training/training_diagnostics.png` | all of the above in one figure |

Only the actor is saved. The training script does not save the Q-networks, α or the replay
buffer, so fine-tuning from here restarts the critics.

## Results

Training outcomes by phase (20 episodes per epoch):

| Epochs | Crash | Loss | Timeout | Win |
|---|---|---|---|---|
| 0–1 | **82%** | 15% | 2% | 0% |
| 2–8 | 4% | **84%** | 8% | 5% |
| 9–20 | 7% | 46% | **43%** | 3% |

Deterministic evaluation of `policy.pt` (15 episodes per set):

| Start conditions | Win | Loss | Timeout | Crash |
|---|---|---|---|---|
| Curriculum-like: 3300–3700 m, \|AA\|, \|ATA\| < 30° | 1 | 2 | 12 | 0 |
| Uniform Table 1: 1900–5100 m, AA/ATA ±180° | 0 | 1 | 12 | 2 |

**Reading it:** the fixes worked for flying, and crashes fell from 82% to single digits within two
epochs. The agent then learned to *defend*: losses fell from 84% to 46% and draws (timeouts) rose to 43%.
It did not learn to *attack*. Wins stayed at 3–5%, against the paper's ~0.6 after 1.5e7 steps.

## Known limitations

1. **No gradient toward the WEZ.** With `c_r=400 m` and 30° angle scales, the main reward
   term is ~0.003 when pointing at the bandit from 3500 m vs ~2.2 inside the WEZ. Replays show the
   policy points at the bandit (ATA ≈ 0°) but never closes inside 1000 m. Wider scales plus an
   Auto-GCAS-style safety gate are on branch `exp/b3-recovery-gate` (unvalidated).
2. **The curriculum is stuck.** The generator only ever proposes tail-chase starts (range ≈ 3500 m,
   |AA|, |ATA| ≲ 60°), drifting 0.23 in weight norm over 20 epochs. That is why it never trains on
   head-on or defensive setups. The critic's Wasserstein estimate is negative from epoch 4 onwards.
3. The win label is "first one-sided WEZ entry" and episodes are not ended by it, so win rates
   are not directly comparable with the paper's.

## Loading

```python
import torch
from env import AngleTacticEnv
from sac import GaussianPolicy

pi = GaussianPolicy(21, 4)
pi.load_state_dict(torch.load("models/angle_tactic_A/policy.pt"))
env = AngleTacticEnv(obs_mode="extended", action_repeat=5)

obs, _ = env.reset(options={"init_condition": (3500.0, -15.0, 0.0)})   # (Range m, AA deg, ATA deg)
done = False
while not done:
    with torch.no_grad():
        action = pi.sample(torch.as_tensor(obs).unsqueeze(0))[2].squeeze(0).numpy()  # deterministic
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(info["outcome"])
```
