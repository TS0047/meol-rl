# LAG baseline actor (via HHMARL 3D)

`baseline_actor.pt` is the only set of trained weights published in
[IDSIA-papers/HHMARL_AirCombat](https://github.com/IDSIA-papers/HHMARL_AirCombat)
(Selmonaj et al., *Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent RL*,
IEEE ICA 2025). It lives there as `env/base_actor/baseline_actor.pt`.

**What it is:**
- The pretrained JSBSim F-16 heading/altitude/speed autopilot from the LAG (Light Aircraft Game) project.
- HHMARL uses it, through its `TrackModel`, as the scripted opponent of its curriculum levels 1–2.
- HHMARL's own trained skills (cannon, track, escape, commander) are **not** published. Its `runs/` folder is created by training.

**Why it isn't committed:** the HHMARL repository has no licence file, so this repo doesn't redistribute the weights. Fetch them with:

```bash
python demo/lag_baseline.py --fetch
```

| | |
|---|---|
| Source | `https://raw.githubusercontent.com/IDSIA-papers/HHMARL_AirCombat/b300fab22ca6b30312c5e59029841434ae9dec04/env/base_actor/baseline_actor.pt` |
| Size | 558,234 bytes |
| SHA-256 | `54e9c996d981aa1fa82c51664a1758afd04fee0b5539dd6ccb128795149dcf69` (checked on fetch) |
| Content | a state dict of 22 tensors, 137,753 parameters: MLP 12-128-128 → GRU 128 → 4 heads (41/41/41/30 bins) |

**How the demo uses it** (`demo/lag_baseline.py`):
- **Loading:** `torch.load(weights_only=True)`, so only tensors are read and no code from the file runs. The network is re-created in this repo; their `baseline_actor.py` is not used.
- **Pilot logic:** HHMARL's pursuit logic, which points at the bandit and matches its altitude and speed. It decides at 5 Hz, LAG's training rate, and is flown as the demo's **"LAG pursuit (HHMARL)"** pilot in either seat.
- **Record:** against the rule-based expert from the 38 standard starts, it wins 20, draws 9 and loses 9.
