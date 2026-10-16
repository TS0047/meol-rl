# MEOL tactics: SAC + WGAN curriculum on a JSBSim F-16

Reproduction of Li et al., *Hierarchical Reinforcement Learning with Automatic Curriculum Generation for
UCAV Tactical Decision-Making in Autonomous Air Combat*, **Drones 2025, 9, 384** (MEOL). The setting is
within-visual-range 1v1 gun combat between two JSBSim F-16s against a Shaw-doctrine rule-based expert.
It has two levels:
- **Low level:** three SAC intra-option tactics (Algorithm 1): **angle**, **snapshot** and **energy**.
  Each is trained under a WGAN-GP automatic curriculum (Algorithm 2) and differs from the others only in
  its shaping reward.
- **High level:** a maximum-entropy option-selection policy (Algorithm 3) that switches between the frozen tactics.

## Results against the full rule-based expert

Each tactic was trained for the paper's full 15M sim steps.

| Tactic | Training: last 20 epochs (win / loss / draw) | Frozen weights, 38 fixed starts (win / draw / loss) | Paper |
|---|---|---|---|
| Angle ([`W`](models/angle_tactic_W)) | 64% / 3% / 32% | 23 / 15 / 0 | ~0.6 |
| Snapshot ([`S`](models/angle_tactic_S)) | 83% / 15% / 2% | 37 / 1 / 0 (about 24 clean, see below) | 0.4–0.5 |
| Energy ([`E`](models/angle_tactic_E)) | 89% / 1% / 10% | 30 / 4 / 4 | 0.5–0.6 |
| **MEOL option layer** ([`meol_options`](models/meol_options)) | 81% / 3% / 16% | **29 / 9 / 0** | ~0.8 |

**Two exploits to keep in mind:**
- **Angle's draws** are a reward loophole: a safe extension away from the expert, because a timeout costs
  0 and a loss −1000.
- **About a third of snapshot's wins** exploit the expert's ground-avoidance rule. It drags the expert into
  a dive until `BFMAgent._safety` forces a wings-level pull-up.

Both are documented in the model READMEs, and [`visualizations`](visualizations) compares all
frozen-weight flights start by start.

**Option layer (Algorithm 3):** [`models/meol_options`](models/meol_options).
- **What it learned:** it flies energy about 60% of the time, angle 30% and snapshot 10%. It commits to
  energy from offensive and neutral starts, and switches between angle and energy from defensive starts
  to reach a safe draw.
- **Against the single tactics:** it matches the best tactic's wins (energy, 29–30 of 38) and **never loses**,
  where energy alone loses 4. It does not add wins as the paper reports. The energy tactic is already strong
  here, and the draw loophole makes a safe draw the selector's best answer to a bad start.

## Layout

| File | Role |
|---|---|
| `env.py` | Gymnasium environment: two JSBSim F-16s at 50 Hz, Eq.(8) geometry, 0.5 s gun-kill WEZ, observation, action repeat; `tactic=` picks the reward |
| `reward.py` | Eqs.(28)–(41) shaping and regularisation for all three tactics, plus ground-safety gate and negative-g penalty |
| `sac.py` | twin-Q SAC (Eqs. 14–20), reward scale, full save/load |
| `wgan.py` | WGAN-GP curriculum over (Range, AA, ATA), Eq.(9) win-rate labelling |
| `adversary.py` | `BFMAgent` adapter and the opponent ladder |
| `f16_bfm_agent.py`, `dogfight_sim.py` | Shaw-doctrine rule-based BFM pilot and a standalone two-agent dogfight sim |
| `train_angle_wgan.py` | tactic training loop (`--tactic angle\|snapshot\|energy`): curriculum, opponent ladder, checkpoints, exact resume |
| `train_options.py` | option-selection layer (Algorithm 3) over the frozen tactics; `--fly` flies it frozen with option timelines |
| `fly_policy.py` | fly a trained tactic with frozen weights over a set of starts; saves 3D plots, panels, an overview and Tacview files |
| `export.py`, `visualize.py`, `plot_training.py` | Tacview ACMI writer with a toggleable WEZ layer (wireframe gun cones from `tacview/MEOL.WEZ.obj`, lock lines, outcome bookmarks; `--install-mesh` once, `--weapons on\|off` rewrites existing files), trajectory plots, training-diagnostics figure |
| `fg_replay.py` | replays `.acmi` fights in FlightGear 2024.1 at real time, or flies a policy live (`--live`): the agent's jet over native-fdm, the opponent as a multiplayer aircraft; `--speed`, `--loop`, `--launch` starts fgfs. Display only |
| `fg_play.py` | you fly against the BFM expert or a trained tactic in FlightGear: your stick comes in over native-ctrls, both jets are `env.py`'s JSBSim F-16s under the training rules; fights saved as `.acmi` |
| `flightgear/` | FlightGear add-on that both scripts load: under `--fdm=null` it feeds the F-16's HUD and gauges (speed, Mach, altitude, FLYUP) |
| `models/` | trained runs with weights, logs and a README each |
| `visualizations/` | frozen-weight engagements of every tactic, with a README comparing them |

## Fly against them

[`fg_play.py`](fg_play.py) puts you in the cockpit against the BFM expert or any trained tactic, in
FlightGear 2024.1 with the FGAddon F-16:

```bash
python fg_play.py --launch
python fg_play.py --launch --vs models/angle_tactic_E/policy_best.pt --start head_on
python fg_play.py --vs straight --start tail_chase --speed 0.5
```

- **Same fight as training:** both jets are `env.py`'s JSBSim `f16.xml` at 50 Hz, under the same kill rule
  (0.5 s inside the ±15°, 150–1,000 m gun cone), the 200 m floor and the 12 km disengage.
  FlightGear is only the cockpit, stick and screen.
- **Seats:** against a bot from the opponent ladder (`straight`, `turn4`, `turn6`, `bfm_energy`,
  `bfm_angles`) you fly the agent's seat. Against a policy `.pt` you take the adversary seat, and the policy
  decides every 0.1 s, as it trained.
- **`--start`:** a Table 1 start seen from your seat, e.g. `tail_chase` puts you 3 km behind the bandit; or `random`.
- **Controls:** FlightGear's own. Use a joystick plugged in before it starts, the keyboard (numpad or arrows,
  PgUp/PgDn throttle) or the mouse yoke (right-click). Trim and speed brake work too, and `P` pauses.
  `--speed 0.5` is slow motion and `--loop` gives a rematch.
- **Saved fights:** each one goes to `logs/fg_play/` as an `.acmi`, for Tacview or `fg_replay.py`.

## Train

The three tactics use the same recipe: a warm start from run A's actor, then the opponent ladder, to 15M sim steps.

```bash
python train_angle_wgan.py --run-dir runs/my_E --tactic energy --resume models/angle_tactic_A --opponent-curriculum --total-env-steps 15000000 --plot-every-n-episodes 50
```

Then the option layer over the frozen tactics:

```bash
python train_options.py --run-dir runs/my_options --angle models/angle_tactic_W/policy_final.pt --snapshot models/angle_tactic_S/policy_final.pt --energy models/angle_tactic_E/policy_final.pt
```

Useful flags for `train_angle_wgan.py`:

| Flag | Effect |
|---|---|
| `--tactic angle\|snapshot\|energy` | which shaping reward (Eq. 38 / 40 / 41). An exact resume keeps the checkpoint's tactic |
| `--resume <checkpoint or model folder>` | exact resume from a checkpoint file, or actor warm start from a model folder |
| `--device cpu --torch-threads 6` | CPU is as fast as the GPU for these small networks; with two runs side by side the GPU run is the slow one |
| `--obs-mode paper --action-repeat 1 --no-terminate-on-kill` | the paper's literal interface |
| `--uniform-frac 1.0` | the no-curriculum ablation (Sec 5.4) |

`python train_angle_wgan.py -h` and `python train_options.py -h` list everything.

## Models

| Run | What it is |
|---|---|
| [`angle_tactic_A`](models/angle_tactic_A) | first run after the crash fixes: survives and defends, rarely attacks (3–5% wins). The warm start for W, S and E |
| [`angle_tactic_W`](models/angle_tactic_W) | angle tactic: climbs the full opponent ladder, ~60% wins vs the expert. Its README explains the chain of failures and fixes |
| [`angle_tactic_S`](models/angle_tactic_S) | snapshot tactic, with a range-weighted AA term (the literal Eq. 40 taught it to dive away and crash) |
| [`angle_tactic_E`](models/angle_tactic_E) | energy tactic, the paper's Eq. 41 unchanged |
| [`meol_options`](models/meol_options) | MEOL option-selection layer over the final W, S and E tactics: 81% wins in training, 0 losses in frozen flights |
