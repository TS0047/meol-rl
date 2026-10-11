"""
Fly a trained angle-tactic policy with frozen weights and save every engagement.

The actor is loaded, switched to eval mode with gradients disabled, and acts
deterministically (tanh of the Gaussian mean, no exploration noise), so nothing
is learned or changed. Each engagement is flown against one opponent from
adversary.LADDER (default: the full BFM expert) and saved as
  <name>_3d.png      3D trajectories of both aircraft (visualize.plot_3d_trajectory)
  <name>_panels.png  paper Fig.5/6-style panels: 3D, AA, ATA, HCA, range, proximity, energy
  <name>.acmi        Tacview file with attitude
plus summary.csv / summary.md for the whole set.

  python fly_policy.py --policy models/angle_tactic_W/policy_best.pt --out visualizations/run_W_best
"""

import argparse
import csv
import os
from collections import Counter

import numpy as np
import torch

from adversary import LADDER
from env import AngleTacticEnv, OBS_DIMS
from export import write_acmi
from sac import GaussianPolicy
from visualize import plot_3d_trajectory, plot_overview, plot_trajectory

# Named starts in Table 1 terms (Range m, AA deg, ATA deg). AA: angle between the bandit's
# nose and the line of sight from us (0 = it points away, 180 = it points at us);
# ATA: our nose off the line of sight (0 = we point at it).
NAMED_STARTS = [
    ("tail_chase", (3000.0, 0.0, 0.0)),
    ("offensive_angle_off", (3000.0, 30.0, 20.0)),
    ("head_on", (4000.0, 180.0, 0.0)),
    ("beam_neutral", (3000.0, 90.0, 90.0)),
    ("crossing", (3500.0, 135.0, 45.0)),
    ("lateral_split", (3500.0, 90.0, -90.0)),
    ("defensive", (2500.0, 180.0, 180.0)),
    ("defensive_offset", (3000.0, 150.0, 150.0)),
]


def load_frozen_policy(path, obs_dim, device="cpu"):
    pi = GaussianPolicy(obs_dim, 4).to(device)
    pi.load_state_dict(torch.load(path, map_location=device))
    pi.eval()
    for p in pi.parameters():
        p.requires_grad_(False)
    return pi


def fly(pi, env, init_condition, seed):
    """One engagement with deterministic actions; returns (outcome, engagement length in s)."""
    env.adversary._rng = np.random.default_rng(seed)        # repeatable BFM randomness (pass side, jinks)
    obs, _ = env.reset(seed=seed, options={"init_condition": init_condition})
    with torch.no_grad():
        while True:
            action = pi.sample(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[2].squeeze(0).numpy()
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return info["outcome"], env.step_count / 50.0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", default="models/angle_tactic_W/policy_best.pt")
    p.add_argument("--out", default="visualizations/run_W_best")
    p.add_argument("--opponent", choices=LADDER, default="bfm_angles")
    p.add_argument("--random-starts", type=int, default=12, help="extra seeded random Table 1 starts")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--obs-mode", choices=list(OBS_DIMS), default="extended")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = AngleTacticEnv(max_steps=6000, log_trajectory=True, obs_mode=args.obs_mode,
                         adversary_type="ladder", opponent_level=LADDER.index(args.opponent))
    pi = load_frozen_policy(args.policy, OBS_DIMS[args.obs_mode])

    rng = np.random.default_rng(args.seed)
    starts = list(NAMED_STARTS) + [
        (f"random_{i + 1:02d}", (float(rng.uniform(1900, 5100)), float(rng.uniform(-180, 180)),
                                 float(rng.uniform(-180, 180)))) for i in range(args.random_starts)]

    rows, sheet = [], []
    for i, (name, ic) in enumerate(starts):
        outcome, secs = fly(pi, env, ic, seed=args.seed + i)
        tag = f"{i + 1:02d}_{name}_{outcome}"
        title = (f"{name}: start R={ic[0]:.0f} m, AA={ic[1]:.0f}, ATA={ic[2]:.0f} -> "
                 f"{outcome.upper()} at {secs:.1f} s vs {args.opponent}")
        plot_3d_trajectory(env.trajectory, os.path.join(args.out, f"{tag}_3d.png"), title=title)
        plot_trajectory(env.trajectory, os.path.join(args.out, f"{tag}_panels.png"), title=title)
        write_acmi(env.trajectory, os.path.join(args.out, f"{tag}.acmi"), title=title, opponent=args.opponent)
        rows.append({"n": i + 1, "start": name, "range_m": round(ic[0]), "aa_deg": round(ic[1]),
                     "ata_deg": round(ic[2]), "outcome": outcome, "seconds": round(secs, 1), "files": tag})
        sheet.append((f"{i + 1:02d} {name} ({secs:.0f} s)", list(env.trajectory), outcome))
        print(f"{i + 1:2d}/{len(starts)} {name:20s} -> {outcome:10s} {secs:6.1f} s", flush=True)

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = Counter(r["outcome"] for r in rows)
    plot_overview(sheet, os.path.join(args.out, "overview_3d.png"),
                  title=f"{args.policy} vs {args.opponent}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Frozen-policy engagements\n\npolicy `{args.policy}`, opponent `{args.opponent}`, "
                f"deterministic actions, {len(rows)} engagements\n\n")
        f.write("**Outcomes:** " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + "\n\n")
        f.write("| # | start | Range (m) | AA | ATA | outcome | time (s) | files |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['n']} | {r['start']} | {r['range_m']} | {r['aa_deg']} | {r['ata_deg']} | "
                    f"{r['outcome']} | {r['seconds']} | `{r['files']}_3d.png` |\n")
    print("outcomes:", dict(c))


if __name__ == "__main__":
    main()
