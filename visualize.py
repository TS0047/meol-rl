"""Reproduces the paper's Figure 5/6-style visualization from a logged trajectory."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(trajectory, out_path, title=""):
    if not trajectory:
        return
    t = np.array([r["t"] for r in trajectory])
    self_pos = np.array([r["self_pos"] for r in trajectory])
    adv_pos = np.array([r["adv_pos"] for r in trajectory])
    AA = np.array([r["AA"] for r in trajectory])
    ATA = np.array([r["ATA"] for r in trajectory])
    HCA = np.array([r["HCA"] for r in trajectory])
    Range = np.array([r["Range"] for r in trajectory])
    Prox = np.array([r["Proximity"] for r in trajectory])
    E_self = np.array([r["E_self"] for r in trajectory])
    E_adv = np.array([r["E_adv"] for r in trajectory])

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(title)

    ax3d = fig.add_subplot(2, 4, 1, projection="3d")
    ax3d.plot(self_pos[:, 0], self_pos[:, 1], self_pos[:, 2], "b-", label="self fighter")
    ax3d.plot(adv_pos[:, 0], adv_pos[:, 1], adv_pos[:, 2], "r-", label="adversary")
    ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)")
    # z-label as the title: a 3D axes draws its z-label outside its own box, onto the AA panel
    ax3d.set_title("trajectory (vertical axis: altitude, m)", fontsize=8)
    ax3d.legend(fontsize=7)

    specs = [
        ("AA (deg)", AA), ("ATA (deg)", ATA), ("HCA (deg)", HCA),
        ("Range (m)", Range), ("Proximity (m/s)", Prox),
    ]
    for i, (label, series) in enumerate(specs, start=2):
        ax = fig.add_subplot(2, 4, i)
        ax.plot(t, series, "b-")
        ax.set_xlabel("time (s)"); ax.set_ylabel(label)

    ax = fig.add_subplot(2, 4, 7)
    ax.plot(t, E_self, "b-", label="self")
    ax.plot(t, E_adv, "r-", label="adversary")
    ax.set_xlabel("time (s)"); ax.set_ylabel("Energy (m)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_3d_trajectory(trajectory, out_path, title=""):
    """Standalone 3D flight-path plot (self vs adversary), larger and
    lighter-weight than the full 7-panel plot_trajectory -- used for the
    per-Nth-episode training-progress snapshots."""
    if not trajectory:
        return
    self_pos = np.array([r["self_pos"] for r in trajectory])
    adv_pos = np.array([r["adv_pos"] for r in trajectory])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(self_pos[:, 0], self_pos[:, 1], self_pos[:, 2], color="tab:blue", lw=2, label="self fighter")
    ax.plot(adv_pos[:, 0], adv_pos[:, 1], adv_pos[:, 2], color="tab:red", lw=2, label="adversary")
    ax.scatter(*self_pos[0], color="tab:blue", marker="s", s=50, label="self start")
    ax.scatter(*adv_pos[0], color="tab:red", marker="s", s=50, label="adversary start")
    ax.scatter(*self_pos[-1], color="tab:blue", marker="^", s=70)
    ax.scatter(*adv_pos[-1], color="tab:red", marker="^", s=70)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


OUTCOME_COLORS = {"win": "tab:blue", "loss": "tab:red", "timeout": "dimgray", "crash": "darkorange",
                  "disengaged": "purple"}


def plot_overview(entries, out_path, title=""):
    """Contact sheet of many engagements: one small 3D plot per (label, trajectory, outcome),
    titled in the outcome's colour. Every 5th logged point is drawn (10 Hz is plenty here)."""
    if not entries:
        return
    cols = 6 if len(entries) > 20 else 5
    rows = -(-len(entries) // cols)
    fig = plt.figure(figsize=(3.1 * cols, 3.0 * rows))
    for i, (label, trajectory, outcome) in enumerate(entries):
        s = np.array([r["self_pos"] for r in trajectory[::5]])
        a = np.array([r["adv_pos"] for r in trajectory[::5]])
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color="tab:blue", lw=1.2)
        ax.plot(a[:, 0], a[:, 1], a[:, 2], color="tab:red", lw=1.2)
        ax.scatter(*s[0], color="tab:blue", marker="s", s=12)
        ax.scatter(*a[0], color="tab:red", marker="s", s=12)
        ax.set_title(f"{label}\n{outcome.upper()}", fontsize=8, color=OUTCOME_COLORS.get(outcome, "black"))
        ax.tick_params(labelsize=5, pad=0)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def plot_win_rate(csv_path, out_path):
    import csv as _csv
    episodes, win_rates = [], []
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            win_rates.append(float(row["win_rate_100"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(episodes, win_rates, "b-")
    ax.set_xlabel("episode"); ax.set_ylabel("success rate (rolling 100)")
    ax.set_title("Angle tactic training: rolling win rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)