"""
Training-diagnostics figure (4x2 small multiples) from a run's training logs.

Reads angle_tactic_wgan_episodes.csv, sac_epochs.csv and wgan_epochs.csv (as written by
train_angle_wgan.py into <run>/logs, or as packaged in models/<run>/training) and draws:
rolling win rate, opponent-ladder rung, outcome shares, win rate vs the full BFM,
critic Q-loss, alpha, policy entropy and the size of the Eq.(9) curriculum pool.

  python plot_training.py models/angle_tactic_W/training --shade-until-epoch 20 --shade-label "run A"
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1 = "#2a78d6"
CAT = {"win": "#2a78d6", "loss": "#eb6834", "timeout": "#1baf7a", "crash": "#eda100", "other": "#e87ba4"}
RUNGS = ["straight", "turn4", "turn6", "bfm_energy", "bfm_angles"]


def _f(row, key):
    return float(row[key]) if row.get(key) not in ("", "None", None) else np.nan


def plot_training(training_dir, out_path, title, shade_until_epoch=None, shade_label=""):
    eps = list(csv.DictReader(open(os.path.join(training_dir, "angle_tactic_wgan_episodes.csv"))))
    sac = list(csv.DictReader(open(os.path.join(training_dir, "sac_epochs.csv"))))
    wg = list(csv.DictReader(open(os.path.join(training_dir, "wgan_epochs.csv"))))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold"})
    fig, axs = plt.subplots(4, 2, figsize=(14, 15), facecolor=SURF)
    for ax in axs.flat:
        ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # (a) rolling win rate per episode, an inherited history optionally shaded
    ax = axs[0, 0]
    ep_n = np.array([int(r["episode"]) for r in eps]); wr = np.array([_f(r, "win_rate_100") for r in eps])
    if shade_until_epoch is not None:
        end = max(int(r["episode"]) for r in eps if int(r["epoch"]) <= shade_until_epoch)
        ax.axvspan(0, end, color=GRID, alpha=0.6, lw=0)
        ax.text(end / 2, 0.93, shade_label, ha="center", va="top", color=INK2, fontsize=8)
    ax.plot(ep_n, wr, color=S1, lw=2)
    ax.axhline(0.6, color=MUTED, lw=1, ls="--")
    ax.text(ep_n.max() * 0.15, 0.615, "paper MEOL ≈ 0.6", ha="left", va="bottom", color=INK2, fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("Rolling win rate (last 100 fights)", loc="left")
    ax.set_xlabel("episode"); ax.set_ylabel("win rate")

    # (b) opponent rung per epoch
    ax = axs[0, 1]
    e = np.array([int(r["epoch"]) for r in sac]); lvl = np.array([_f(r, "opp_level") for r in sac])
    ax.step(e, lvl, where="post", color=S1, lw=2)
    ax.set_yticks(range(len(RUNGS))); ax.set_yticklabels(RUNGS); ax.set_ylim(-0.4, len(RUNGS) - 0.6)
    ax.set_title("Opponent-ladder rung", loc="left"); ax.set_xlabel("epoch")

    # (c) outcome shares per epoch (stacked)
    ax = axs[1, 0]
    bottom = np.zeros(len(sac)); tot = np.array([sum(_f(r, k) for k in CAT) for r in sac])
    for k, col in CAT.items():
        v = np.array([_f(r, k) for r in sac]) / np.maximum(tot, 1)
        ax.fill_between(e, bottom, bottom + v, color=col, lw=0, label=k, step="post"); bottom += v
    ax.set_ylim(0, 1); ax.set_title("Episode outcomes per epoch (share)", loc="left", pad=22)
    ax.set_xlabel("epoch"); ax.set_ylabel("share of episodes")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0),
              borderaxespad=0.2, handlelength=1.2)

    # (d) win rate vs the full BFM per epoch + 10-epoch average
    ax = axs[1, 1]
    tw, tn = np.array([_f(r, "top_wins") for r in sac]), np.array([_f(r, "top_eps") for r in sac])
    ok = tn > 0
    ax.plot(e[ok], tw[ok] / tn[ok], "o", color=S1, ms=3, alpha=0.45, label="per epoch")
    k = 10
    cw, cn = np.convolve(tw, np.ones(k), "valid"), np.convolve(tn, np.ones(k), "valid")
    ax.plot(e[k - 1:][cn > 0], (cw / np.maximum(cn, 1))[cn > 0], color=S1, lw=2, label="10-epoch average")
    ax.axhline(0.6, color=MUTED, lw=1, ls="--")
    ax.set_ylim(0, 1); ax.set_title("Win rate vs the full BFM expert (top rung)", loc="left")
    ax.set_xlabel("epoch"); ax.set_ylabel("win rate")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")

    # (e) critic Q-loss, (f) alpha, (g) entropy, (h) curriculum pool
    def line(ax, x, y, t, yl):
        ax.plot(x, y, color=S1, lw=2); ax.set_title(t, loc="left"); ax.set_xlabel("epoch"); ax.set_ylabel(yl)
    line(axs[2, 0], e, [_f(r, "q_loss") for r in sac], "Critic Q-loss", "MSE")
    line(axs[2, 1], e, [_f(r, "alpha") for r in sac], "Entropy temperature α", "α"); axs[2, 1].set_yscale("log")
    line(axs[3, 0], e, [_f(r, "entropy") for r in sac], "Policy entropy (target −4)", "entropy")
    axs[3, 0].axhline(-4, color=MUTED, lw=1, ls="--")
    first = (shade_until_epoch + 1) if shade_until_epoch is not None else 0
    wg = [r for r in wg if int(r["epoch"]) >= first]
    line(axs[3, 1], [int(r["epoch"]) for r in wg], [_f(r, "feasible_buffer_size") for r in wg],
         "Eq.(9) medium-difficulty pool (curriculum)", "start conditions")

    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=110, facecolor=SURF)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("training_dir", help="folder with the three training CSVs")
    p.add_argument("--out", default=None, help="default: <training_dir>/training_diagnostics.png")
    p.add_argument("--title", default=None)
    p.add_argument("--shade-until-epoch", type=int, default=None,
                   help="shade episodes up to this epoch, e.g. a warm start's inherited history")
    p.add_argument("--shade-label", default="")
    args = p.parse_args()
    sac = list(csv.DictReader(open(os.path.join(args.training_dir, "sac_epochs.csv"))))
    steps = int(sac[-1]["env_steps"]) / 1e6
    title = args.title or f"Training diagnostics: {args.training_dir} (to {steps:.2f}M sim steps)"
    out = args.out or os.path.join(args.training_dir, "training_diagnostics.png")
    plot_training(args.training_dir, out, title, args.shade_until_epoch, args.shade_label)
    print("figure:", out)


if __name__ == "__main__":
    main()
