"""
Demo engine: fly N engagements between any two of the five pilots, either seat, and write one
folder with every fight's plots and Tacview file plus a report.

Pilots (best checkpoints):
  angle, snapshot, energy  SAC tactic policies, models/angle_tactic_{W,S,E}/policy_best.pt
  meol                     MEOL option layer, models/meol_options/options_best.pt, over the
                           three tactics it was trained with (models/meol_options/tactics.json)
  bfm                      the rule-based Shaw-doctrine expert (BFMAgent, angles fight)

Both aircraft are JSBSim F-16s. Each seat sees the fight from its own side: a trained policy in
the opponent's seat gets the same 21-dim observation it was trained on, computed with the roles
swapped. Policies decide every 5 sim steps (10 Hz, as trained), MEOL picks a tactic every
second, the BFM acts at 50 Hz. Iterations start from the same (Range, AA, ATA) and differ through
stochastic actions -- each policy samples from its learned distribution -- and a fresh BFM seed.
"""

import csv
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
if RL not in sys.path:
    sys.path.insert(0, RL)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from adversary import BFMAdversary  # noqa: E402
from env import OBS_DIMS, AngleTacticEnv, _read_state  # noqa: E402
from export import write_acmi, wez_states  # noqa: E402
from geometry import combat_geometry, proximity  # noqa: E402
from sac import GaussianPolicy  # noqa: E402
from train_options import NONE, OPTIONS, OptionAgent  # noqa: E402
from visualize import plot_3d_trajectory, plot_overview, plot_trajectory  # noqa: E402

PILOTS = {
    "angle": {"label": "Angle tactic", "short": "Angle", "path": "models/angle_tactic_W/policy_best.pt"},
    "snapshot": {"label": "Snapshot tactic", "short": "Snapshot", "path": "models/angle_tactic_S/policy_best.pt"},
    "energy": {"label": "Energy tactic", "short": "Energy", "path": "models/angle_tactic_E/policy_best.pt"},
    "meol": {"label": "MEOL option layer", "short": "MEOL", "path": "models/meol_options/options_best.pt"},
    "bfm": {"label": "Rule-based BFM expert", "short": "BFM", "path": None},
}
OBS_DIM = OBS_DIMS["extended"]
DECISION_EVERY = 5                 # sim steps per policy decision (training: action_repeat = 5)
OPTION_EVERY = 10                  # policy decisions per MEOL option decision (train_options default)
START_ALT_M, START_SPEED_MPS = 4500.0, 200.0   # env.reset's fixed start altitude and speed
OUTCOME_COLORS = {"win": "#2a78d6", "loss": "#eb6834", "timeout": "#7f7f7f", "crash": "#eda100",
                  "disengaged": "#9b59b6", "adversary_crash": "#1baf7a"}


def _rl(path):
    return os.path.join(RL, path)


def _load_actor(path):
    pi = GaussianPolicy(OBS_DIM, 4)
    pi.load_state_dict(torch.load(_rl(path), map_location="cpu"))
    pi.eval()
    for p in pi.parameters():
        p.requires_grad_(False)
    return pi


def seat_obs(env, me, other):
    """The observation a policy in this seat was trained on, with `me` as the ego aircraft."""
    geom = combat_geometry(me["pos"], me["e_hat"], other["pos"], other["e_hat"])
    prox = proximity(me["pos"], me["v"], other["pos"], other["v"])
    return env._obs(geom, prox, me, other)


def _policy_action(pi, obs, stochastic):
    """[aileron, rudder, elevator, throttle] with throttle in 0..1 (the 'native' convention)."""
    with torch.no_grad():
        a, _, mean = pi.sample(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    a = (a if stochastic else mean).squeeze(0).numpy().astype(np.float32)
    a[3] = (a[3] + 1.0) / 2.0
    return a


class TacticPilot:
    def __init__(self, key, stochastic):
        self.pi, self.stochastic = _load_actor(PILOTS[key]["path"]), stochastic

    def reset(self, seed):
        self.k, self.action = 0, None

    def act(self, env, me, other):
        if self.k % DECISION_EVERY == 0:
            self.action = _policy_action(self.pi, seat_obs(env, me, other), self.stochastic)
        self.k += 1
        return self.action


class MEOLPilot:
    """Option policy over frozen tactics. Stochastic = sample the tactic from the learned option
    distribution; the tactics themselves act deterministically, as during option training."""

    def __init__(self, stochastic):
        self.agent = OptionAgent(OBS_DIM)
        sd = torch.load(_rl(PILOTS["meol"]["path"]), map_location="cpu", weights_only=False)
        self.agent.pi.load_state_dict(sd["agent"]["pi"] if "agent" in sd else sd)
        self.agent.pi.eval()
        paths = json.load(open(_rl("models/meol_options/tactics.json")))
        self.tactics = [_load_actor(paths[name]) for name in OPTIONS]
        self.stochastic = stochastic

    def reset(self, seed):
        self.k, self.prev, self.opt, self.action, self.switches, self.timeline = 0, NONE, None, None, 0, []

    def act(self, env, me, other):
        if self.k % DECISION_EVERY == 0:
            obs = seat_obs(env, me, other)
            if self.k % (DECISION_EVERY * OPTION_EVERY) == 0:
                self.opt = self.agent.act(obs, self.prev, deterministic=not self.stochastic)
                self.switches += int(self.prev != NONE and self.opt != self.prev)
                self.prev = self.opt
            self.action = _policy_action(self.tactics[self.opt], obs, stochastic=False)
        self.timeline.append(self.opt)
        self.k += 1
        return self.action


class BFMPilot:
    def __init__(self):
        self.bfm = BFMAdversary(style="angles")

    def reset(self, seed):
        self.bfm._rng = np.random.default_rng(seed)
        self.bfm.reset()

    def act(self, env, me, other):
        return self.bfm.act(me, other)          # BFM throttle is already 0..1


def make_pilot(key, stochastic):
    if key == "bfm":
        return BFMPilot()
    if key == "meol":
        return MEOLPilot(stochastic)
    return TacticPilot(key, stochastic)


class OpponentSeat:
    """Puts a pilot in env.adversary: env.step calls act(ego, enm) every sim step with the
    opponent as ego, and writes the returned controls (throttle 0..1) to the opponent's FDM."""

    def __init__(self, pilot, env, name):
        self.pilot, self.env, self.current = pilot, env, name

    def reset(self):
        pass                                    # run_iteration resets the pilot with its seed

    def act(self, ego_state, enm_state):
        return self.pilot.act(self.env, ego_state, enm_state)


_CACHE = {}


def _setup(own, adv, stochastic, max_steps):
    key = (own, adv, stochastic, max_steps)
    if key not in _CACHE:
        torch.set_num_threads(1)
        env = AngleTacticEnv(max_steps=max_steps, log_trajectory=True, action_repeat=1, adversary_type="bfm")
        own_p, adv_p = make_pilot(own, stochastic), make_pilot(adv, stochastic)
        env.adversary = OpponentSeat(adv_p, env, PILOTS[adv]["label"])
        _CACHE.clear()
        _CACHE[key] = (env, own_p, adv_p)
    return _CACHE[key]


def run_iteration(cfg, i):
    """One engagement -> files in cfg['out_dir'] and a stats dict (picklable, for the pool)."""
    seed = cfg["seed"] + i
    env, own, adv = _setup(cfg["own"], cfg["adversary"], cfg["stochastic"], cfg["max_steps"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    own.reset(seed)
    adv.reset(seed + 100_000)
    ic = (cfg["range_m"], cfg["aa_deg"], cfg["ata_deg"])
    env.reset(seed=seed, options={"init_condition": ic})
    t0 = time.time()
    while True:
        me, other = _read_state(env.fdm_self), _read_state(env.fdm_adv)
        a = own.act(env, me, other).copy()
        a[3] = 2.0 * a[3] - 1.0                 # env maps the own seat's throttle from -1..1
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    outcome, traj = info["outcome"], env.trajectory
    secs = env.step_count / 50.0

    own_l, adv_l = PILOTS[cfg["own"]]["label"], PILOTS[cfg["adversary"]]["label"]
    tag = f"iter_{i + 1:03d}_{outcome}"
    title = (f"#{i + 1}: {own_l} vs {adv_l} | R={ic[0]:.0f} m AA={ic[1]:.0f} ATA={ic[2]:.0f} -> "
             f"{outcome.upper()} at {secs:.1f} s")
    out = cfg["out_dir"]
    plot_3d_trajectory(traj, os.path.join(out, f"{tag}_3d.png"), title=title)
    plot_trajectory(traj, os.path.join(out, f"{tag}_panels.png"), title=title)
    write_acmi(traj, os.path.join(out, f"{tag}.acmi"), title=title, opponent=adv_l, agent=own_l,
               outcome=outcome, weapons=cfg["weapons"])

    meol = {}
    for side, pilot in (("own", own), ("adversary", adv)):
        if isinstance(pilot, MEOLPilot):
            tl = np.array(pilot.timeline[:len(traj) - 1])
            meol[side] = {"share": [round(float(np.mean(tl == k)), 3) for k in range(len(OPTIONS))],
                          "switches": pilot.switches, "timeline": tl[::5].tolist()}
    if meol:
        _plot_options(traj, meol, os.path.join(out, f"{tag}_options.png"), title)

    wez = wez_states(traj)
    rng = np.array([r["Range"] for r in traj])
    aa, ata = np.array([r["AA"] for r in traj]), np.array([r["ATA"] for r in traj])
    e_ratio = np.array([r["E_self"] / max(r["E_adv"], 1e-6) for r in traj])
    return {
        "n": i + 1, "seed": seed, "outcome": outcome, "seconds": round(secs, 2), "files": tag,
        "min_range_m": round(float(rng.min()), 1), "final_range_m": round(float(rng[-1]), 1),
        "own_gun_s": round(sum(w[0] for w in wez) / 50.0, 2), "adv_gun_s": round(sum(w[1] for w in wez) / 50.0, 2),
        "offensive": round(float(np.mean((aa <= 60) & (ata <= 60))), 3),
        "defensive": round(float(np.mean((aa >= 120) & (ata >= 120))), 3),
        "energy_ratio": round(float(np.mean(e_ratio)), 3),
        "own_min_alt_m": round(float(min(r["self_pos"][2] for r in traj)), 0),
        "adv_min_alt_m": round(float(min(r["adv_pos"][2] for r in traj)), 0),
        "meol": meol, "wall_s": round(time.time() - t0, 2),
        "range_series": rng[::25].round(0).tolist(),
        "overview": [{"self_pos": list(map(float, r["self_pos"])), "adv_pos": list(map(float, r["adv_pos"]))}
                     for r in traj[::5]],
    }


def _plot_options(traj, meol, path, title):
    t = np.array([r["t"] for r in traj[1:]])[::5]
    fig, axes = plt.subplots(len(meol) + 1, 1, figsize=(10, 2.2 * (len(meol) + 1)), sharex=True)
    for ax, (side, m) in zip(axes, meol.items()):
        tl = np.array(m["timeline"])
        n = min(len(t), len(tl))
        ax.step(t[:n], tl[:n], where="post", color="tab:blue")
        ax.set_yticks(range(len(OPTIONS))); ax.set_yticklabels(OPTIONS)
        ax.set_title(f"MEOL in the {side} seat: tactic in use ({m['switches']} switches)", fontsize=9, loc="left")
    axes[-1].plot([r["t"] for r in traj], [r["Range"] for r in traj], color="tab:gray")
    axes[-1].set_ylabel("range (m)"); axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_demo(cfg, progress=None):
    """cfg: own, adversary, range_m, aa_deg, ata_deg, iterations, seed, stochastic, weapons,
    max_steps, out_root. progress(done, total, result) is called after every fight."""
    cfg = dict(cfg)
    cfg.setdefault("stochastic", True)
    cfg.setdefault("weapons", True)
    cfg.setdefault("max_steps", 6000)
    cfg.setdefault("seed", int(time.time()) % 100_000)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = (f"{stamp}_{cfg['own']}-vs-{cfg['adversary']}_R{cfg['range_m']:.0f}_AA{cfg['aa_deg']:.0f}"
            f"_ATA{cfg['ata_deg']:.0f}_n{cfg['iterations']}")
    cfg["out_dir"] = os.path.join(cfg.get("out_root", os.path.join(RL, "demo_runs")), name)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    cfg["run_name"] = name
    json.dump({k: v for k, v in cfg.items()}, open(os.path.join(cfg["out_dir"], "config.json"), "w"), indent=2)

    n = cfg["iterations"]
    results, t0 = [], time.time()
    workers = cfg.get("workers") or max(1, min(n, (os.cpu_count() or 4) // 3, 6))
    if workers == 1 or n < 3:
        for i in range(n):
            results.append(run_iteration(cfg, i))
            if progress:
                progress(len(results), n, results[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_iteration, cfg, i) for i in range(n)]
            for fut in as_completed(futures):
                results.append(fut.result())
                if progress:
                    progress(len(results), n, results[-1])
    results.sort(key=lambda r: r["n"])
    summary = summarize(cfg, results, time.time() - t0)
    write_outputs(cfg, results, summary)
    return summary


def summarize(cfg, results, wall):
    n = len(results)
    c = Counter(r["outcome"] for r in results)
    wins = [r["seconds"] for r in results if r["outcome"] == "win"]
    losses = [r["seconds"] for r in results if r["outcome"] == "loss"]
    lo, hi = wilson(c["win"], n)
    mean = lambda xs: round(float(np.mean(xs)), 2) if xs else None
    s = {
        "run_name": cfg["run_name"], "own": cfg["own"], "adversary": cfg["adversary"],
        "own_label": PILOTS[cfg["own"]]["label"], "adv_label": PILOTS[cfg["adversary"]]["label"],
        "start": {"range_m": cfg["range_m"], "aa_deg": cfg["aa_deg"], "ata_deg": cfg["ata_deg"],
                  "alt_m": START_ALT_M, "speed_mps": START_SPEED_MPS},
        "iterations": n, "seed": cfg["seed"], "stochastic": cfg["stochastic"], "wall_s": round(wall, 1),
        "outcomes": dict(c), "win_rate": round(c["win"] / n, 3), "win_ci95": [round(lo, 3), round(hi, 3)],
        "loss_rate": round((c["loss"] + c["crash"] + c["disengaged"]) / n, 3),
        "draw_rate": round((c["timeout"] + c["adversary_crash"]) / n, 3),
        "time_to_kill_s": {"mean": mean(wins), "median": round(float(np.median(wins)), 2) if wins else None,
                           "min": min(wins) if wins else None},
        "time_to_loss_s": mean(losses), "fight_s": mean([r["seconds"] for r in results]),
        "own_gun_s": mean([r["own_gun_s"] for r in results]), "adv_gun_s": mean([r["adv_gun_s"] for r in results]),
        "offensive": mean([r["offensive"] for r in results]), "defensive": mean([r["defensive"] for r in results]),
        "energy_ratio": mean([r["energy_ratio"] for r in results]),
        "min_range_m": mean([r["min_range_m"] for r in results]),
        "own_min_alt_m": min(r["own_min_alt_m"] for r in results),
        "adv_min_alt_m": min(r["adv_min_alt_m"] for r in results),
    }
    for side in ("own", "adversary"):
        ms = [r["meol"][side] for r in results if side in r["meol"]]
        if ms:
            s[f"meol_{side}"] = {"share": {OPTIONS[k]: round(float(np.mean([m["share"][k] for m in ms])), 3)
                                           for k in range(len(OPTIONS))},
                                 "switches": mean([m["switches"] for m in ms])}
    return s


def write_outputs(cfg, results, s):
    out = cfg["out_dir"]
    cols = ["n", "seed", "outcome", "seconds", "min_range_m", "final_range_m", "own_gun_s", "adv_gun_s",
            "offensive", "defensive", "energy_ratio", "own_min_alt_m", "adv_min_alt_m", "files"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    slim = [{k: v for k, v in r.items() if k not in ("overview", "range_series")} for r in results]
    json.dump({"summary": s, "iterations": slim}, open(os.path.join(out, "summary.json"), "w"), indent=2)
    plot_overview([(f"#{r['n']} ({r['seconds']:.0f} s)", r["overview"], r["outcome"]) for r in results],
                  os.path.join(out, "overview_3d.png"),
                  title=f"{s['own_label']} (blue) vs {s['adv_label']} (red): " +
                        ", ".join(f"{k} {v}" for k, v in Counter(r["outcome"] for r in results).most_common()))
    _plot_summary(results, s, os.path.join(out, "summary.png"))
    _write_report(cfg, results, s, os.path.join(out, "report.md"))


def _plot_summary(results, s, path):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    order = [k for k in OUTCOME_COLORS if k in s["outcomes"]]
    ax = axs[0, 0]
    ax.bar(order, [s["outcomes"][k] for k in order], color=[OUTCOME_COLORS[k] for k in order])
    for x, k in enumerate(order):
        ax.text(x, s["outcomes"][k], f"{s['outcomes'][k] / s['iterations']:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Outcomes for {s['own_label']} (n = {s['iterations']})", loc="left", fontsize=10)
    ax = axs[0, 1]
    ax.bar([r["n"] for r in results], [r["seconds"] for r in results],
           color=[OUTCOME_COLORS.get(r["outcome"], "k") for r in results])
    ax.set_xlabel("iteration"); ax.set_ylabel("fight length (s)")
    ax.set_title("Fight length per iteration (colour = outcome)", loc="left", fontsize=10)
    ax = axs[1, 0]
    for r in results:
        ax.plot(np.arange(len(r["range_series"])) * 0.5, r["range_series"], color=OUTCOME_COLORS.get(r["outcome"], "k"),
                lw=1, alpha=0.6)
    ax.axhspan(150, 1000, color="#2a78d6", alpha=0.08); ax.text(1, 1050, "gun range 150-1000 m", fontsize=8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("range (m)")
    ax.set_title("Range over time, every iteration", loc="left", fontsize=10)
    ax = axs[1, 1]
    nn = [r["n"] for r in results]
    off = np.array([r["offensive"] for r in results]); dfn = np.array([r["defensive"] for r in results])
    ax.bar(nn, off, color="#2a78d6", label="offensive (on its six)")
    ax.bar(nn, 1 - off - dfn, bottom=off, color="#cfcfcf", label="neutral")
    ax.bar(nn, dfn, bottom=1 - dfn, color="#eb6834", label="defensive (it is on ours)")
    ax.set_ylim(0, 1); ax.set_xlabel("iteration"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Share of each fight spent offensive / neutral / defensive ({s['own_label']})", loc="left", fontsize=10)
    fig.suptitle(f"{s['own_label']} vs {s['adv_label']} from R={s['start']['range_m']:.0f} m, "
                 f"AA={s['start']['aa_deg']:.0f}, ATA={s['start']['ata_deg']:.0f}", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def _variation(s):
    parts = []
    if s["stochastic"] and not s["own"] == s["adversary"] == "bfm":
        parts.append("stochastic actions (each trained policy samples from its learned distribution)")
    if "bfm" in (s["own"], s["adversary"]):
        parts.append("a new random seed for the rule-based expert")
    return " and ".join(parts) or "none: deterministic policies give the same fight every iteration"


def _write_report(cfg, results, s, path):
    pct = lambda x: f"{100 * x:.0f}%"
    o = s["outcomes"]
    ttk = s["time_to_kill_s"]
    lines = [
        f"# {s['own_label']} vs {s['adv_label']}", "",
        f"{s['iterations']} engagements from the same start, {time.strftime('%Y-%m-%d %H:%M')}. "
        f"Run folder `{s['run_name']}`.", "",
        "## Setup", "",
        "| | |", "|---|---|",
        f"| Own aircraft (blue) | {s['own_label']} |",
        f"| Adversary (red) | {s['adv_label']} |",
        f"| Start | Range {s['start']['range_m']:.0f} m, AA {s['start']['aa_deg']:.0f}°, ATA {s['start']['ata_deg']:.0f}°; "
        f"both at {START_ALT_M:.0f} m and {START_SPEED_MPS:.0f} m/s, wings level |",
        f"| Variation between iterations | {_variation(s)} |",
        f"| Weights | best checkpoints (`policy_best.pt`; MEOL: `options_best.pt` over the tactics it was trained with) |",
        f"| Fight limit | {cfg['max_steps'] / 50:.0f} s; a kill is a 0.5 s hold inside the gun cone (150–1000 m, ±15°) |",
        f"| Seeds | {s['seed']} … {s['seed'] + s['iterations'] - 1} |", "",
        "## Result", "",
        f"**{s['own_label']} won {o.get('win', 0)} of {s['iterations']} ({pct(s['win_rate'])}, 95% CI "
        f"{pct(s['win_ci95'][0])}–{pct(s['win_ci95'][1])}).**", "",
        "| Outcome | Count | Share |", "|---|---|---|",
    ]
    names = {"win": "Win (gun kill)", "loss": "Loss (shot down)", "timeout": "Draw (time out)", "crash": "Own crash",
             "disengaged": "Disengaged (range > 12 km)", "adversary_crash": "Adversary crashed"}
    for k in OUTCOME_COLORS:
        if k in o:
            lines.append(f"| {names[k]} | {o[k]} | {pct(o[k] / s['iterations'])} |")
    lines += ["", "## How it fought", "",
              "| Measure | Value |", "|---|---|",
              f"| Time to kill (wins) | mean {ttk['mean']} s, median {ttk['median']} s, fastest {ttk['min']} s |" if ttk["mean"] else "| Time to kill | no wins |",
              f"| Time to be shot down (losses) | mean {s['time_to_loss_s']} s |" if s["time_to_loss_s"] else "| Time to be shot down | no losses |",
              f"| Mean fight length | {s['fight_s']} s |",
              f"| Gun solution held per fight (own / adversary) | {s['own_gun_s']} s / {s['adv_gun_s']} s |",
              f"| Time offensive (on its six: AA ≤ 60°, ATA ≤ 60°) | {pct(s['offensive'])} |",
              f"| Time defensive (it is on ours: AA ≥ 120°, ATA ≥ 120°) | {pct(s['defensive'])} |",
              f"| Mean energy ratio (own / adversary, Eq. 31) | {s['energy_ratio']} |",
              f"| Mean closest approach | {s['min_range_m']} m |",
              f"| Lowest altitude reached (own / adversary) | {s['own_min_alt_m']:.0f} m / {s['adv_min_alt_m']:.0f} m |"]
    for side in ("own", "adversary"):
        m = s.get(f"meol_{side}")
        if m:
            lines.append(f"| MEOL ({side} seat): tactic share angle / snapshot / energy, switches per fight | "
                         f"{pct(m['share']['angle'])} / {pct(m['share']['snapshot'])} / {pct(m['share']['energy'])}, "
                         f"{m['switches']} |")
    lines += ["", "![summary](summary.png)", "", "![every fight](overview_3d.png)", "",
              "## Every iteration", "",
              "| # | Outcome | Time (s) | Closest (m) | Gun time own / adv (s) | Offensive | Files |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        extra = f", [tactics]({r['files']}_options.png)" if r["meol"] else ""
        lines.append(f"| {r['n']} | {r['outcome']} | {r['seconds']} | {r['min_range_m']:.0f} | {r['own_gun_s']} / "
                     f"{r['adv_gun_s']} | {pct(r['offensive'])} | [3D]({r['files']}_3d.png), "
                     f"[panels]({r['files']}_panels.png), [Tacview]({r['files']}.acmi){extra} |")
    lines += ["", "## Notes", "",
              "- Each fight is from the own aircraft's side: *win* means the blue jet held a gun solution for 0.5 s first.",
              "- The policies were trained against the rule-based expert only; policy-vs-policy fights are outside their "
              "training (the paper's confrontation experiment).",
              "- The simulator has no collision model: the jets can pass within metres of each other (see 'Closest').",
              "- The snapshot tactic exploits the expert's ground-avoidance rule in about a third of its wins against it "
              "(see `models/angle_tactic_S/README.md`); watch the lowest-altitude row.",
              "- Tacview files carry the gun cones (install once: `python export.py --install-mesh`), lock lines and an "
              "outcome bookmark."]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
