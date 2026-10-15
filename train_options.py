"""
Option utilization, the high level of MEOL (Algorithm 3, Eqs. 21-25).

The three trained tactic policies -- angle, snapshot, energy -- are loaded FROZEN as
options. A maximum-entropy option-selection policy pi(omega_t | s_t, omega_{t-1}) picks
which tactic flies the aircraft. Conditioning on the previous option folds the
termination function into the policy (the paper's Eq. 4): "keep the current tactic"
and "switch" are both just choices of omega_t. It is trained with discrete soft
actor-critic on the sparse combat outcome only, R_tactic = R_goal (Eq. 42).

  soft option value Q(s, omega), twin, with target nets              Eq. (21)-(22)
  U(s', omega) = sum_w' pi(w'|s', omega) [Qbar(s', w') - alpha log pi]  Eq. (23)
  policy loss  = E_s sum_w pi(w|s, w_prev) [alpha log pi - min Q(s, w)] Eq. (24)
  alpha        tuned so that the option entropy tracks H_bar            Eq. (25)

Choices the paper does not fix (flagged):
  * option_interval = 10 agent decisions (1 s): one option decision per second, an
    SMDP step. gamma = 0.99 is applied per option decision: per 0.1 s agent step it
    would discount a win 120 s away to ~1e-5 and the sparse reward could not be seen.
  * target entropy 0.5 nats (paper Table 2: 1.0). Over three options 1.0 nats is 91%
    of the maximum, i.e. a near-uniform choice; at one decision per second that
    switches tactic about every other second, so the tactics never get to act.
  * outcome rewards in units of 1000: win +1.5, loss / crash / disengage -1.0,
    timeout and opponent crash 0 -- the same R_goal the tactics were trained on.
  * starts are uniform over Table 1 (Algorithm 3 has no curriculum); the opponent
    is the full BFM expert (bfm_angles); intra-option actions are deterministic.

  python train_options.py --run-dir runs/O_meol \
      --angle models/angle_tactic_W/policy_final.pt \
      --snapshot models/angle_tactic_S/policy_final.pt \
      --energy models/angle_tactic_E/policy_final.pt
  python train_options.py --fly runs/O_meol/checkpoints/best_options.pt --out visualizations/O_meol ...
"""

import argparse
import copy
import json
import os
import time
from collections import Counter, deque

import numpy as np
import torch
import torch.nn.functional as F

from adversary import LADDER, TOP as LADDER_TOP
from env import AngleTacticEnv, OBS_DIMS
from export import write_acmi
from fly_policy import NAMED_STARTS, load_frozen_policy
from sac import mlp, resolve_device
from train_angle_wgan import _append_row, _write_rows
from visualize import plot_3d_trajectory

OPTIONS = ("angle", "snapshot", "energy")
NONE = len(OPTIONS)            # "no previous option" (first decision of an episode)
OUTCOME_REWARD = {"win": 1.5, "loss": -1.0, "crash": -1.0, "disengaged": -1.0}   # anything else: 0
CKPT_FORMAT = "meol-options-v1"

EP_HEADER = ["episode", "epoch", "env_steps", "outcome", "win_rate_100", "switches",
             "share_angle", "share_snapshot", "share_energy", "init_range", "init_aa", "init_ata"]
EPOCH_HEADER = ["epoch", "env_steps", "updates", "q_loss", "pi_loss", "alpha", "entropy", "sim_steps_per_s",
                "win", "loss", "timeout", "crash", "other", "win_rate_100",
                "share_angle", "share_snapshot", "share_energy", "switches_per_ep"]


class OptionAgent:
    """Discrete SAC over options; the policy also sees the previous option (one-hot, NONE at start)."""

    def __init__(self, obs_dim, n_opt=len(OPTIONS), hidden=256, gamma=0.99, tau=5e-3, lr_q=3e-4,
                 lr_pi=3e-5, lr_alpha=3e-5, init_alpha=0.1, target_entropy=0.5, device="cpu"):
        self.device = resolve_device(device)
        self.n, self.gamma, self.tau, self.target_entropy = n_opt, gamma, tau, target_entropy
        self.pi = mlp([obs_dim + n_opt + 1, hidden, hidden, n_opt]).to(self.device)
        self.q1 = mlp([obs_dim, hidden, hidden, n_opt]).to(self.device)
        self.q2 = mlp([obs_dim, hidden, hidden, n_opt]).to(self.device)
        self.q1_targ, self.q2_targ = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for p in list(self.q1_targ.parameters()) + list(self.q2_targ.parameters()):
            p.requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_q)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=lr_pi)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _log_pi(self, obs, prev):
        x = torch.cat([obs, F.one_hot(prev, self.n + 1).float()], dim=-1)
        return F.log_softmax(self.pi(x), dim=-1)

    @torch.no_grad()
    def act(self, obs, prev, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        p = torch.as_tensor([prev], device=self.device)
        logp = self._log_pi(o, p)[0]
        return int(logp.argmax()) if deterministic else int(torch.multinomial(logp.exp(), 1))

    def update(self, b):
        obs, prev, opt, rew, nobs, done = (b[k] for k in ("obs", "prev", "opt", "rew", "next_obs", "done"))
        with torch.no_grad():                                                    # Eq. (22)-(23)
            logp_n = self._log_pi(nobs, opt)                # at s' the previous option is the one taken at s
            q_n = torch.min(self.q1_targ(nobs), self.q2_targ(nobs))
            u_n = (logp_n.exp() * (q_n - self.alpha * logp_n)).sum(-1)
            y = rew + self.gamma * (1.0 - done) * u_n
        q1 = self.q1(obs).gather(1, opt[:, None]).squeeze(1)
        q2 = self.q2(obs).gather(1, opt[:, None]).squeeze(1)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)                           # Eq. (21)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        logp = self._log_pi(obs, prev)                                           # Eq. (24)
        p = logp.exp()
        with torch.no_grad():
            q = torch.min(self.q1(obs), self.q2(obs))
        pi_loss = (p * (self.alpha.detach() * logp - q)).sum(-1).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()

        entropy = -(p * logp).sum(-1).mean().detach()                            # Eq. (25)
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        with torch.no_grad():
            for net, targ in ((self.q1, self.q1_targ), (self.q2, self.q2_targ)):
                for p_, pt in zip(net.parameters(), targ.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p_)
        return {"q_loss": q_loss.item(), "pi_loss": pi_loss.item(), "alpha": self.alpha.item(),
                "entropy": entropy.item()}

    def state_dict(self):
        mods = ("pi", "q1", "q2", "q1_targ", "q2_targ", "q_opt", "pi_opt", "alpha_opt")
        sd = {m: getattr(self, m).state_dict() for m in mods}
        sd["log_alpha"] = self.log_alpha.detach().cpu()
        return sd

    def load_state_dict(self, sd):
        for m in ("pi", "q1", "q2", "q1_targ", "q2_targ", "q_opt", "pi_opt", "alpha_opt"):
            getattr(self, m).load_state_dict(sd[m])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))


class OptionReplay:
    def __init__(self, obs_dim, size=int(5e5)):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.prev = np.zeros(size, np.int64)
        self.opt = np.zeros(size, np.int64)
        self.rew = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, o, prev, opt, r, no, d):
        i = self.ptr
        self.obs[i], self.prev[i], self.opt[i], self.rew[i], self.next_obs[i], self.done[i] = o, prev, opt, r, no, d
        self.ptr, self.size = (i + 1) % self.max_size, min(self.size + 1, self.max_size)

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, n)
        t = lambda a: torch.as_tensor(a[idx], device=device)
        return {"obs": t(self.obs), "prev": t(self.prev), "opt": t(self.opt), "rew": t(self.rew),
                "next_obs": t(self.next_obs), "done": t(self.done)}

    def state_dict(self):
        n = self.size
        return {k: getattr(self, k)[:n].copy() for k in ("obs", "next_obs", "prev", "opt", "rew", "done")} | \
               {"ptr": self.ptr, "size": n}

    def load_state_dict(self, sd):
        n = sd["size"]
        for k in ("obs", "next_obs", "prev", "opt", "rew", "done"):
            getattr(self, k)[:n] = sd[k]
        self.ptr, self.size = sd["ptr"] % self.max_size, n


def load_tactics(paths, obs_dim):
    """Frozen intra-option policies -> list of obs -> deterministic action functions."""
    fns = []
    for name in OPTIONS:
        pi = load_frozen_policy(paths[name], obs_dim)
        fns.append(lambda obs, pi=pi: pi.sample(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[2]
                   .squeeze(0).numpy())
    return fns


def run_episode(env, agent, tactics, option_interval, deterministic, init_condition=None, on_decision=None):
    """One engagement. Returns (outcome, sim steps, option sim-step counts, switches, option per decision)."""
    obs, _ = env.reset(options={"init_condition": init_condition} if init_condition else None)
    prev, counts, switches, timeline, steps = NONE, np.zeros(len(OPTIONS)), 0, [], 0
    while True:
        opt = agent.act(obs, prev, deterministic)
        switches += int(prev != NONE and opt != prev)
        s0, term, trunc = obs, False, False
        for _ in range(option_interval):
            obs, _, term, trunc, info = env.step(tactics[opt](obs))
            counts[opt] += info["sim_steps"]; steps += info["sim_steps"]
            timeline += [opt] * info["sim_steps"]
            if term or trunc:
                break
        r = OUTCOME_REWARD.get(info["outcome"], 0.0) if term else 0.0
        if on_decision:
            on_decision(s0, prev, opt, r, obs, float(term))   # truncation (timeout) still bootstraps
        prev = opt
        if term or trunc:
            return info["outcome"], steps, counts, switches, timeline


def train(tactic_paths, run_dir, total_env_steps=int(1.5e7), option_interval=10, episodes_per_epoch=20,
          batch=256, updates_per_decision=1, target_entropy=0.5, buffer_size=int(5e5), seed=0,
          device="cpu", resume=None, obs_mode="extended"):
    config = {k: v for k, v in locals().items() if k != "resume"}
    log_dir, ckpt_dir = os.path.join(run_dir, "logs"), os.path.join(run_dir, "checkpoints")
    for d in (log_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)
    ep_log, epoch_log = os.path.join(log_dir, "option_episodes.csv"), os.path.join(log_dir, "option_epochs.csv")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_latest.pt")
    np.random.seed(seed); torch.manual_seed(seed)

    env = AngleTacticEnv(max_steps=6000, log_trajectory=False, obs_mode=obs_mode,
                         adversary_type="ladder", opponent_level=LADDER_TOP)
    obs_dim = OBS_DIMS[obs_mode]
    tactics = load_tactics(tactic_paths, obs_dim)
    agent = OptionAgent(obs_dim, target_entropy=target_entropy, device=device)
    buf = OptionReplay(obs_dim, buffer_size)
    c = {"epoch": 0, "episode": 0, "env_steps": 0, "updates": 0, "win_hist": [], "best": -1.0}
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        assert ck.get("format") == CKPT_FORMAT, f"{resume} is not a {CKPT_FORMAT} checkpoint"
        agent.load_state_dict(ck["agent"]); buf.load_state_dict(ck["buffer"]); c = ck["counters"]
        np.random.set_state(ck["rng"]["numpy"]); torch.set_rng_state(ck["rng"]["torch"])
        print(f"[resume] {resume}: epoch {c['epoch']}, env_steps {c['env_steps']}, replay {buf.size}", flush=True)
    else:
        _write_rows(ep_log, EP_HEADER, []); _write_rows(epoch_log, EPOCH_HEADER, [])
    print("[options] " + ", ".join(f"{n}={tactic_paths[n]}" for n in OPTIONS), flush=True)
    win_hist = deque(c["win_hist"], maxlen=100)
    stats = {"q": [], "pi": [], "ent": []}

    def on_decision(*tr):
        buf.add(*tr)
        if buf.size >= batch:
            for _ in range(updates_per_decision):
                s = agent.update(buf.sample(batch, agent.device))
                stats["q"].append(s["q_loss"]); stats["pi"].append(s["pi_loss"]); stats["ent"].append(s["entropy"])
                c["updates"] += 1

    while c["env_steps"] < total_env_steps:
        t0, steps0 = time.time(), c["env_steps"]
        outcomes, share, sw = Counter(), np.zeros(len(OPTIONS)), 0
        stats = {"q": [], "pi": [], "ent": []}
        for _ in range(episodes_per_epoch):
            outcome, steps, counts, switches, _ = run_episode(env, agent, tactics, option_interval, False,
                                                              on_decision=on_decision)
            c["episode"] += 1; c["env_steps"] += steps
            outcomes[outcome] += 1; share += counts; sw += switches
            win_hist.append(int(outcome == "win"))
            ic = env.last_init_condition
            _append_row(ep_log, [c["episode"], c["epoch"], c["env_steps"], outcome, float(np.mean(win_hist)), switches,
                                 *(counts / max(counts.sum(), 1)).round(3), *ic])
        share /= max(share.sum(), 1)
        rate = (c["env_steps"] - steps0) / max(time.time() - t0, 1e-6)
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        other = sum(v for k, v in outcomes.items() if k not in ("win", "loss", "timeout", "crash"))
        wr = float(np.mean(win_hist))
        _append_row(epoch_log, [c["epoch"], c["env_steps"], c["updates"], mean(stats["q"]), mean(stats["pi"]),
                                agent.alpha.item(), mean(stats["ent"]), round(rate, 1), outcomes["win"], outcomes["loss"],
                                outcomes["timeout"], outcomes["crash"], other, wr, *share.round(3),
                                sw / episodes_per_epoch])
        print(f"[epoch {c['epoch']:4d}] env_steps {c['env_steps']:8d} | win {outcomes['win']} loss {outcomes['loss']} "
              f"timeout {outcomes['timeout']} crash {outcomes['crash']} other {other} | win_rate(100) {wr:.3f} | "
              f"options A/S/E {share[0]:.2f}/{share[1]:.2f}/{share[2]:.2f} switches/ep {sw / episodes_per_epoch:.1f} | "
              f"alpha {agent.alpha.item():.4f} entropy {mean(stats['ent']):.2f} | {rate:.0f} sim steps/s", flush=True)
        if len(win_hist) == win_hist.maxlen and wr > c["best"]:
            c["best"] = wr
            torch.save(agent.pi.state_dict(), os.path.join(ckpt_dir, "best_options.pt"))
            with open(os.path.join(ckpt_dir, "best_options.json"), "w") as f:
                json.dump({"epoch": c["epoch"], "env_steps": c["env_steps"], "win_rate_100": wr}, f, indent=2)
            print(f"[best] epoch {c['epoch']}: rolling win rate {wr:.2f} -> checkpoints/best_options.pt", flush=True)
        c["epoch"] += 1; c["win_hist"] = list(win_hist)
        tmp = ckpt_path + ".tmp"
        torch.save({"format": CKPT_FORMAT, "agent": agent.state_dict(), "buffer": buf.state_dict(), "counters": c,
                    "rng": {"numpy": np.random.get_state(), "torch": torch.get_rng_state()}, "config": config}, tmp)
        os.replace(tmp, ckpt_path)
        torch.save(agent.pi.state_dict(), os.path.join(ckpt_dir, "options_latest.pt"))
    torch.save(agent.pi.state_dict(), os.path.join(ckpt_dir, "options_final.pt"))
    print("Training complete. Final win_rate(100):", float(np.mean(win_hist)) if win_hist else float("nan"), flush=True)


def fly(policy_path, tactic_paths, out, random_starts=30, seed=2026, option_interval=10, obs_mode="extended",
        weapons=True):
    """Frozen hierarchical agent (argmax option, deterministic tactics) over fly_policy's start set;
    saves a 3D plot, an option timeline, a Tacview file per fight, and summary.csv."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    obs_dim = OBS_DIMS[obs_mode]
    agent = OptionAgent(obs_dim)
    sd = torch.load(policy_path, map_location="cpu", weights_only=False)
    agent.pi.load_state_dict(sd["agent"]["pi"] if "agent" in sd else sd)
    agent.pi.eval()
    tactics = load_tactics(tactic_paths, obs_dim)
    env = AngleTacticEnv(max_steps=6000, log_trajectory=True, obs_mode=obs_mode, adversary_type="ladder",
                         opponent_level=LADDER_TOP)
    rng = np.random.default_rng(seed)
    starts = list(NAMED_STARTS) + [(f"random_{i + 1:02d}", (float(rng.uniform(1900, 5100)), float(rng.uniform(-180, 180)),
                                                            float(rng.uniform(-180, 180)))) for i in range(random_starts)]
    rows = []
    for i, (name, ic) in enumerate(starts):
        env.adversary._rng = np.random.default_rng(seed + i)
        env.reset(seed=seed + i)
        outcome, steps, counts, switches, timeline = run_episode(env, agent, tactics, option_interval, True, ic)
        tag, secs = f"{i + 1:02d}_{name}_{outcome}", steps / 50.0
        title = f"{name}: R={ic[0]:.0f} AA={ic[1]:.0f} ATA={ic[2]:.0f} -> {outcome.upper()} at {secs:.1f} s (MEOL options)"
        plot_3d_trajectory(env.trajectory, os.path.join(out, f"{tag}_3d.png"), title=title)
        write_acmi(env.trajectory, os.path.join(out, f"{tag}.acmi"), title=title, opponent=LADDER[LADDER_TOP],
                   outcome=outcome, weapons=weapons)
        t = np.array([r["t"] for r in env.trajectory[1:]])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        a1.step(t[:len(timeline)], timeline[:len(t)], where="post", color="tab:blue")
        a1.set_yticks(range(len(OPTIONS))); a1.set_yticklabels(OPTIONS); a1.set_title(title, fontsize=9)
        a2.plot(t, [r["Range"] for r in env.trajectory[1:]], color="tab:gray", label="range (m)")
        a2.set_xlabel("time (s)"); a2.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_options.png"), dpi=110); plt.close(fig)
        share = counts / max(counts.sum(), 1)
        rows.append({"n": i + 1, "start": name, "range_m": round(ic[0]), "aa_deg": round(ic[1]), "ata_deg": round(ic[2]),
                     "outcome": outcome, "seconds": round(secs, 1), "switches": switches,
                     **{f"share_{n}": round(float(s), 2) for n, s in zip(OPTIONS, share)}})
        print(f"{i + 1:2d}/{len(starts)} {name:20s} -> {outcome:10s} {secs:6.1f} s  switches {switches:3d}  "
              f"A/S/E {share[0]:.2f}/{share[1]:.2f}/{share[2]:.2f}", flush=True)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("outcomes:", dict(Counter(r["outcome"] for r in rows)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MEOL option utilization (Algorithm 3) over frozen tactic policies")
    p.add_argument("--angle", default="models/angle_tactic_W/policy_final.pt")
    p.add_argument("--snapshot", default="models/angle_tactic_S/policy_final.pt")
    p.add_argument("--energy", default="models/angle_tactic_E/policy_final.pt")
    p.add_argument("--run-dir", default="runs/O_meol")
    p.add_argument("--total-env-steps", type=int, default=int(1.5e7), help="50 Hz sim steps (paper Fig. 13: 1.5e7)")
    p.add_argument("--option-interval", type=int, default=10, help="agent decisions (0.1 s) per option decision")
    p.add_argument("--episodes-per-epoch", type=int, default=20)
    p.add_argument("--updates-per-decision", type=int, default=1)
    p.add_argument("--target-entropy", type=float, default=0.5, help="nats; paper Table 2: 1.0 (see module doc)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="checkpoint file for an exact resume")
    p.add_argument("--fly", default=None, help="option policy (.pt or checkpoint) to fly frozen instead of training")
    p.add_argument("--out", default="visualizations/O_meol")
    p.add_argument("--random-starts", type=int, default=30)
    p.add_argument("--no-weapons", action="store_true", help="--fly: leave the Tacview weapons layer out")
    a = p.parse_args()
    if a.torch_threads > 0:
        torch.set_num_threads(a.torch_threads)
    paths = {"angle": a.angle, "snapshot": a.snapshot, "energy": a.energy}
    if a.fly:
        fly(a.fly, paths, a.out, a.random_starts, option_interval=a.option_interval, weapons=not a.no_weapons)
    else:
        train(paths, a.run_dir, a.total_env_steps, a.option_interval, a.episodes_per_epoch,
              updates_per_decision=a.updates_per_decision, target_entropy=a.target_entropy, seed=a.seed,
              device=a.device, resume=a.resume)
