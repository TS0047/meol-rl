"""
Angle-tactic training loop implementing Algorithm 2's structure:

  for each epoch:
      for k critic steps:                                  (Alg.2 lines 5-11)
          update critic on feasible-buffer reals vs G(z) fakes
      sample n initial conditions x0_j (generator + a uniform share)   (line 12)
      for each x0_j:                                       (lines 13-17)
          run `episodes_per_point` COMPLETE episodes from x0_j. Every agent
          decision stores a transition and does one SAC update (Algorithm 1).
          An episode ends on a gun kill (0.5 s one-sided WEZ hold), a crash,
          disengagement, adversary crash or timeout.
          win_rate_j = wins / completed episodes at x0_j
      label Eq.(9), train R_hat, run WGAN-GP generator steps   (lines 18-19)

An episode cut short by the global step budget is discarded (not counted),
so win rates are only ever computed from finished episodes.

env_steps / total_env_steps count 50 Hz SIM steps (as in the paper's 1.5e7), not
agent decisions: with action_repeat=5 the agent decides -- and SAC updates --
once per 5 sim steps.

Resuming (--resume):
  * a checkpoint file (checkpoints/checkpoint_latest.pt, written every epoch)
    restores everything -- networks, optimizers, alpha, replay buffer,
    curriculum, counters, RNG -- and continues the same run;
  * a model folder (e.g. models/angle_tactic_A) warm-starts the ACTOR only: alpha
    from its last logged value, epoch/step counters and logs carried over. The
    critics and the curriculum start fresh, so for the first
    --actor-warmup-updates updates only the critics train: an actor following
    the gradients of untrained critics would be wrecked.
"""

import os
import csv
import json
import time
import argparse
from collections import Counter, deque

import numpy as np
import torch

from env import AngleTacticEnv
from sac import SACAgent, ReplayBuffer
from wgan import WGANCurriculum
from adversary import LADDER, TOP as LADDER_TOP
from export import write_acmi
from visualize import plot_3d_trajectory, plot_win_rate

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
CKPT_DIR = os.path.join(HERE, "checkpoints")
PLOT_DIR = os.path.join(HERE, "plots")
for d in (LOG_DIR, CKPT_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)


def set_run_dir(run_dir):
    """Write this run's logs/, checkpoints/ and plots/ under run_dir instead of next to
    this file, so several runs can train side by side without overwriting each other."""
    global LOG_DIR, CKPT_DIR, PLOT_DIR
    LOG_DIR, CKPT_DIR, PLOT_DIR = (os.path.join(run_dir, d) for d in ("logs", "checkpoints", "plots"))
    for d in (LOG_DIR, CKPT_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)

CKPT_FORMAT = "meol-rl-full-v1"
EP_HEADER = ["episode", "epoch", "env_steps", "return", "outcome", "win_rate_100", "x0_range", "x0_aa", "x0_ata",
             "opponent"]
EPOCH_HEADER = ["epoch", "episode", "env_steps", "d_loss", "wasserstein_est", "g_loss", "predictor_loss",
                "n_valid_points", "epoch_win_rate", "epoch_completions", "feasible_buffer_size"]
SAC_HEADER = ["epoch", "env_steps", "updates", "actor_frozen_updates", "q_loss", "pi_loss", "alpha", "entropy",
              "sim_steps_per_s", "win", "loss", "timeout", "crash", "other",
              "opp_level", "level_wins", "level_eps", "top_wins", "top_eps"]


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


_pending = {}   # path -> rows waiting for a file lock to clear


def _append_row(path, row):
    """Append one CSV row. If another program holds the file -- Excel locks a CSV it
    has open -- keep the row in memory and retry on the next append, instead of
    killing a multi-hour run (run W died this way mid-epoch)."""
    rows = _pending.get(path, []) + [row]
    try:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        if path in _pending:
            print(f"[log] {path} writable again; wrote {len(rows)} buffered rows", flush=True)
        _pending.pop(path, None)
    except PermissionError:
        if path not in _pending:
            print(f"[log] {path} is locked by another program (open in Excel?); buffering rows "
                  f"until it is closed", flush=True)
        _pending[path] = rows


def _keep_epochs_before(path, header, epoch):
    """Drop rows from epochs >= `epoch` (an interrupted epoch's partial rows)."""
    try:
        if os.path.exists(path):
            _write_rows(path, header, [r for r in _read_rows(path) if int(r["epoch"]) < epoch])
        else:
            _write_rows(path, header, [])
    except PermissionError:
        raise SystemExit(f"{path} is locked by another program (open in Excel?). Close it and resume again.")


def save_checkpoint(path, agent, buffer, curriculum, counters, config):
    tmp = path + ".tmp"
    torch.save({"format": CKPT_FORMAT, "agent": agent.state_dict(), "buffer": buffer.state_dict(),
                "curriculum": curriculum.state_dict(), "counters": dict(counters),
                "rng": {"numpy": np.random.get_state(), "torch": torch.get_rng_state()},
                "config": config}, tmp)
    os.replace(tmp, path)   # atomic: an interrupted save never leaves a half-written checkpoint


def _resume_full(path, agent, buffer, curriculum):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != CKPT_FORMAT:
        raise ValueError(f"{path} is not a {CKPT_FORMAT} checkpoint")
    agent.load_state_dict(ck["agent"])
    buffer.load_state_dict(ck["buffer"])
    curriculum.load_state_dict(ck["curriculum"])
    np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"])
    return dict(ck["counters"])


def _warm_start_from_model_dir(model_dir, agent, ep_log_path, epoch_log_path):
    """Actor + alpha + counters + logs from a models/<run>/ folder (see its README)."""
    agent.pi.load_state_dict(torch.load(os.path.join(model_dir, "policy.pt"), map_location=agent.device))
    tr = os.path.join(model_dir, "training")
    epochs = _read_rows(os.path.join(tr, "wgan_epochs.csv"))
    last_epoch = int(epochs[-1]["epoch"])
    episodes = [r for r in _read_rows(os.path.join(tr, "angle_tactic_wgan_episodes.csv"))
                if int(r["epoch"]) <= last_epoch]
    _write_rows(ep_log_path, EP_HEADER, episodes)
    _write_rows(epoch_log_path, EPOCH_HEADER, epochs)
    sac_log = os.path.join(tr, "sac_updates.csv")
    if os.path.exists(sac_log):
        agent.set_alpha(float(_read_rows(sac_log)[-1]["alpha"]))
    return {"epoch": last_epoch + 1, "episode": int(episodes[-1]["episode"]),
            "env_steps": int(episodes[-1]["env_steps"]), "updates": 0,
            "win_hist": [int(r["outcome"] == "win") for r in episodes[-100:]]}


def train(total_env_steps=int(1.5e7), n_generator_samples=10, episodes_per_point=2,
          mini_batch=256, max_ep_steps=6000, seed=0, wgan_critic_batch=64,
          plot_every_n_episodes=50, ckpt_every_n_epochs=5, obs_mode="extended", action_repeat=5,
          device="cpu", buffer_size=int(1e6), uniform_frac=0.3, terminate_on_kill=True,
          resume=None, actor_warmup_updates=None, reward_scale=0.1,
          opponent_curriculum=False, opponent_level=0, p_top=0.25):
    """n_generator_samples (Algorithm 2's n), episodes_per_point and
    wgan_critic_batch are NOT given in the paper -- flagged assumptions.
    mini_batch=256 is Table 2 exact. One epoch costs at most
    n_generator_samples * episodes_per_point * max_ep_steps sim steps.
    buffer_size 1e6 (Garage's default): run A's 2e5 flushed its rare crash and
    loss transitions within ~10 epochs, and crashes and losses came back.
    reward_scale 0.1: with +1500 / -1000 terminal outcomes at scale 1, run C's
    Q-loss rose 15-40x over run A's and alpha climbed 31x in three epochs at the
    maximum rate (policy entropy pinned below target, actions saturating) while
    crashes went 1 -> 9 -> 7. Scaling leaves the optimal policy unchanged and puts
    the balancing alpha back near run A's ~0.02.
    opponent_curriculum: each curriculum point plays adversary.LADDER[opp_level]
    (straight -> turn4 -> turn6 -> bfm_energy -> bfm_angles), or the top rung -- the
    paper's BFM expert -- with probability p_top. After each epoch the rung moves up
    if the win rate on it was >= 0.5 and down if < 0.1. Against the full BFM the
    warm-started policy won 0 of 10 even from easy starts, so the win reward and
    Eq.(9) never had anything to work with."""
    config = {k: v for k, v in locals().items() if k != "resume"}
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = AngleTacticEnv(max_steps=max_ep_steps, log_trajectory=True, obs_mode=obs_mode,
                         action_repeat=action_repeat, terminate_on_kill=terminate_on_kill,
                         adversary_type="ladder" if opponent_curriculum else "bfm",
                         opponent_level=opponent_level if opponent_curriculum else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim, device=device, reward_scale=reward_scale)
    buffer = ReplayBuffer(obs_dim, act_dim, size=buffer_size)
    curriculum = WGANCurriculum(uniform_frac=uniform_frac)

    ep_log_path = os.path.join(LOG_DIR, "angle_tactic_wgan_episodes.csv")
    epoch_log_path = os.path.join(LOG_DIR, "wgan_epochs.csv")
    sac_log_path = os.path.join(LOG_DIR, "sac_epochs.csv")
    ckpt_path = os.path.join(CKPT_DIR, "checkpoint_latest.pt")

    counters = {"epoch": 0, "episode": 0, "env_steps": 0, "updates": 0, "win_hist": []}
    if resume and os.path.isfile(resume):
        counters = _resume_full(resume, agent, buffer, curriculum)
        for path, header in ((ep_log_path, EP_HEADER), (epoch_log_path, EPOCH_HEADER), (sac_log_path, SAC_HEADER)):
            _keep_epochs_before(path, header, counters["epoch"])
        warmup = actor_warmup_updates or 0
        print(f"[resume] full checkpoint {resume}: epoch {counters['epoch']}, env_steps {counters['env_steps']}, "
              f"replay {buffer.size}, alpha {agent.alpha.item():.4f}")
    elif resume:
        counters = _warm_start_from_model_dir(resume, agent, ep_log_path, epoch_log_path)
        _write_rows(sac_log_path, SAC_HEADER, [])
        np.random.seed(seed + counters["epoch"])
        torch.manual_seed(seed + counters["epoch"])
        warmup = 20000 if actor_warmup_updates is None else actor_warmup_updates
        print(f"[resume] warm start from {resume}: actor + alpha {agent.alpha.item():.4f}; continuing at epoch "
              f"{counters['epoch']}, env_steps {counters['env_steps']}; critics + curriculum fresh, "
              f"actor frozen for {warmup} updates")
    else:
        for path, header in ((ep_log_path, EP_HEADER), (epoch_log_path, EPOCH_HEADER), (sac_log_path, SAC_HEADER)):
            _write_rows(path, header, [])
        warmup = actor_warmup_updates or 0

    episode, epoch, env_steps = counters["episode"], counters["epoch"], counters["env_steps"]
    win_hist = deque(counters["win_hist"], maxlen=100)  # rolling window over individual episodes
    opp_level = int(counters.get("opp_level", opponent_level))
    best_wr, best_ep = float(counters.get("best_win_rate", -1.0)), counters.get("best_epoch")
    os.makedirs(os.path.join(CKPT_DIR, "snapshots"), exist_ok=True)
    frozen_left = warmup

    while env_steps < total_env_steps:
        t_epoch, steps_epoch0 = time.time(), env_steps
        outcomes, q_losses, pi_losses, entropies, n_updates, n_frozen = Counter(), [], [], [], 0, 0
        opp_eps, opp_wins = Counter(), Counter()

        # ---- Algorithm 2, lines 5-11: k critic steps ----
        d_stats = curriculum.run_critic_steps(batch_size=wgan_critic_batch)

        # ---- line 12: sample n curriculum points ----
        z, x0_batch = curriculum.sample_for_training(n_generator_samples)

        per_point_win_rates, per_point_returns = [], []
        epoch_wins, epoch_completions = 0, 0

        # ---- lines 13-17: Algorithm 1 at each x0_j, whole episodes ----
        for j in range(n_generator_samples):
            x0_j = tuple(float(v) for v in x0_batch[j])
            point_wins, point_completions, point_returns = 0, 0, []
            if opponent_curriculum:   # same opponent for every episode at this point
                env.adversary.level = LADDER_TOP if np.random.rand() < p_top else opp_level

            for _ in range(episodes_per_point):
                if env_steps >= total_env_steps:
                    break
                obs, _ = env.reset(options={"init_condition": x0_j})
                ep_ret, done, info = 0.0, False, {}

                while not done and env_steps < total_env_steps:
                    a = agent.act(obs, deterministic=False)
                    next_obs, r, term, trunc, info = env.step(a)
                    done = term or trunc
                    buffer.add(obs, a, r, next_obs, float(term))  # truncation still bootstraps
                    ep_ret += r
                    env_steps += info["sim_steps"]
                    obs = next_obs

                    if buffer.size >= mini_batch:
                        stats = agent.update(buffer.sample(mini_batch), update_actor=frozen_left <= 0)
                        n_updates += 1
                        q_losses.append(stats["q_loss"])
                        if frozen_left > 0:
                            frozen_left -= 1
                            n_frozen += 1
                        else:
                            pi_losses.append(stats["pi_loss"])
                            entropies.append(stats["entropy"])

                if not done:
                    break  # step budget ran out mid-episode: discard, do not count

                # ---- episode completed ----
                episode += 1
                outcomes[info["outcome"]] += 1
                is_win = info["outcome"] == "win"
                opp_eps[info["opponent"]] += 1
                opp_wins[info["opponent"]] += int(is_win)
                win_hist.append(1 if is_win else 0)
                point_wins += int(is_win)
                point_completions += 1
                point_returns.append(ep_ret)
                epoch_wins += int(is_win)
                epoch_completions += 1

                _append_row(ep_log_path, [episode, epoch, env_steps, ep_ret, info["outcome"],
                                          float(np.mean(win_hist)), *x0_j, info["opponent"]])
                print(f"ep {episode:5d} | epoch {epoch:4d} | steps {env_steps:8d} "
                      f"| return {ep_ret:9.2f} | outcome {info['outcome']:15s} "
                      f"| win_rate(100) {float(np.mean(win_hist)):.3f}", flush=True)

                if plot_every_n_episodes > 0 and episode % plot_every_n_episodes == 0:
                    tag = f"ep{episode:05d}_{info['outcome']}"
                    plot_3d_trajectory(env.trajectory, os.path.join(PLOT_DIR, f"{tag}_3d.png"), title=tag)
                    write_acmi(env.trajectory, os.path.join(LOG_DIR, f"{tag}.acmi"))

            # ---- line 16: win rate / mean return at x0_j (None if nothing finished) ----
            if point_completions > 0:
                per_point_win_rates.append(point_wins / point_completions)
                per_point_returns.append(float(np.mean(point_returns)))
            else:
                per_point_win_rates.append(None)
                per_point_returns.append(None)

        # ---- lines 18-19: Eq.(9) labels, R_hat, WGAN-GP generator steps ----
        g_stats = curriculum.update_from_results(z, x0_batch, per_point_win_rates, per_point_returns)

        epoch_win_rate = (epoch_wins / epoch_completions) if epoch_completions > 0 else float("nan")
        _append_row(epoch_log_path, [
                epoch, episode, env_steps,
                d_stats.get("d_loss") if d_stats else None,
                d_stats.get("wasserstein_est") if d_stats else None,
                g_stats["g_loss"], g_stats["predictor_loss"], g_stats["n_valid"],
                epoch_win_rate, epoch_completions, len(curriculum.feasible),
            ])
        rate = (env_steps - steps_epoch0) / max(time.time() - t_epoch, 1e-6)
        other = sum(v for k, v in outcomes.items() if k not in ("win", "loss", "timeout", "crash"))
        used_level = opp_level
        level_name, top_name = LADDER[used_level], LADDER[LADDER_TOP]
        level_wins, level_eps = opp_wins[level_name], opp_eps[level_name]
        top_wins, top_eps = opp_wins[top_name], opp_eps[top_name]
        if opponent_curriculum and level_eps >= 4:
            if level_wins / level_eps >= 0.5 and opp_level < LADDER_TOP:
                opp_level += 1
            elif level_wins / level_eps < 0.1 and opp_level > 0:
                opp_level -= 1
        _append_row(sac_log_path, [
                epoch, env_steps, n_updates, n_frozen,
                float(np.mean(q_losses)) if q_losses else None,
                float(np.mean(pi_losses)) if pi_losses else None,
                agent.alpha.item(), float(np.mean(entropies)) if entropies else None, round(rate, 1),
                outcomes["win"], outcomes["loss"], outcomes["timeout"], outcomes["crash"], other,
                used_level if opponent_curriculum else None, level_wins, level_eps, top_wins, top_eps,
            ])
        print(f"[epoch {epoch:4d}] env_steps {env_steps:8d} | win {outcomes['win']} loss {outcomes['loss']} "
              f"timeout {outcomes['timeout']} crash {outcomes['crash']} other {other} | win_rate(100) "
              f"{float(np.mean(win_hist)) if win_hist else float('nan'):.3f} | feasible {len(curriculum.feasible)} "
              f"| alpha {agent.alpha.item():.4f} entropy {np.mean(entropies) if entropies else float('nan'):.2f}"
              f"{' (actor frozen)' if frozen_left > 0 else ''} "
              f"| {rate:.0f} sim steps/s"
              + (f" | opp {level_name} {level_wins}/{level_eps}, BFM {top_wins}/{top_eps}"
                 f" -> next {LADDER[opp_level]}" if opponent_curriculum else ""), flush=True)

        # Keep the best policy: checkpoint_latest.pt is overwritten every epoch, and run W's
        # ~80%-win peak (epochs 123-127) was lost that way when it later dipped.
        if len(win_hist) == win_hist.maxlen and np.mean(win_hist) > best_wr:
            best_wr, best_ep = float(np.mean(win_hist)), epoch
            torch.save(agent.pi.state_dict(), os.path.join(CKPT_DIR, "best_actor.pt"))
            with open(os.path.join(CKPT_DIR, "best_actor.json"), "w") as f:
                json.dump({"epoch": epoch, "env_steps": env_steps, "win_rate_100": best_wr,
                           "opponent": LADDER[opp_level] if opponent_curriculum else "bfm_angles"}, f, indent=2)
            print(f"[best] epoch {epoch}: rolling win rate {best_wr:.2f} -> checkpoints/best_actor.pt", flush=True)
        if epoch % 5 == 0:   # small actor-only history (~0.3 MB each) for evaluating the run's course
            torch.save(agent.pi.state_dict(), os.path.join(CKPT_DIR, "snapshots", f"actor_epoch{epoch:04d}.pt"))

        epoch += 1
        counters = {"epoch": epoch, "episode": episode, "env_steps": env_steps,
                    "updates": counters["updates"] + n_updates, "win_hist": list(win_hist),
                    "opp_level": opp_level, "best_win_rate": best_wr, "best_epoch": best_ep}
        save_checkpoint(ckpt_path, agent, buffer, curriculum, counters, config)
        if (epoch - 1) % ckpt_every_n_epochs == 0:
            torch.save(agent.pi.state_dict(), os.path.join(CKPT_DIR, "angle_policy_wgan_latest.pt"))
            torch.save(curriculum.G.state_dict(), os.path.join(CKPT_DIR, "wgan_generator_latest.pt"))
            torch.save(curriculum.D.state_dict(), os.path.join(CKPT_DIR, "wgan_critic_latest.pt"))
            if episode > 0:
                try:
                    plot_win_rate(ep_log_path, os.path.join(PLOT_DIR, "win_rate_curve.png"))
                except OSError as e:   # a locked/open log must not end the run
                    print(f"[log] skipped win-rate plot: {e}", flush=True)

    torch.save(agent.pi.state_dict(), os.path.join(CKPT_DIR, "angle_policy_wgan_final.pt"))
    torch.save(curriculum.G.state_dict(), os.path.join(CKPT_DIR, "wgan_generator_final.pt"))
    torch.save(curriculum.D.state_dict(), os.path.join(CKPT_DIR, "wgan_critic_final.pt"))
    print("Training complete. Final win_rate(100):", float(np.mean(win_hist)) if win_hist else float("nan"))
    return agent, curriculum


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total-env-steps", type=int, default=int(1.5e7),
                   help="TOTAL sim-step budget, including steps already done by a resumed run")
    p.add_argument("--n-generator-samples", type=int, default=10,
                   help="Algorithm 2's n -- curriculum points sampled per epoch (NOT in paper)")
    p.add_argument("--episodes-per-point", type=int, default=2,
                   help="complete episodes run at each curriculum point (NOT in paper)")
    p.add_argument("--mini-batch", type=int, default=256, help="Table 2 exact: SAC replay mini-batch")
    p.add_argument("--max-ep-steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wgan-critic-batch", type=int, default=64)
    p.add_argument("--plot-every-n-episodes", type=int, default=5,
                   help="3D trajectory plot every Nth episode (0 disables)")
    p.add_argument("--ckpt-every-n-epochs", type=int, default=5,
                   help="actor/G/D snapshots; the full resumable checkpoint is written every epoch")
    p.add_argument("--obs-mode", choices=["extended", "paper"], default="extended",
                   help="'paper' = Eq.(43) only; 'extended' adds attitude, rates and body-frame bandit direction")
    p.add_argument("--action-repeat", type=int, default=5,
                   help="sim steps (50 Hz) per agent decision; 1 = paper's literal 50 Hz decisions")
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | auto. The 2x256 nets at batch 256 are launch-overhead bound: "
                        "measured ~equal speed on CPU and an RTX 5060, so CPU is the default")
    p.add_argument("--buffer-size", type=int, default=int(1e6), help="replay capacity in agent decisions")
    p.add_argument("--uniform-frac", type=float, default=0.3,
                   help="share of curriculum points drawn uniformly from Table 1 (1.0 = no-curriculum ablation)")
    p.add_argument("--no-terminate-on-kill", dest="terminate_on_kill", action="store_false",
                   help="run A's rule: kills only label the episode, never end it, no terminal reward")
    p.add_argument("--reward-scale", type=float, default=0.1,
                   help="multiplier on rewards inside the SAC backup (Table 2 'reward scale'); see train()")
    p.add_argument("--opponent-curriculum", action="store_true",
                   help="climb adversary.LADDER (straight .. bfm_angles) by win rate; see train()")
    p.add_argument("--opponent-level", type=int, default=0, help="starting rung of the opponent ladder")
    p.add_argument("--p-top", type=float, default=0.25,
                   help="share of curriculum points played against the top rung (full BFM) regardless")
    p.add_argument("--run-dir", default=None,
                   help="put logs/, checkpoints/, plots/ under this folder (e.g. runs/D_fresh); default: next to this file")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="cap PyTorch CPU threads (0 = PyTorch default); useful when runs share the CPU")
    p.add_argument("--resume", default=None,
                   help="checkpoint file (exact resume) or model folder such as models/angle_tactic_A (actor warm start)")
    p.add_argument("--actor-warmup-updates", type=int, default=None,
                   help="critic-only updates before the actor trains (default 20000 for a model-folder warm start, else 0)")
    args = p.parse_args()
    if args.run_dir:
        set_run_dir(args.run_dir)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    train(total_env_steps=args.total_env_steps, n_generator_samples=args.n_generator_samples,
          episodes_per_point=args.episodes_per_point, mini_batch=args.mini_batch,
          max_ep_steps=args.max_ep_steps, seed=args.seed, wgan_critic_batch=args.wgan_critic_batch,
          plot_every_n_episodes=args.plot_every_n_episodes, ckpt_every_n_epochs=args.ckpt_every_n_epochs,
          obs_mode=args.obs_mode, action_repeat=args.action_repeat, device=args.device,
          buffer_size=args.buffer_size, uniform_frac=args.uniform_frac,
          terminate_on_kill=args.terminate_on_kill, resume=args.resume,
          actor_warmup_updates=args.actor_warmup_updates, reward_scale=args.reward_scale,
          opponent_curriculum=args.opponent_curriculum, opponent_level=args.opponent_level, p_top=args.p_top)
