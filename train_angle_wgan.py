"""
Angle-tactic training loop implementing Algorithm 2's structure:

  for each epoch:
      for k critic steps:                                  (Alg.2 lines 5-11)
          update critic on feasible-buffer reals vs G(z) fakes
      sample n latent codes -> n initial conditions x0_j = G(z_j)   (line 12)
      for each x0_j:                                       (lines 13-17)
          run `episodes_per_point` COMPLETE episodes from x0_j. Every step
          stores a transition and does one SAC update (Algorithm 1).
          Episodes always run to their natural end (timeout / crash / disengage /
          adversary crash) -- a goal never terminates them.
          Win / loss is decided at episode end by whoever satisfied the WEZ
          condition FIRST (env sets info["outcome"] to win / loss / draw).
          win_rate_j = wins / completed episodes at x0_j
          return_j   = mean episode return at x0_j
      update generator from the (x0_j, win_rate_j, return_j) triples (lines 18-19)

An episode cut short by the global step budget is discarded (not counted),
so win rates are only ever computed from finished episodes.

env_steps / total_env_steps count 50 Hz SIM steps (as in the paper's 1.5e7), not
agent decisions: with action_repeat=5 the agent decides -- and SAC updates --
once per 5 sim steps.
"""

import os
import csv
import argparse
from collections import deque

import numpy as np
import torch

from env import AngleTacticEnv
from sac import SACAgent, ReplayBuffer
from wgan import WGANCurriculum
from export import write_acmi
from visualize import plot_3d_trajectory, plot_win_rate

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
CKPT_DIR = os.path.join(HERE, "checkpoints")
PLOT_DIR = os.path.join(HERE, "plots")
for d in (LOG_DIR, CKPT_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)


def train(total_env_steps=int(1.5e7), n_generator_samples=10, episodes_per_point=2,
          mini_batch=256, max_ep_steps=6000, seed=0, wgan_critic_batch=64,
          plot_every_n_episodes=50, ckpt_every_n_epochs=5, obs_mode="extended", action_repeat=5):
    """n_generator_samples (Algorithm 2's n), episodes_per_point and
    wgan_critic_batch are NOT given in the paper -- flagged assumptions.
    mini_batch=256 is Table 2 exact. One epoch costs roughly
    n_generator_samples * episodes_per_point * max_ep_steps env steps."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = AngleTacticEnv(max_steps=max_ep_steps, log_trajectory=True,
                         obs_mode=obs_mode, action_repeat=action_repeat)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim)
    buffer = ReplayBuffer(obs_dim, act_dim, size=200000)
    curriculum = WGANCurriculum()

    win_hist = deque(maxlen=100)  # rolling window over individual episodes
    ep_log_path = os.path.join(LOG_DIR, "angle_tactic_wgan_episodes.csv")
    epoch_log_path = os.path.join(LOG_DIR, "wgan_epochs.csv")
    with open(ep_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "epoch", "env_steps", "return",
                                 "outcome", "win_rate_100", "x0_range", "x0_aa", "x0_ata"])
    with open(epoch_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "episode", "env_steps", "d_loss", "wasserstein_est",
                                 "g_loss", "predictor_loss", "n_valid_points",
                                 "epoch_win_rate", "epoch_completions", "feasible_buffer_size"])

    episode = 0
    epoch = 0
    env_steps = 0

    while env_steps < total_env_steps:
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
                        agent.update(buffer.sample(mini_batch))

                if not done:
                    break  # step budget ran out mid-episode: discard, do not count

                # ---- episode completed ----
                episode += 1
                is_win = info["outcome"] == "win"  # self satisfied the WEZ condition first
                win_hist.append(1 if is_win else 0)
                point_wins += int(is_win)
                point_completions += 1
                point_returns.append(ep_ret)
                epoch_wins += int(is_win)
                epoch_completions += 1

                with open(ep_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([episode, epoch, env_steps, ep_ret, info["outcome"],
                                             float(np.mean(win_hist)), *x0_j])
                print(f"ep {episode:5d} | epoch {epoch:4d} | steps {env_steps:8d} "
                      f"| return {ep_ret:9.2f} | outcome {info['outcome']:15s} "
                      f"| win_rate(100) {float(np.mean(win_hist)):.3f}")

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

        # ---- lines 18-19: environment-quality loss + generator update ----
        g_stats = curriculum.update_from_results(z, x0_batch, per_point_win_rates, per_point_returns)

        epoch_win_rate = (epoch_wins / epoch_completions) if epoch_completions > 0 else float("nan")
        with open(epoch_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, episode, env_steps,
                d_stats.get("d_loss") if d_stats else None,
                d_stats.get("wasserstein_est") if d_stats else None,
                g_stats["g_loss"], g_stats["predictor_loss"], g_stats["n_valid"],
                epoch_win_rate, epoch_completions, len(curriculum.feasible),
            ])
        print(f"[epoch {epoch:4d}] env_steps {env_steps:8d} | epoch win_rate "
              f"{epoch_win_rate:.3f} ({epoch_completions} completed episodes over "
              f"{n_generator_samples} points) | feasible_buffer {len(curriculum.feasible)}")

        if epoch % ckpt_every_n_epochs == 0:
            torch.save(agent.pi.state_dict(), os.path.join(CKPT_DIR, "angle_policy_wgan_latest.pt"))
            torch.save(curriculum.G.state_dict(), os.path.join(CKPT_DIR, "wgan_generator_latest.pt"))
            torch.save(curriculum.D.state_dict(), os.path.join(CKPT_DIR, "wgan_critic_latest.pt"))
            if episode > 0:
                plot_win_rate(ep_log_path, os.path.join(PLOT_DIR, "win_rate_curve.png"))

        epoch += 1

    torch.save(agent.pi.state_dict(), os.path.join(CKPT_DIR, "angle_policy_wgan_final.pt"))
    torch.save(curriculum.G.state_dict(), os.path.join(CKPT_DIR, "wgan_generator_final.pt"))
    torch.save(curriculum.D.state_dict(), os.path.join(CKPT_DIR, "wgan_critic_final.pt"))
    print("Training complete. Final win_rate(100):", float(np.mean(win_hist)) if win_hist else float("nan"))
    return agent, curriculum


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total-env-steps", type=int, default=int(1.5e7))
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
    p.add_argument("--ckpt-every-n-epochs", type=int, default=5)
    p.add_argument("--obs-mode", choices=["extended", "paper"], default="extended",
                   help="'paper' = Eq.(43) only; 'extended' adds attitude, rates and body-frame bandit direction")
    p.add_argument("--action-repeat", type=int, default=5,
                   help="sim steps (50 Hz) per agent decision; 1 = paper's literal 50 Hz decisions")
    args = p.parse_args()
    train(total_env_steps=args.total_env_steps, n_generator_samples=args.n_generator_samples,
          episodes_per_point=args.episodes_per_point, mini_batch=args.mini_batch,
          max_ep_steps=args.max_ep_steps, seed=args.seed, wgan_critic_batch=args.wgan_critic_batch,
          plot_every_n_episodes=args.plot_every_n_episodes, ckpt_every_n_epochs=args.ckpt_every_n_epochs,
          obs_mode=args.obs_mode, action_repeat=args.action_repeat)