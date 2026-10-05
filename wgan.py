"""
Automatic curriculum generation via WGAN-GP, Section 3.1 / Algorithm 2.

Reproduces:
  Eq.(9)  -- labelling positive (intermediate-difficulty) initial conditions
  Eq.(10) -- WGAN minimax objective
  Eq.(11) -- gradient penalty
  Eq.(12) -- environment-quality generator loss L_env
  Eq.(13) -- combined L_total = E[D(real)] - E[D(fake)] + gamma*L_env + L_GP

Section 3.1 explicitly cites two papers for this mechanism, and this module
now pulls its remaining (paper-unspecified) constants from THEM rather than
from arbitrary guesses:

  [29] Gulrajani et al., "Improved Training of Wasserstein GANs" (NeurIPS
       2017) -- cited for the gradient penalty (Eq.11). Their Algorithm 1
       gives exact defaults: lambda=10, n_critic=5, Adam(lr=1e-4, beta1=0,
       beta2=0.9). Table 2 of the base paper gives lambda, dz, and the G/D
       learning rates directly; n_critic and the Adam betas were previously
       guessed (betas=(0.5,0.9), a DCGAN-era convention) -- corrected below
       to Gulrajani's actual (0, 0.9).

  [27] Florensa, Held, Geng, Abbeel, "Automatic Goal Generation for RL"
       (ICML 2018) -- cited for the labelling scheme (Eq.9's GOID_i is their
       Eq.4, verbatim). Their reported hyperparameters: R_min=0.1, R_max=0.9
       (Appendix C: "any Rmin in (0,0.25) and Rmax in (0.75,1) yields
       basically the same result" -- 0.1/0.9 is their actual choice, not
       just a midpoint convention); generator = 2x128 ReLU, discriminator =
       2x256 ReLU (Appendix B.4). Previously used an unjustified 0.3/0.7
       band and a uniform 256/256 architecture -- both corrected below.

  [26] Florensa, Held, Wulfmeier, Zhang, Abbeel, "Reverse Curriculum
       Generation for RL" (CoRL 2017) -- the base paper's OTHER citation
       here. Same Rmin/Rmax=0.1/0.9. Contributes the "start-state" framing
       (their curriculum is over initial conditions, matching Table 1's
       (Range,AA,ATA) parameterization much more literally than [27]'s
       goal-space framing) and a replay buffer of past good starts to avoid
       catastrophic forgetting -- our FeasibleBuffer already plays this
       role. Their alternative state-generation mechanism (Brownian-motion
       "SampleNearby" expansion from known good points, Algorithm 1/
       Procedure 2) is NOT adopted here, since the base paper's own Eq.9-13
       commit to a GAN generator network, matching [27]'s mechanism, not
       [26]'s random-walk one -- noted as a documented alternative, not
       implemented.

Neither [26] nor [27] has a separate differentiable "environment quality"
term like Eq.(12)'s L_env: in [27], fooling the discriminator IS the full
generator signal (their D is trained directly on the binary GOID label, so
G's loss is already end-to-end differentiable through D with no extra
network needed). The base paper's Eq.(12)/(13) write L_env as a term
SEPARATE from the Wasserstein distribution-matching loss -- a structure
neither cited paper has. The WinRatePredictor below (a shallow regression
surrogate for the non-differentiable rollout-based R(pi*,G(z))) is there
because Eq.(12) specifically requires it, not because either source paper
uses one; this is flagged as our own addition, needed for THIS paper's
specific equations rather than taken from [26]/[27].
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Table 2 -- given exactly
LATENT_DIM = 128
GP_LAMBDA = 10.0
LR_GENERATOR = 1e-4
LR_CRITIC = 5e-5

# From [29] Gulrajani et al. Algorithm 1 (exact values, not guessed)
N_CRITIC = 5
ADAM_BETAS = (0.0, 0.9)  # was (0.5, 0.9) -- wrong, DCGAN-era convention

# From [26]/[27] Florensa et al. (their actual reported values, not a guess)
P_LO, P_HI = 15.0, 85.0  # feasible band = 15th..85th percentile of the episode-reward distribution

# NOT given anywhere (paper, [26], or [27]) -- still a genuine assumption
GAMMA_ENV = 1.0

# Table 1 domain for (Range, AA, ATA), used to de-normalize generator output
RANGE_BOUNDS = (1900.0, 5100.0)
AA_BOUNDS = (-180.0, 180.0)
ATA_BOUNDS = (-180.0, 180.0)


def _mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """G_psi: z ~ N(0,I), dim=128 (Table 2) -> x0 = (Range, AA, ATA), Eq.(26).
    hidden=128 matches [27] Florensa18a Appendix B.4 ("two hidden layers
    with 128 nodes" for their goal generator)."""

    def __init__(self, latent_dim=LATENT_DIM, hidden=128):
        super().__init__()
        self.net = _mlp([latent_dim, hidden, hidden, 3], out_act=nn.Tanh)

    def forward(self, z):
        raw = self.net(z)  # in (-1, 1)^3
        rng = (raw[:, 0] + 1) / 2 * (RANGE_BOUNDS[1] - RANGE_BOUNDS[0]) + RANGE_BOUNDS[0]
        aa = raw[:, 1] * AA_BOUNDS[1]
        ata = raw[:, 2] * ATA_BOUNDS[1]
        return torch.stack([rng, aa, ata], dim=-1)


class Critic(nn.Module):
    """D_phi: scores an initial condition x=(Range,AA,ATA) for Wasserstein
    distance. hidden=256 matches [27] Florensa18a's goal discriminator
    ("two hidden layers with 256 nodes"). No batch norm, per [29] Gulrajani
    Sec 4 ("No critic batch normalization") -- plain MLP as below."""

    def __init__(self, hidden=256):
        super().__init__()
        self.net = _mlp([3, hidden, hidden, 1])

    def forward(self, x):
        x_norm = torch.stack([
            (x[:, 0] - RANGE_BOUNDS[0]) / (RANGE_BOUNDS[1] - RANGE_BOUNDS[0]) * 2 - 1,
            x[:, 1] / AA_BOUNDS[1],
            x[:, 2] / ATA_BOUNDS[1],
        ], dim=-1)
        return self.net(x_norm).squeeze(-1)


class WinRatePredictor(nn.Module):
    """Shallow surrogate network R_hat(x0) ~ observed rollout win rate.

    Rollout win rate is a non-differentiable black-box (it requires running
    the JSBSim simulator to completion), so the true Eq.(12) gradient
    d(-R(pi*, G(z)))/d(psi) cannot be backpropagated directly -- the previous
    implementation silently zeroed this term (win_rates was detached numpy,
    no grad graph). This predictor is trained by supervised regression on
    observed (x0, win_rate) pairs, then substituted for R in Eq.(12)/(13):
    L_env = -E_z[R_hat(G(z))], which IS differentiable end-to-end through
    the predictor and through G, giving the generator a real gradient signal.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.net = _mlp([3, hidden, hidden, 1], out_act=nn.Sigmoid)  # win rate in [0,1]

    def forward(self, x):
        x_norm = torch.stack([
            (x[:, 0] - RANGE_BOUNDS[0]) / (RANGE_BOUNDS[1] - RANGE_BOUNDS[0]) * 2 - 1,
            x[:, 1] / AA_BOUNDS[1],
            x[:, 2] / ATA_BOUNDS[1],
        ], dim=-1)
        return self.net(x_norm).squeeze(-1)


class WinRateDataset:
    """All observed (x0, win_rate) pairs -- unfiltered (unlike FeasibleBuffer,
    which only keeps the Eq.9 intermediate-difficulty band). The predictor
    needs the full landscape (easy, hard, and intermediate points) to learn
    a useful regression surface."""

    def __init__(self, capacity=20000):
        self.x = []
        self.y = []
        self.capacity = capacity

    def add(self, x0, win_rate):
        self.x.append(x0)
        self.y.append(win_rate)
        if len(self.x) > self.capacity:
            self.x.pop(0)
            self.y.pop(0)

    def sample(self, n):
        if len(self.x) == 0:
            return None, None
        idx = np.random.randint(0, len(self.x), size=min(n, len(self.x)))
        xb = torch.as_tensor(np.array(self.x)[idx], dtype=torch.float32)
        yb = torch.as_tensor(np.array(self.y)[idx], dtype=torch.float32)
        return xb, yb

    def __len__(self):
        return len(self.x)


class FeasibleBuffer:
    """Initial conditions whose episode reward lies between the P_LO-th and
    P_HI-th percentile of the rolling distribution of episode rewards seen so
    far. The band is recomputed on every add, so it moves with the rewards.
    Until `min_history` rewards exist, points wait in `pending` and are
    labelled against the first band computed."""

    def __init__(self, capacity=20000, history=2000, min_history=20):
        self.data = []
        self.capacity = capacity
        self.rewards = deque(maxlen=history)
        self.min_history = min_history
        self.pending = []

    def band(self):
        if len(self.rewards) < self.min_history:
            return None
        lo, hi = np.percentile(np.asarray(self.rewards), [P_LO, P_HI])
        return float(lo), float(hi)

    def _label(self, x0, reward, band):
        if band[0] <= reward <= band[1]:
            self.data.append(x0)
            if len(self.data) > self.capacity:
                self.data.pop(0)

    def add(self, x0, reward):
        self.rewards.append(reward)
        b = self.band()
        if b is None:
            self.pending.append((x0, reward))
            return
        for px, pr in self.pending:
            self._label(px, pr, b)
        self.pending = []
        self._label(x0, reward, b)

    def sample(self, n):
        if len(self.data) == 0:
            return None
        idx = np.random.randint(0, len(self.data), size=n)
        return torch.as_tensor(np.array(self.data)[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.data)


class WGANCurriculum:
    """Algorithm 2: automatic curriculum generation with WGANs."""

    def __init__(self, latent_dim=LATENT_DIM, gp_lambda=GP_LAMBDA,
                 lr_g=LR_GENERATOR, lr_d=LR_CRITIC, n_critic=N_CRITIC,
                 gamma_env=GAMMA_ENV, lr_predictor=1e-3, predictor_steps=20):
        self.latent_dim = latent_dim
        self.gp_lambda = gp_lambda
        self.n_critic = n_critic
        self.gamma_env = gamma_env
        self.predictor_steps = predictor_steps  # NOT given in paper -- assumed

        self.G = Generator(latent_dim)
        self.D = Critic()
        self.R_hat = WinRatePredictor()
        self.g_opt = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=ADAM_BETAS)
        self.d_opt = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=ADAM_BETAS)
        self.r_opt = torch.optim.Adam(self.R_hat.parameters(), lr=lr_predictor)  # lr NOT given -- assumed

        self.feasible = FeasibleBuffer()
        self.win_rate_data = WinRateDataset()

    def _train_predictor(self, steps=None):
        """Supervised regression: R_hat(x0) -> observed win_rate (MSE)."""
        steps = steps or self.predictor_steps
        losses = []
        for _ in range(steps):
            xb, yb = self.win_rate_data.sample(64)
            if xb is None:
                break
            pred = self.R_hat(xb)
            loss = F.mse_loss(pred, yb)
            self.r_opt.zero_grad()
            loss.backward()
            self.r_opt.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else None

    def sample_latent(self, n):
        return torch.randn(n, self.latent_dim)

    def _gradient_penalty(self, real, fake):
        """Eq.(11): E[(||grad_xhat D(xhat)||_2 - 1)^2], xhat interpolated."""
        eps = torch.rand(real.size(0), 1)
        xhat = (eps * real + (1 - eps) * fake).requires_grad_(True)
        d_xhat = self.D(xhat)
        grads = torch.autograd.grad(
            outputs=d_xhat, inputs=xhat,
            grad_outputs=torch.ones_like(d_xhat),
            create_graph=True, retain_graph=True,
        )[0]
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def critic_step(self, batch_size=64):
        """Algorithm 2, 'for k critic steps': update D via Eq.(13)'s
        Wasserstein + GP terms (L_env has zero gradient wrt phi)."""
        real = self.feasible.sample(batch_size)
        if real is None:
            return None  # not enough labelled-positive samples yet
        z = self.sample_latent(batch_size)
        fake = self.G(z).detach()

        d_real = self.D(real).mean()
        d_fake = self.D(fake).mean()
        gp = self._gradient_penalty(real, fake)
        d_loss = -(d_real - d_fake) + self.gp_lambda * gp  # minimize -(Wasserstein est.)

        self.d_opt.zero_grad()
        d_loss.backward()
        self.d_opt.step()
        return {"d_loss": d_loss.item(), "wasserstein_est": (d_real - d_fake).item()}

    def sample_for_training(self, n):
        """Algorithm 2, line 12: sample n latent codes and the initial
        conditions G(z) they produce. The caller (train_angle_wgan.py) uses
        these n points to actually EXECUTE ALGORITHM 1 (lines 13-17) --
        i.e. run real SAC training batches reset to each x0_j -- and tallies
        win/loss outcomes from those same training episodes as they occur.
        This class does not itself run any environment steps or policy
        training; it only produces the curriculum points and, afterwards
        (update_from_results), consumes the resulting win rates."""
        z = self.sample_latent(n)
        with torch.no_grad():
            x0 = self.G(z).numpy()
        return z, x0

    @staticmethod
    def _ok(v):
        return v is not None and not np.isnan(float(v))

    def update_from_results(self, z, x0, win_rates, rewards=None, update_generator=True):
        """Algorithm 2, lines 18-19. Per curriculum point j the caller supplies:
          win_rates[j] -- fraction of COMPLETED episodes at x0[j] that self won,
                          or None if no episode completed (point is skipped).
          rewards[j]   -- mean episode return at x0[j] (fills the feasible buffer via
                          the dynamic percentile band); optional, None entries skipped.
        update_generator=False: only the buffers/predictor are filled."""
        valid = [i for i in range(len(x0)) if self._ok(win_rates[i])]
        for i in valid:
            self.win_rate_data.add(x0[i], float(win_rates[i]))
            if rewards is not None and self._ok(rewards[i]):
                self.feasible.add(x0[i], float(rewards[i]))

        if not valid:  # nothing completed this epoch: no evidence, no update
            return {"g_loss": None, "mean_win_rate": float("nan"),
                    "predictor_loss": None, "n_valid": 0}

        predictor_loss = self._train_predictor()
        mean_win_rate = float(np.mean([float(win_rates[i]) for i in valid]))

        if not update_generator or z is None:
            return {"g_loss": None, "mean_win_rate": mean_win_rate,
                    "predictor_loss": predictor_loss, "n_valid": len(valid)}

        # ---- Eq.(12)/(13): recompute G(z) WITH gradient, valid points only ----
        x0_grad = self.G(z[valid])
        L_env = -self.R_hat(x0_grad).mean()           # backprops through R_hat AND G
        d_fake = self.D(x0_grad).mean()
        g_loss = -d_fake + self.gamma_env * L_env

        self.g_opt.zero_grad()
        g_loss.backward()
        self.g_opt.step()
        return {"g_loss": g_loss.item(), "mean_win_rate": mean_win_rate,
                "predictor_loss": predictor_loss, "n_valid": len(valid)}

    def run_critic_steps(self, batch_size=64):
        """Algorithm 2, lines 5-11: n_critic critic steps for one epoch."""
        last_d_stats = None
        for _ in range(self.n_critic):
            last_d_stats = self.critic_step(batch_size)
        return last_d_stats

    def sample_curriculum(self, n):
        """Draw n initial conditions from the generator for env.reset()."""
        with torch.no_grad():
            z = self.sample_latent(n)
            x0 = self.G(z).numpy()
        return x0  # each row: (Range, AA, ATA)