"""
Automatic curriculum generation via WGAN-GP, Section 3.1 / Algorithm 2.

Reproduces:
  Eq.(9)  -- labelling positive (intermediate-difficulty) initial conditions
  Eq.(10) -- WGAN minimax objective
  Eq.(11) -- gradient penalty
  Eq.(12) -- environment-quality generator loss L_env
  Eq.(13) -- combined L_total = E[D(real)] - E[D(fake)] + gamma*L_env + L_GP

Section 3.1 cites two papers for this mechanism; the constants the base paper
leaves open are taken from them:

  [29] Gulrajani et al., "Improved Training of Wasserstein GANs" (NeurIPS 2017)
       -- gradient penalty (Eq.11). Their Algorithm 1: lambda=10, n_critic=5,
       Adam(lr=1e-4, beta1=0, beta2=0.9). Table 2 of the base paper gives lambda,
       dz and the G/D learning rates; n_critic and the Adam betas come from here.

  [27] Florensa, Held, Geng, Abbeel, "Automatic Goal Generation for RL" (ICML
       2018) -- the labelling scheme (Eq.9 is their GOID set). Their R_min=0.1,
       R_max=0.9 ("any Rmin in (0,0.25) and Rmax in (0.75,1) yields basically the
       same result", App. C), generator 2x128 ReLU, discriminator 2x256 ReLU.

  [26] Florensa et al., "Reverse Curriculum Generation for RL" (CoRL 2017) -- the
       start-state framing (a curriculum over initial conditions, matching
       Table 1's (Range, AA, ATA)) and mixing in old/broad starts so the policy
       does not forget them.

Design choices (each fixes a stall seen in run A, where the generator only ever
proposed near-identical tail chases and its weights moved 0.23 in 20 epochs):

  * Eq.(9) labels by WIN RATE in [0.1, 0.9], as written. Run A labelled by a
    15-85th percentile of episode return, which marks ~70% of points "feasible"
    whatever their difficulty -- and since draws earned the highest returns, the
    curriculum drifted toward wherever avoiding the fight paid.
  * Angles live on the unit circle: G emits [R, sin AA, cos AA, sin ATA, cos ATA]
    with each pair L2-normalised, and D / R_hat read that embedding. AA = +180 and
    -180 deg are the same geometry; as raw inputs they sat at opposite ends.
    (Same embedding as the Implementations/WGAN project.)
  * Several WGAN-GP iterations per epoch (n_critic critic steps + 1 generator
    step, each on a fresh latent batch) instead of one generator step on the <=10
    rollout points. L_env goes through the differentiable predictor R_hat, so
    generator steps need no new rollouts.
  * A fraction of each epoch's points are drawn uniformly from Table 1, so
    regions the generator never proposes still get evaluated and can become
    feasible -- and uniform_frac=1.0 is the paper's no-curriculum ablation (Sec 5.4).
  * L_env pulls the predicted win rate toward the CENTRE of the Eq.(9) band
    (0.5). Eq.(12) as written, -E[R(pi, G(z))], maximises win rate, i.e. pushes
    the curriculum toward the easiest starts -- the opposite of Eq.(9)'s
    intermediate-difficulty aim.

Neither [26] nor [27] has a separate "environment quality" term like Eq.(12).
WinRatePredictor (a shallow regression surrogate for the non-differentiable,
rollout-based R(pi*, G(z))) exists because Eq.(12) requires a gradient through
it -- our addition, not taken from the cited papers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Table 2 -- given exactly
LATENT_DIM = 128
GP_LAMBDA = 10.0
LR_GENERATOR = 1e-4
LR_CRITIC = 5e-5

# From [29] Gulrajani et al. Algorithm 1
N_CRITIC = 5
ADAM_BETAS = (0.0, 0.9)

# From [27] Florensa et al.: Eq.(9)'s intermediate-difficulty win-rate band
WIN_BAND = (0.1, 0.9)

# NOT given anywhere (paper, [26] or [27]) -- assumptions
GAMMA_ENV = 1.0          # weight of L_env in Eq.(13)
ENV_TARGET = 0.5         # L_env target: centre of WIN_BAND
N_GEN_STEPS = 5          # WGAN-GP iterations (n_critic D steps + 1 G step) per epoch
G_BATCH = 64             # latent batch per generator step
UNIFORM_FRAC = 0.3       # share of each epoch's points drawn uniformly from Table 1

# Table 1 domain for (Range, AA, ATA)
RANGE_BOUNDS = (1900.0, 5100.0)
AA_BOUNDS = (-180.0, 180.0)
ATA_BOUNDS = (-180.0, 180.0)
EMB_DIM = 5


def encode(x):
    """(Range m, AA deg, ATA deg) -> [R_norm, sin AA, cos AA, sin ATA, cos ATA]."""
    r = (x[:, 0] - RANGE_BOUNDS[0]) / (RANGE_BOUNDS[1] - RANGE_BOUNDS[0]) * 2 - 1
    aa, ata = torch.deg2rad(x[:, 1]), torch.deg2rad(x[:, 2])
    return torch.stack([r, torch.sin(aa), torch.cos(aa), torch.sin(ata), torch.cos(ata)], dim=-1)


def decode(e):
    """Inverse of encode: embedding -> (Range m, AA deg, ATA deg), angles in (-180, 180]."""
    r = (e[:, 0] + 1) / 2 * (RANGE_BOUNDS[1] - RANGE_BOUNDS[0]) + RANGE_BOUNDS[0]
    aa = torch.rad2deg(torch.atan2(e[:, 1], e[:, 2]))
    ata = torch.rad2deg(torch.atan2(e[:, 3], e[:, 4]))
    return torch.stack([r, aa, ata], dim=-1)


def _mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """G_psi: z ~ N(0,I), dim=128 (Table 2) -> x0 = (Range, AA, ATA), Eq.(26).
    hidden=128 matches [27]'s goal generator. The head emits the embedding: tanh for
    range (always inside the Table 1 box) and an L2-normalised (sin, cos) pair per
    angle, so fakes lie exactly on the manifold real embeddings occupy -- otherwise
    the critic could separate them by the pair's radius alone."""

    def __init__(self, latent_dim=LATENT_DIM, hidden=128):
        super().__init__()
        self.net = _mlp([latent_dim, hidden, hidden, EMB_DIM])

    def forward_emb(self, z):
        raw = self.net(z)
        return torch.cat([torch.tanh(raw[:, :1]), F.normalize(raw[:, 1:3], dim=-1),
                          F.normalize(raw[:, 3:5], dim=-1)], dim=-1)

    def forward(self, z):
        return decode(self.forward_emb(z))


class Critic(nn.Module):
    """D_phi: scores an initial condition for the Wasserstein distance, reading the
    embedding. hidden=256 matches [27]'s discriminator; no batch norm, per [29]
    (it would make D's output depend on the rest of the batch and break the
    per-sample gradient penalty)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.net = _mlp([EMB_DIM, hidden, hidden, 1])

    def forward_emb(self, e):
        return self.net(e).squeeze(-1)

    def forward(self, x):
        return self.forward_emb(encode(x))


class WinRatePredictor(nn.Module):
    """Shallow surrogate R_hat(x0) ~ observed rollout win rate, trained by regression
    on every observed (x0, win_rate). Rollout win rate needs the JSBSim simulator, so
    d(-R(pi*, G(z)))/d(psi) cannot be backpropagated directly; L_env goes through
    R_hat instead, which is differentiable end-to-end through G."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = _mlp([EMB_DIM, hidden, hidden, 1], out_act=nn.Sigmoid)  # win rate in [0,1]

    def forward_emb(self, e):
        return self.net(e).squeeze(-1)

    def forward(self, x):
        return self.forward_emb(encode(x))


class WinRateDataset:
    """All observed (x0, win_rate) pairs -- unfiltered (unlike FeasibleBuffer). The
    predictor needs the full landscape (easy, hard and intermediate) to regress on."""

    def __init__(self, capacity=20000):
        self.x = []
        self.y = []
        self.capacity = capacity

    def add(self, x0, win_rate):
        self.x.append([float(v) for v in x0])
        self.y.append(float(win_rate))
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
    """Eq.(9): initial conditions whose observed win rate lies in WIN_BAND -- learnable
    but not yet mastered. These are the WGAN's 'real' samples."""

    def __init__(self, capacity=20000, band=WIN_BAND):
        self.data = []
        self.capacity = capacity
        self.band = band

    def add(self, x0, win_rate):
        if self.band[0] <= win_rate <= self.band[1]:
            self.data.append([float(v) for v in x0])
            if len(self.data) > self.capacity:
                self.data.pop(0)

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
                 gamma_env=GAMMA_ENV, env_target=ENV_TARGET, n_gen_steps=N_GEN_STEPS,
                 g_batch=G_BATCH, uniform_frac=UNIFORM_FRAC, lr_predictor=1e-3, predictor_steps=20):
        self.latent_dim = latent_dim
        self.gp_lambda = gp_lambda
        self.n_critic = n_critic
        self.gamma_env = gamma_env
        self.env_target = env_target
        self.n_gen_steps = n_gen_steps
        self.g_batch = g_batch
        self.uniform_frac = uniform_frac
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
            loss = F.mse_loss(self.R_hat(xb), yb)
            self.r_opt.zero_grad()
            loss.backward()
            self.r_opt.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else None

    def sample_latent(self, n):
        return torch.randn(n, self.latent_dim)

    def _gradient_penalty(self, e_real, e_fake):
        """Eq.(11): E[(||grad D(e_hat)||_2 - 1)^2], e_hat interpolated in the critic's
        input space (the embedding)."""
        eps = torch.rand(e_real.size(0), 1)
        e_hat = (eps * e_real + (1 - eps) * e_fake).requires_grad_(True)
        d_hat = self.D.forward_emb(e_hat)
        grads = torch.autograd.grad(outputs=d_hat, inputs=e_hat, grad_outputs=torch.ones_like(d_hat),
                                    create_graph=True, retain_graph=True)[0]
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def critic_step(self, batch_size=64):
        """One critic update on Eq.(13)'s Wasserstein + GP terms (L_env has no
        gradient wrt the critic). None until Eq.(9) has labelled anything."""
        real = self.feasible.sample(batch_size)
        if real is None:
            return None
        e_real = encode(real)
        with torch.no_grad():
            e_fake = self.G.forward_emb(self.sample_latent(batch_size))
        d_real = self.D.forward_emb(e_real).mean()
        d_fake = self.D.forward_emb(e_fake).mean()
        d_loss = -(d_real - d_fake) + self.gp_lambda * self._gradient_penalty(e_real, e_fake)
        self.d_opt.zero_grad()
        d_loss.backward()
        self.d_opt.step()
        return {"d_loss": d_loss.item(), "wasserstein_est": (d_real - d_fake).item()}

    def run_critic_steps(self, batch_size=64):
        """Algorithm 2, lines 5-11: n_critic critic steps."""
        last = None
        for _ in range(self.n_critic):
            last = self.critic_step(batch_size)
        return last

    def _generator_step(self, adversarial):
        """Algorithm 2, line 19: one generator step on a fresh latent batch.
        Wasserstein term only once the critic has real samples to compare against;
        L_env only once the predictor has seen any outcome."""
        if not adversarial and len(self.win_rate_data) == 0:
            return None
        e = self.G.forward_emb(self.sample_latent(self.g_batch))
        g_loss = torch.zeros(())
        if adversarial:
            g_loss = g_loss - self.D.forward_emb(e).mean()
        if len(self.win_rate_data) > 0:
            L_env = ((self.R_hat.forward_emb(e) - self.env_target) ** 2).mean()
            g_loss = g_loss + self.gamma_env * L_env
        self.g_opt.zero_grad()
        g_loss.backward()
        self.g_opt.step()
        return g_loss.item()

    def sample_for_training(self, n):
        """Algorithm 2, line 12: n initial conditions -- round(n * uniform_frac) drawn
        uniformly from Table 1, the rest from G. The caller runs Algorithm 1 from each
        (lines 13-17) and hands the win rates back to update_from_results."""
        n_uni = int(round(n * self.uniform_frac))
        z = self.sample_latent(n - n_uni)
        with torch.no_grad():
            x_gen = self.G(z).numpy()
        x_uni = np.column_stack([np.random.uniform(*RANGE_BOUNDS, n_uni),
                                 np.random.uniform(*AA_BOUNDS, n_uni),
                                 np.random.uniform(*ATA_BOUNDS, n_uni)])
        return z, np.concatenate([x_gen, x_uni]).astype(np.float32)

    @staticmethod
    def _ok(v):
        return v is not None and not np.isnan(float(v))

    def update_from_results(self, z, x0, win_rates, rewards=None, update_generator=True):
        """Algorithm 2, lines 16-19. win_rates[j] = fraction of COMPLETED episodes at
        x0[j] that self won (None if none completed -> skipped). Labels Eq.(9), trains
        R_hat, then runs n_gen_steps WGAN-GP iterations. `z` and `rewards` are accepted
        for interface compatibility; neither is used (labels are by win rate, and
        generator steps draw fresh latents)."""
        valid = [i for i in range(len(x0)) if self._ok(win_rates[i])]
        for i in valid:
            self.win_rate_data.add(x0[i], win_rates[i])
            self.feasible.add(x0[i], float(win_rates[i]))
        if not valid:
            return {"g_loss": None, "mean_win_rate": float("nan"), "predictor_loss": None, "n_valid": 0}

        predictor_loss = self._train_predictor()
        mean_win_rate = float(np.mean([float(win_rates[i]) for i in valid]))
        g_losses = []
        if update_generator:
            for _ in range(self.n_gen_steps):
                d_stats = self.run_critic_steps()
                g = self._generator_step(adversarial=d_stats is not None)
                if g is not None:
                    g_losses.append(g)
        return {"g_loss": float(np.mean(g_losses)) if g_losses else None, "mean_win_rate": mean_win_rate,
                "predictor_loss": predictor_loss, "n_valid": len(valid)}

    def sample_curriculum(self, n):
        """Draw n initial conditions from the generator for env.reset()."""
        with torch.no_grad():
            return self.G(self.sample_latent(n)).numpy()

    def state_dict(self):
        sd = {m: getattr(self, m).state_dict() for m in ("G", "D", "R_hat", "g_opt", "d_opt", "r_opt")}
        sd["feasible"] = list(self.feasible.data)
        sd["win_rate_x"], sd["win_rate_y"] = list(self.win_rate_data.x), list(self.win_rate_data.y)
        return sd

    def load_state_dict(self, sd):
        for m in ("G", "D", "R_hat", "g_opt", "d_opt", "r_opt"):
            getattr(self, m).load_state_dict(sd[m])
        self.feasible.data = list(sd["feasible"])
        self.win_rate_data.x, self.win_rate_data.y = list(sd["win_rate_x"]), list(sd["win_rate_y"])
