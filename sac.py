"""
Soft Actor-Critic matching Algorithm 1 (Eq. 14-20) and Table 2 hyperparameters:
2 hidden layers x256, ReLU, Adam, batch 256, gamma 0.99, tau 5e-3,
target entropy H_bar = -4, initial alpha 0.1, lr 3e-4 (Q, pi, alpha).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def resolve_device(device="auto"):
    """'auto' -> CUDA when available, else CPU. Anything else is passed to torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.body = mlp([obs_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.body(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # tanh-squash log-prob correction (Eq. 17-18 reparameterized policy gradient)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        return action, log_prob, torch.tanh(mu)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6)):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.as_tensor(self.obs[idx]), torch.as_tensor(self.act[idx]),
                torch.as_tensor(self.rew[idx]), torch.as_tensor(self.next_obs[idx]),
                torch.as_tensor(self.done[idx]))


class SACAgent:
    """Implements Algorithm 1's soft option-action-value + intra-option policy updates.
    Eq.(14)-(16) with TWIN soft Q-networks (Q_phi1, Q_phi2) and their targets: the
    bootstrap and the policy objective use min(Q1, Q2). Fig.3 labels both critic
    boxes "Double Q", and Garage's SAC -- the library the paper trained with
    (Sec 5.1) -- is twin-Q, so this matches what was actually run. A single Q
    overestimates under the max-entropy backup and destabilised training here.
    The Q-loss is minimized via standard MSE + autograd + Adam, which implements
    the paper's own stated objective ("minimizing the squared residual error",
    Sec 2.2/4.2) with the mathematically-correct gradient-descent sign -- see
    the worked sign derivation for Eq.(14) in the accompanying analysis: the
    equation as literally transcribed (-xi * grad_Q * (target - Q)) is gradient
    ASCENT on that squared error and would diverge; MSE+Adam is the intended,
    equivalent-in-objective implementation.
    """

    def __init__(self, obs_dim, act_dim, hidden=256, gamma=0.99, tau=5e-3,
                 lr_q=3e-4, lr_pi=3e-4, lr_alpha=3e-4, target_entropy=-4.0,
                 init_alpha=0.1, device="cpu"):
        self.gamma, self.tau = gamma, tau
        self.device = resolve_device(device)
        dev = self.device
        self.q1, self.q2 = QNet(obs_dim, act_dim, hidden).to(dev), QNet(obs_dim, act_dim, hidden).to(dev)
        self.q1_targ = QNet(obs_dim, act_dim, hidden).to(dev)
        self.q2_targ = QNet(obs_dim, act_dim, hidden).to(dev)
        for q, q_targ in ((self.q1, self.q1_targ), (self.q2, self.q2_targ)):
            q_targ.load_state_dict(q.state_dict())
            for p in q_targ.parameters():
                p.requires_grad = False
        self.pi = GaussianPolicy(obs_dim, act_dim, hidden).to(dev)

        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_q)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=lr_pi)

        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, device=dev, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _, a_det = self.pi.sample(obs_t)
        return (a_det if deterministic else a).squeeze(0).cpu().numpy()

    def _soft_update(self):
        # Algorithm 1, line 12: phi_bar <- sigma*phi + (1-sigma)*phi_bar, sigma=tau (Table 2)
        with torch.no_grad():
            for q, q_targ in ((self.q1, self.q1_targ), (self.q2, self.q2_targ)):
                for p, pt in zip(q.parameters(), q_targ.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def update(self, batch):
        obs, act, rew, next_obs, done = (t.to(self.device, non_blocking=True) for t in batch)

        # ---- Eq.(15)-(16): bootstrapped target from the smaller of the two target Qs ----
        with torch.no_grad():
            next_a, next_logp, _ = self.pi.sample(next_obs)
            q_next = torch.min(self.q1_targ(next_obs, next_a), self.q2_targ(next_obs, next_a))
            u_targ = q_next - self.alpha * next_logp                        # Eq.(16)
            backup = rew + self.gamma * (1 - done) * u_targ                  # Eq.(15)

        # ---- Eq.(14): minimize squared residual (correct-sign gradient descent) ----
        q_loss = F.mse_loss(self.q1(obs, act), backup) + F.mse_loss(self.q2(obs, act), backup)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ---- Eq.(17)-(18): reparameterized policy gradient (autograd = chain rule) ----
        a, logp, _ = self.pi.sample(obs)
        q_pi = torch.min(self.q1(obs, a), self.q2(obs, a))
        pi_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        # ---- Eq.(19)-(20): entropy temperature update ----
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update()

        return {"q_loss": q_loss.item(), "pi_loss": pi_loss.item(),
                "alpha": self.alpha.item(), "alpha_loss": alpha_loss.item()}

    def save(self, path):
        torch.save({"pi": self.pi.state_dict(), "q1": self.q1.state_dict(),
                    "q2": self.q2.state_dict(), "log_alpha": self.log_alpha}, path)