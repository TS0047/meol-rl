"""
The one set of frozen weights published with HHMARL 3D (IDSIA-papers/HHMARL_AirCombat):
env/base_actor/baseline_actor.pt, the pretrained JSBSim F-16 autopilot from the LAG project
(Light Aircraft Game) that HHMARL flies as its scripted opponent in curriculum levels 1-2.

It is not a dogfighting policy. Given (altitude error, heading error, speed error) plus its own
attitude and body velocities, it outputs aileron / elevator / rudder / throttle -- the same raw
JSBSim commands this repo's environment uses. HHMARL's TrackModel turns it into a pursuer by
aiming those errors at the opponent; LAGPursuitPilot does the same here, so it can fly either seat
of the demo against our pilots.

The weights are third-party and the HHMARL repo has no licence file, so they are not committed:

  python demo/lag_baseline.py --fetch      # download, pinned to a commit and checked by SHA-256

They are read with torch.load(weights_only=True): tensors only, no code from the file runs. The
network below re-creates HHMARL's BaselineActor layer for layer (same parameter names).
"""

import argparse
import hashlib
import os
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(RL, "third_party", "hhmarl_lag_baseline", "baseline_actor.pt")
COMMIT = "b300fab22ca6b30312c5e59029841434ae9dec04"
URL = f"https://raw.githubusercontent.com/IDSIA-papers/HHMARL_AirCombat/{COMMIT}/env/base_actor/baseline_actor.pt"
SHA256 = "54e9c996d981aa1fa82c51664a1758afd04fee0b5539dd6ccb128795149dcf69"
DECISION_EVERY = 10          # 50 Hz sim steps per decision: 5 Hz, the rate LAG trained it at
M_PER_FT = 0.3048


def available():
    return os.path.exists(WEIGHTS)


def fetch():
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    data = urllib.request.urlopen(URL, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"SHA-256 mismatch for {URL}: got {digest}, expected {SHA256}")
    with open(WEIGHTS, "wb") as f:
        f.write(data)
    return WEIGHTS


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.LayerNorm(b)]
    return nn.Sequential(*layers)


class _Head(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.logits_net = nn.Linear(n_in, n_out)


class BaselineActor(nn.Module):
    """MLP 12-128-128 (ReLU + LayerNorm) -> GRU 128 -> LayerNorm -> argmax of 4 heads [41, 41, 41, 30]."""

    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.mlp = nn.Module()
        self.base.mlp.fc = _mlp([12, 128, 128])
        self.rnn = nn.Module()
        self.rnn.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1)
        self.rnn.norm = nn.LayerNorm(128)
        self.act = nn.Module()
        self.act.action_outs = nn.ModuleList([_Head(128, n) for n in (41, 41, 41, 30)])

    def forward(self, obs, h):
        """obs (1, 12), h (1, 1, 128) -> bin indices (4,), new h."""
        x = self.base.mlp.fc(obs)
        x, h = self.rnn.gru(x.unsqueeze(0), h.transpose(0, 1).contiguous())
        x = self.rnn.norm(x.squeeze(0))
        idx = torch.cat([head.logits_net(x).argmax(dim=-1, keepdim=True) for head in self.act.action_outs], dim=-1)
        return idx.squeeze(0), h.transpose(0, 1)


def load_actor(path=WEIGHTS):
    net = BaselineActor()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LAGPursuitPilot:
    """HHMARL TrackModel's pursuit: steer the heading error to the line of sight, match the
    opponent's altitude and forward speed. Needs .fdm / .other_fdm (its own and the opponent's
    JSBSim FDM), set by the demo engine. Returns [aileron, rudder, elevator, throttle 0..1]."""

    def __init__(self, stochastic=False):
        self.net = load_actor()          # deterministic argmax, as in HHMARL; stochastic is ignored
        self.fdm = self.other_fdm = None

    def reset(self, seed):
        self.k, self.action = 0, None
        self.h = torch.zeros(1, 1, 128)

    def _obs(self, me, other):
        f, o = self.fdm, self.other_fdm
        # positions/velocities in North-East order, as HHMARL's get_position(): the sign of the
        # heading error (positive = target to the right) depends on it
        dn, de = other["pos"][1] - me["pos"][1], other["pos"][0] - me["pos"][0]
        vn, ve = me["v"][1], me["v"][0]
        R, v = np.hypot(dn, de), np.hypot(vn, ve)
        ao = np.arccos(np.clip((dn * vn + de * ve) / (R * v + 1e-8), -1, 1))
        heading_err = ao * np.sign(vn * de - ve * dn)
        u = f["velocities/u-fps"] * M_PER_FT
        obs = np.array([
            (other["pos"][2] - me["pos"][2]) / 1000.0,                       # altitude error
            _wrap(heading_err),                                             # heading error
            (o["velocities/u-fps"] * M_PER_FT - u) / 340.0,                 # speed error
            f["position/h-sl-ft"] * M_PER_FT / 5000.0,
            np.sin(f["attitude/phi-rad"]), np.cos(f["attitude/phi-rad"]),
            np.sin(f["attitude/theta-rad"]), np.cos(f["attitude/theta-rad"]),
            u / 340.0, f["velocities/v-fps"] * M_PER_FT / 340.0, f["velocities/w-fps"] * M_PER_FT / 340.0,
            f["velocities/vc-fps"] * M_PER_FT / 340.0,
        ], dtype=np.float32)
        return torch.as_tensor(obs).unsqueeze(0)

    def act(self, env, me, other):
        if self.k % DECISION_EVERY == 0:
            with torch.no_grad():
                idx, self.h = self.net(self._obs(me, other), self.h)
            a, e, r, t = (int(x) for x in idx)
            # LAG's bins: aileron/elevator/rudder 41 over [-1, 1], throttle 30 over [0.4, 0.9]
            self.action = np.array([a / 20.0 - 1.0, r / 20.0 - 1.0, e / 20.0 - 1.0, 0.4 + t * 0.5 / 29.0],
                                   dtype=np.float32)
        self.k += 1
        return self.action


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fetch", action="store_true", help="download the weights (pinned commit, SHA-256 checked)")
    a = p.parse_args()
    if a.fetch:
        print("saved", fetch())
    elif available():
        net = load_actor()
        print(f"{WEIGHTS}: {sum(x.numel() for x in net.parameters())} parameters, loads cleanly")
    else:
        sys.exit("weights missing: python demo/lag_baseline.py --fetch")
