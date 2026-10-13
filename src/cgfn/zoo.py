"""
F13 — Modell-Zoo: standard-architekturen unter unserem Messgeraet.

Zweck (ROADMAP F13): Unser F11-Wissen (Spektralregeln messen die
ungetriebene Strecke; die getriebene Kante liegt woanders) auf
BESTEHENDE Modelle anwenden: Wo irren die Design-Kriterien, die
die Praxis tatsaechlich nutzt (Echo-State rho(W)<1, Reservoir-Rho)?

Architekturen:
  - 'ct'    — unser ContinuousRNN (RK2, substeps) als Anker.
  - 'gru'   — handgeschriebene GRU-Zelle (standard-Gleichungen,
              identisch zu torch.nn.GRUCell — s. tests/test_f13.py).
  - 'cornn' — Coupled-OSCILLATOR RNN (Rusch & Mishra, NeurIPS 2021,
              diskretisierte CT-Form mit Motor-Neuronen y).

ALLE Modelle erfuellen EINEN Vertrag (das macht die Messmethodik
modell-agnostisch und damit allgemeingueltig):
    step(z, x_in) -> z_next   — einzelner Zustands-Schritt OHNE Batch
    forward(x_seq) -> (out, states)
    design() -> dict(rho_W, sv_W, kappa, design_ok)
Design-Maasse = das, was Designer an W ablesen (rho(W), sv(W));
design_ok = das klassische Kriterium (CT: tau*rho<1, diskret: rho<1).
"""
import math

import torch
import torch.nn as nn

from cgfn.model import ContinuousRNN


# ─────────────────────────────────────────────
# GRU — handgeschrieben (standard-Gleichungen)
# ─────────────────────────────────────────────

class _GruCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        b = 1.0 / math.sqrt(hidden_dim)
        self.weight_ih = nn.Parameter((torch.rand(3 * hidden_dim, input_dim, generator=g) * 2 - 1) * b)
        self.weight_hh = nn.Parameter((torch.rand(3 * hidden_dim, hidden_dim, generator=g) * 2 - 1) * b)
        self.bias = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.hidden_dim = hidden_dim

    def step(self, h, x):
        gi = self.weight_ih @ x
        gh = self.weight_hh @ h
        ri, zi, ni = gi.chunk(3)
        rh, zh, nh = gh.chunk(3)
        bi = self.bias.chunk(3)
        r = torch.sigmoid(ri + rh + bi[0])
        z = torch.sigmoid(zi + zh + bi[1])
        n = torch.tanh(ni + r * (nh + bi[2]))
        return (1.0 - z) * n + z * h


# ─────────────────────────────────────────────
# coRNN — gekoppelte Oszillator-RNN (diskretisiert)
# ─────────────────────────────────────────────

class _CoRNNCell(nn.Module):
    """coRNN-Schritt (Rusch & Mishra 2021, diskretisierte Form):

    x' = (1-dt) x + dt * tanh(Wx x + Wy y + Bx u) - dt * gamma * y
    y' = (1-gamma*dt) y + dt * tanh(Wyx x + By u)
    """

    def __init__(self, input_dim, hidden_dim, dt=0.02, gamma=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        b = 1.0 / math.sqrt(hidden_dim)
        self.Wx = nn.Parameter((torch.rand(hidden_dim, hidden_dim, generator=g) * 2 - 1) * b)
        self.Wy = nn.Parameter((torch.rand(hidden_dim, hidden_dim, generator=g) * 2 - 1) * b)
        self.Wyx = nn.Parameter((torch.rand(hidden_dim, hidden_dim, generator=g) * 2 - 1) * b)
        self.Bx = nn.Parameter((torch.rand(input_dim, hidden_dim, generator=g) * 2 - 1) * b)
        self.By = nn.Parameter((torch.rand(input_dim, hidden_dim, generator=g) * 2 - 1) * b)
        self.dt, self.gamma, self.hidden_dim = dt, gamma, hidden_dim

    def step(self, s, x):
        H = self.hidden_dim
        xs, ys = s[:H], s[H:]
        x_new = (1.0 - self.dt) * xs \
            + self.dt * torch.tanh(self.Wx @ xs + self.Wy @ ys + self.Bx.T @ x) \
            - self.dt * self.gamma * ys
        y_new = (1.0 - self.gamma * self.dt) * ys \
            + self.dt * torch.tanh(self.Wyx @ xs + self.By.T @ x)
        return torch.cat([x_new, y_new])


# ─────────────────────────────────────────────
# Einheitlicher Wrapper (der Zoo-Vertrag)
# ─────────────────────────────────────────────

class ZooModel(nn.Module):
    """Adapter: step(z,x)->z', forward(x_seq), design().

    z ist IMMER ein flacher Zustandsvektor (coRNN: cat(x,y)).
    Alle Messgeraete (driven_lyap, memory_capacity) nutzen nur
    forward/step — dadurch ist die Methodik modell-agnostisch.
    """

    def __init__(self, name, input_dim, hidden_dim, seed=0):
        super().__init__()
        self.name = name
        self.seed = seed
        torch.manual_seed(seed)
        if name == "ct":
            self.core = ContinuousRNN(input_dim=input_dim, task_dim=0,
                                      hidden_dim=hidden_dim, tau=1.0)
            self.state_dim = hidden_dim
        elif name == "gru":
            self.core = _GruCell(input_dim, hidden_dim, seed)
            self.state_dim = hidden_dim
        elif name == "cornn":
            self.core = _CoRNNCell(input_dim, hidden_dim, seed=seed)
            self.state_dim = 2 * hidden_dim
        else:
            raise ValueError(name)
        self.readout = nn.Linear(self.state_dim, input_dim)

    # ── Vertrag: Schritt ──
    def step(self, z, x_in):
        if self.name == "ct":
            m = self.core
            for _ in range(m.substeps):
                k1 = m.dynamics(z.unsqueeze(0), x_in.unsqueeze(0), None)[0]
                k2 = m.dynamics((z + 0.5 * m.dt * k1).unsqueeze(0),
                                x_in.unsqueeze(0), None)[0]
                z = z + m.dt * k2
            return z
        return self.core.step(z, x_in)

    # ── Vertrag: Rollout ──
    def forward(self, x_seq, task_vec=None):
        B, T, D = x_seq.shape
        z = torch.zeros(B, self.state_dim, device=x_seq.device)
        states = []
        for t in range(T):
            z = self.step_batch(z, x_seq[:, t])
            states.append(z)
        states = torch.stack(states, dim=1)
        return self.readout(states), states

    def step_batch(self, z, x):
        out = []
        for i in range(z.shape[0]):
            out.append(self.step(z[i], x[i]))
        return torch.stack(out, dim=0)

    # ── Vertrag: Design-Maasse (was Designer an W ablesen) ──
    def design(self):
        if self.name == "ct":
            W = self.core.W.detach()
            rho = torch.linalg.eigvals(W).abs().max().item()
            sv = torch.linalg.svdvals(W)[0].item()
            return {"rho_W": rho, "sv_W": sv, "kappa": sv / max(rho, 1e-9),
                    "design_ok": self.core.tau * rho < 1.0}
        if self.name == "gru":
            W = self.core.weight_hh.detach()  # (3H, H) — Standard-ESN-Maass
            rho = torch.linalg.eigvals(W).abs().max().item()
            sv = torch.linalg.svdvals(W)[0].item()
            return {"rho_W": rho, "sv_W": sv, "kappa": sv / max(rho, 1e-9),
                    "design_ok": rho < 1.0}
        # coRNN: linearisierte Schritt-Map am Ursprung auf (x, y)
        c = self.core
        H = c.hidden_dim
        dt, gam = c.dt, c.gamma
        I = torch.eye(H)
        Jxx = (1 - dt) * I + dt * c.Wx.detach()
        Jxy = dt * c.Wy.detach() - dt * gam * I
        Jyx = dt * c.Wyx.detach()
        Jyy = (1 - gam * dt) * I
        J = torch.cat([torch.cat([Jxx, Jxy], dim=1),
                       torch.cat([Jyx, Jyy], dim=1)], dim=0)
        rho = torch.linalg.eigvals(J).abs().max().item()
        sv = torch.linalg.svdvals(J)[0].item()
        return {"rho_W": rho, "sv_W": sv, "kappa": sv / max(rho, 1e-9),
                "design_ok": rho < 1.0}


def make_zoo_model(name, input_dim=1, hidden_dim=32, seed=0):
    return ZooModel(name, input_dim, hidden_dim, seed)
