"""
F5 — Kontinuierliche Tasks (ROADMAP):
Alle bislang binary toy tasks (copy/reverse/parity/mod3/delayed_xor).
Hier: kontinuierliche Sequenz-Tasks mit MSE-Loss:

  sinus_memory — y_t = x_{t-delay}: Sinus mit zufälliger Frequenz/Phase
                 muss über delay Steps im Zustand gehalten werden.
  lorenz       — next-step Vorhersage der (skalierten) x-Komponente
                 eines Lorenz-Attraktors — chaotische Dynamik.
  ou_noise     — next-step Vorhersage eines Ornstein-Uhlenbeck-Prozesses
                 — stochastische, glatte Dynamik.

Frage (F5): Bleibt der Criticality-Sweet-Spot (FTLE~0, minimaler Loss)
auch auf kontinuierlichen, teils chaotischen Aufgaben erhalten?
"""
import math

import torch

CONTINUOUS_TASKS = ["sinus_memory", "lorenz", "ou_noise"]


def _sinus_memory(batch_size, seq_len, device):
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freq = 0.25 + 1.75 * torch.rand(batch_size, 1, device=device)  # rad/step
    phase = 2 * math.pi * torch.rand(batch_size, 1, device=device)
    x = torch.sin(freq * t.unsqueeze(0) + phase)  # (B, T)
    delay = 3
    y = x.clone()
    y[:, delay:] = x[:, :-delay]  # y_t = x_{t-delay}
    return x.unsqueeze(-1), y.unsqueeze(-1)


def _lorenz(batch_size, seq_len, device, dt=0.01, stride=5, burn=500):
    """Vektorisierte Lorenz-x-Zeitreihe, next-step Vorhersage, /20 skaliert."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x = torch.empty(batch_size, 3, device=device)
    x[:, 0] = 1.0 + 0.1 * torch.randn(batch_size, device=device)
    x[:, 1] = 1.0 + 0.1 * torch.randn(batch_size, device=device)
    x[:, 2] = 20.0 + 0.5 * torch.randn(batch_size, device=device)
    out = torch.empty(seq_len + 1, batch_size, device=device)
    with torch.no_grad():
        idx = 0
        for i in range(burn + stride * seq_len + 1):
            dx = sigma * (x[:, 1] - x[:, 0])
            dy = x[:, 0] * (rho - x[:, 2]) - x[:, 1]
            dz = x[:, 0] * x[:, 1] - beta * x[:, 2]
            x = x + dt * torch.stack([dx, dy, dz], dim=1)
            if i >= burn and (i - burn) % stride == 0:
                out[idx] = x[:, 0]
                idx += 1
    out = out / 20.0  # Lorenz-x ~ [-20, 20] -> ~[-1, 1]
    xs, ys = out[:-1].T, out[1:].T  # (B, T)
    return xs.unsqueeze(-1), ys.unsqueeze(-1)


def _ou_noise(batch_size, seq_len, device, theta=0.15, sigma=0.4, dt=1.0):
    traj = torch.empty(batch_size, seq_len + 1, device=device)
    with torch.no_grad():
        cur = 0.5 * torch.randn(batch_size, device=device)
        for i in range(seq_len + 1):
            traj[:, i] = cur
            cur = cur - theta * cur * dt + sigma * math.sqrt(dt) * torch.randn(
                batch_size, device=device)
    xs, ys = traj[:, :-1], traj[:, 1:]
    return xs.unsqueeze(-1), ys.unsqueeze(-1)


def make_continuous_batch(task, batch_size=32, seq_len=32, device="cpu"):
    """(B, T, 1)-Paare für kontinuierliche Tasks — MSE-Loss verwenden."""
    if task == "sinus_memory":
        return _sinus_memory(batch_size, seq_len, device)
    if task == "lorenz":
        return _lorenz(batch_size, seq_len, device)
    if task == "ou_noise":
        return _ou_noise(batch_size, seq_len, device)
    raise ValueError(task)
