"""
F1 — Scaling-Gesetz des Sweet Spots (ROADMAP):
Verschiebt sich der kritische Sweet Spot (FTLE ~ 0) mit hidden_dim?

Design:
  - hidden_dim-Grid (default 16, 32, 64, 128)
  - tau-Grid (default 0.5, 1.0, 2.0, 4.0)
  - pro (dim, tau): seeds x tasks trainieren, FTLE + d_eff + Loss messen
  - Ausgabe: Tabelle + grobe Sweet-Spot-Lage (kleinstes |FTLE|)

Hypothesen:
  (a) größeninvariant (FTLE-Sweet-Spot ~0 unabhaengig von N)
  (b) systematischer Drift mit N  ->  would be interesting!

Selbst-contained: nutzt nur model/tasks/ftle aus src/cgfn plus eine
lokale d_eff (Participation Ratio) — keine Abhaengigkeit von experiment-
spezifischen Helpers.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.ftle import ftle_summary


def train(seed, task, hidden_dim, tau, steps=1500, lr=1e-3, seq_len=8,
          device="cpu", task_dim=None):
    if task_dim is None:
        task_dim = len(TASKS)
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim, tau=tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    for _ in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = torch.zeros(x.shape[0], task_dim, device=device)
        tv[:, idx] = 1.0
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()


@torch.no_grad()
def d_eff(states):
    """Participation Ratio der Zustands-Kovarianz (B, T, H) -> float."""
    if states.dim() == 4:
        states = states.squeeze(0)
    z = states.reshape(-1, states.shape[-1])
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return ((eig.sum() ** 2) / (eig ** 2).sum()).item()


@torch.no_grad()
def measure(model, task, task_dim, seq_len, device):
    idx = TASKS.index(task)
    x, _ = make_batch(task, batch_size=32, seq_len=seq_len, device=device)
    tv = torch.zeros(x.shape[0], task_dim, device=device)
    tv[:, idx] = 1.0
    _, states = model(x, tv)
    if states.dim() == 4:
        states = states.squeeze(0)
    ftle = ftle_summary(model, states)
    return ftle["ftle_mean"], d_eff(states)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dims", default="16,32,64,128")
    p.add_argument("--taus", default="0.5,1.0,2.0,4.0")
    p.add_argument("--tasks", default="copy,parity")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--out", default="results/scaling_results.txt")
    args = p.parse_args()

    dims = [int(v) for v in args.dims.split(",")]
    taus = [float(v) for v in args.taus.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("F1 SCALING: Verschiebt sich der Sweet Spot (FTLE~0) mit hidden_dim?")
    lines.append(f"dims={dims}  taus={taus}  tasks={tasks}  seeds={seeds}  "
                 f"steps={args.steps}  seq_len={args.seq_len}")
    lines.append("=" * 78)
    lines.append(f"{'dim':>5s} {'tau':>6s} {'ftle_mean':>10s} {'d_eff':>7s} "
                 f"{'mean_loss':>10s}")

    device = "cpu"
    task_dim = len(TASKS)
    rows = []
    for dim in dims:
        for tau in taus:
            ftles, deffs, losses = [], [], []
            for seed in seeds:
                for task in tasks:
                    m, loss = train(seed, task, hidden_dim=dim, tau=tau,
                                    steps=args.steps, seq_len=args.seq_len,
                                    device=device, task_dim=task_dim)
                    f, d = measure(m, task, task_dim, args.seq_len, device)
                    ftles.append(f); deffs.append(d); losses.append(loss)
            ftle_m = sum(ftles) / len(ftles)
            deff_m = sum(deffs) / len(deffs)
            loss_m = sum(losses) / len(losses)
            rows.append((dim, tau, ftle_m, deff_m, loss_m))
            line = (f"{dim:5d} {tau:6.2f} {ftle_m:+10.4f} {deff_m:7.2f} "
                    f"{loss_m:10.4f}")
            lines.append(line)
            print(line, flush=True)

    # Sweet-Spot-Lage pro dim: tau mit kleinstem |ftle_mean|
    lines.append("")
    lines.append("--- Sweet-Spot-Lage (kleinstes |FTLE| pro hidden_dim) ---")
    for dim in dims:
        sub = [r for r in rows if r[0] == dim]
        best = min(sub, key=lambda r: abs(r[2]))
        lines.append(f"dim={dim:4d}: tau*={best[1]:.2f}  "
                     f"ftle={best[2]:+.4f}  d_eff={best[3]:.2f}  loss={best[4]:.4f}")

    lines.append("")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
