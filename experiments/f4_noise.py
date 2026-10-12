"""
F4 — Noise-Robustheit des Sweet Spots (ROADMAP):
Ist der Criticality-Sweet-Spot auch unter Eingangs-Rauschen optimal?

Design:
  - Training auf copy/parity mit additivem Gauß-Rauschen auf den Input
    (sigma in noise-Grid), Evaluation CLEAN (kein Rauschen).
  - tau-Grid wie in F1, damit vergleichbar.
  - Messung: clean loss, FTLE_mean, d_eff.
  - Frage: driftet tau*(loss) mit sigma? Wird die chaotische Phase
    (kleines tau) robuster oder anfälliger?
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.ftle import ftle_summary
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def train_noisy(seed, task, tau, sigma, steps=1000, lr=1e-3, seq_len=8,
                device="cpu"):
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=32, tau=tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    loss = None
    for _ in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        if sigma > 0:
            x = (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)
        tv = torch.zeros(x.shape[0], len(TASKS), device=device)
        tv[:, idx] = 1.0
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def d_eff(states):
    if states.dim() == 4:
        states = states.squeeze(0)
    z = states.reshape(-1, states.shape[-1])
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return ((eig.sum() ** 2) / (eig ** 2).sum()).item()


@torch.no_grad()
def evaluate_clean(model, task, seq_len, device):
    idx = TASKS.index(task)
    x, y = make_batch(task, batch_size=32, seq_len=seq_len, device=device)
    tv = torch.zeros(x.shape[0], len(TASKS), device=device)
    tv[:, idx] = 1.0
    out, states = model(x, tv)
    clean = torch.nn.BCEWithLogitsLoss()(out, y).item()
    ftle = ftle_summary(model, states)
    return clean, ftle["ftle_mean"], d_eff(states)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", default="copy,parity")
    p.add_argument("--taus", default="0.5,1.0,2.0,4.0")
    p.add_argument("--sigmas", default="0.0,0.1,0.3")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--out", default="results/f4_noise_results.txt")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    taus = [float(v) for v in args.taus.split(",")]
    sigmas = [float(v) for v in args.sigmas.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 80)
    lines.append("F4 NOISE: Driftet der Sweet-Spot unter Eingangs-Rauschen? "
                 "(Training noisy, Eval clean)")
    lines.append(f"tasks={tasks}  taus={taus}  sigmas={sigmas}  seeds={seeds}  "
                 f"steps={args.steps}")
    lines.append("=" * 80)
    lines.append(f"{'task':>8s} {'sigma':>6s} {'tau':>6s} {'loss_clean':>11s} "
                 f"{'ftle_mean':>10s} {'d_eff':>7s}")

    rows = []
    for task in tasks:
        for sigma in sigmas:
            for tau in taus:
                losses, ftles, deffs = [], [], []
                for seed in seeds:
                    m = train_noisy(seed, task, tau, sigma,
                                    steps=args.steps, seq_len=args.seq_len)
                    loss, ftle, d = evaluate_clean(m, task, args.seq_len, "cpu")
                    losses.append(loss); ftles.append(ftle); deffs.append(d)
                lm = sum(losses) / len(losses)
                fm = sum(ftles) / len(ftles)
                dm = sum(deffs) / len(deffs)
                rows.append((task, sigma, tau, lm, fm, dm))
                line = f"{task:>8s} {sigma:6.2f} {tau:6.2f} {lm:11.4f} {fm:+10.4f} {dm:7.2f}"
                lines.append(line)
                print(line, flush=True)

    lines.append("")
    lines.append("--- Auswertung: tau*(min clean loss) pro (task, sigma) ---")
    for task in tasks:
        for sigma in sigmas:
            sub = [r for r in rows if r[0] == task and r[1] == sigma]
            best = min(sub, key=lambda r: r[3])
            lines.append(f"{task:>8s} sigma={sigma:.2f}: tau*={best[2]:.2f}  "
                         f"loss_clean={best[3]:.4f}  ftle={best[4]:+.4f}")
    lines.append("")
    lines.append("Leseart: tau* konstant ueber sigma => Sweet-Spot ist "
                 "noise-robust (FTLE-Kalibrierung uebertraegt sich); "
                 "tau* driftet => Kritikalitaet muss unter Rauschen neu "
                 "kalibriert werden.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
