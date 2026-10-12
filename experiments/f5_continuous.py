"""
F5 — Sweet-Spot auf kontinuierlichen Tasks (ROADMAP):
Bleibt der Criticality-Sweet-Spot (FTLE~0 / minimaler Loss) erhalten,
wenn wir von binary toy tasks auf kontinuierliche, teils chaotische
Sequenzen wechseln (sinus_memory, lorenz, ou_noise — cgfn.continuous)?

Design:
  - tau-Grid, seeds, MSE-Training (kein BCE — kontinuierliche Ziele!)
  - pro (task, tau): Loss (clean eval) + FTLE_mean + d_eff
  - Auswertung: tau*(loss) vs tau*(|FTLE|) pro Task — driftet der
    Sweet-Spot weg vom binary-Befund (tau≈1..2, kritisch ab tau≈1)?
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.continuous import CONTINUOUS_TASKS, make_continuous_batch
from cgfn.ftle import ftle_summary
from cgfn.model import ContinuousRNN


def train_continuous(seed, task, tau, steps=800, lr=2e-3, batch_size=32,
                     seq_len=32, hidden_dim=32, device="cpu"):
    torch.manual_seed(seed)
    model = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=hidden_dim, tau=tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.MSELoss()
    loss = None
    for _ in range(steps):
        x, y = make_continuous_batch(task, batch_size, seq_len, device)
        out, _ = model(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, (loss.item() if loss is not None else float("nan"))


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
def evaluate(model, task, seq_len, device):
    x, y = make_continuous_batch(task, batch_size=32, seq_len=seq_len, device=device)
    out, states = model(x)
    clean = torch.nn.functional.mse_loss(out, y).item()
    ftle = ftle_summary(model, states)
    return clean, ftle["ftle_mean"], d_eff(states)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", default="sinus_memory,lorenz,ou_noise")
    p.add_argument("--taus", default="0.5,1.0,2.0,4.0")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--out", default="results/f5_continuous_results.txt")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    for t in tasks:
        assert t in CONTINUOUS_TASKS, f"unbekannter kontinuierlicher Task: {t}"
    taus = [float(v) for v in args.taus.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("F5 KONTINUIERLICH: Bleibt der Sweet-Spot auf sinus/lorenz/OU erhalten?")
    lines.append(f"tasks={tasks}  taus={taus}  seeds={seeds}  steps={args.steps}  "
                 f"seq_len={args.seq_len}  hidden={args.hidden_dim}  loss=MSE")
    lines.append("=" * 78)
    lines.append(f"{'task':>13s} {'tau':>6s} {'loss_eval':>10s} {'ftle_mean':>10s} {'d_eff':>7s}")

    rows = []
    for task in tasks:
        for tau in taus:
            losses, ftles, deffs = [], [], []
            for seed in seeds:
                m, _ = train_continuous(seed, task, tau, steps=args.steps,
                                        seq_len=args.seq_len,
                                        hidden_dim=args.hidden_dim)
                loss, ftle, d = evaluate(m, task, args.seq_len, "cpu")
                losses.append(loss); ftles.append(ftle); deffs.append(d)
            lm = sum(losses) / len(losses)
            fm = sum(ftles) / len(ftles)
            dm = sum(deffs) / len(deffs)
            rows.append((task, tau, lm, fm, dm))
            line = f"{task:>13s} {tau:6.2f} {lm:10.4f} {fm:+10.4f} {dm:7.2f}"
            lines.append(line)
            print(line, flush=True)

    lines.append("")
    lines.append("--- Auswertung: tau*(min loss) vs tau*(min |FTLE|) pro Task ---")
    for task in tasks:
        sub = [r for r in rows if r[0] == task]
        by_loss = min(sub, key=lambda r: r[2])
        by_ftle = min(sub, key=lambda r: abs(r[3]))
        lines.append(
            f"{task:>13s}: tau*(loss)={by_loss[1]:.2f} (loss={by_loss[2]:.4f})   "
            f"tau*(|FTLE|)={by_ftle[1]:.2f} (ftle={by_ftle[3]:+.4f})")
    lines.append("")
    lines.append("Referenz binary-Tasks: Sweet-Spot im mittleren tau-Bereich, "
                 "kritisch ab tau~1 (sweetspot_results.txt).")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
