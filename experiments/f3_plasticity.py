"""
F3 — Plasticity-Loss am CT-RNN (ROADMAP F3, Wang et al. NeurIPS 2025).

These: Plasticity-Loss = Lyapunov-KONTRAKTION. Sequenzielle Tasks machen das
Netz traeg, weil FTLE waehrend des Trainings sinkt; der Loss-Einbruch im
spaeten Task korreliert mit der FTLE-Drift.

Design:
  - Task-Sequenz (default copy -> parity -> mod3 -> delayed_xor)
  - pro Phase: steps trainieren, dann Messung:
      FTLE, dormant_fraction, weight_churn (seit Phasenstart), w_norm
      + Eval-Loss auf ALLEN bisherigen Tasks (Retention-Kurve)
  - Korrelation: FTLE-Drift (Phase 1 vs. aktuelle Phase) gegen
    Loss-Einbruch auf Task 1 (Retention-Drift). Erwartung: positiv.

Zusaetzlicher Gegentest (Paper-Radar 2610.01583, Neuroevolution-These):
  --frozen-arm schaltet lr=0: Churn muss ~0, FTLE-Drift muss ~0 sein —
  Kontrolle, dass die Metriken wirklich gradientengetrieben messen.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.plasticity import (correlate, dormant_fraction, plasticity_snapshot,
                             weight_churn)


def evaluate(model, task, task_dim, seq_len, device, batches=8):
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(task, batch_size=32, seq_len=seq_len, device=device)
            tv = torch.zeros(x.shape[0], task_dim, device=device)
            tv[:, idx] = 1.0
            out, _ = model(x, tv)
            total += lossfn(out, y).item()
    return total / batches


@torch.no_grad()
def probe_states(model, task, task_dim, seq_len, device):
    idx = TASKS.index(task)
    x, _ = make_batch(task, batch_size=32, seq_len=seq_len, device=device)
    tv = torch.zeros(x.shape[0], task_dim, device=device)
    tv[:, idx] = 1.0
    _, states = model(x, tv)
    return states


def train_phase(model, task, steps, lr, seq_len, device):
    task_dim = len(TASKS)
    idx = TASKS.index(task)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        x, y = make_batch(task, batch_size=32, seq_len=seq_len, device=device)
        tv = torch.zeros(x.shape[0], task_dim, device=device)
        tv[:, idx] = 1.0
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def run_sequence(seed, tasks, steps, lr, seq_len, device, frozen=False):
    task_dim = len(TASKS)
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=32, tau=1.0)
    phase_lr = 0.0 if frozen else lr
    snapshots = []
    for phase, task in enumerate(tasks):
        w_before = type(model)(task_dim=task_dim, hidden_dim=32, tau=1.0)
        w_before.load_state_dict(model.state_dict())
        train_loss = train_phase(model, task, steps, phase_lr, seq_len, device)
        states = probe_states(model, task, task_dim, seq_len, device)
        snap = plasticity_snapshot(model, states)
        snap.update({
            "phase": phase, "task": task,
            "churn": weight_churn(w_before, model),
            "train_loss": train_loss,
            "retention": {t: evaluate(model, t, task_dim, seq_len, device)
                          for t in tasks[:phase + 1]},
        })
        snapshots.append(snap)
    return snapshots


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--tasks", default="copy,parity,mod3,delayed_xor")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--frozen-arm", action="store_true",
                   help="zusaetzlich lr=0-Arm als Gegentest (Churn/FTLE-Drift ~0)")
    p.add_argument("--out", default="results/f3_plasticity_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    device = "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 78)
    lines.append("F3 PLASTICITY: Sequenzielle Tasks — FTLE-Kontraktion vs. Retention")
    lines.append(f"tasks={tasks}  seeds={seeds}  steps={args.steps}  lr={args.lr}")
    lines.append("=" * 78)

    all_runs = []
    for seed in seeds:
        snaps = run_sequence(seed, tasks, args.steps, args.lr, args.seq_len, device)
        all_runs.append(snaps)
        lines.append(f"\n--- seed {seed} ---")
        lines.append(f"{'phase':>5s} {'task':>12s} {'ftle':>7s} {'dorm':>6s} "
                     f"{'churn':>7s} {'loss':>7s} {'ret(task1)':>10s}")
        for s in snaps:
            first_task = tasks[0]
            ret1 = s["retention"].get(first_task, float("nan"))
            lines.append(f"{s['phase']:5d} {s['task']:>12s} {s['ftle_mean']:+7.3f} "
                         f"{s['dormant_fraction']:6.3f} {s['churn']:7.4f} "
                         f"{s['train_loss']:7.4f} {ret1:10.4f}")

    # Korrelation ueber alle (seed, phase>=1): FTLE-Drift vs. Retention-Drift
    ftle_drifts, ret_drifts = [], []
    for snaps in all_runs:
        base_ftle = snaps[0]["ftle_mean"]
        base_ret = snaps[0]["retention"][tasks[0]]
        for s in snaps[1:]:
            ftle_drifts.append(s["ftle_mean"] - base_ftle)
            ret_drifts.append(s["retention"][tasks[0]] - base_ret)
    corr = correlate(ftle_drifts, ret_drifts)
    lines.append("")
    lines.append(f"corr(FTLE-Drift, Retention-Drift auf Task 1) = {corr:+.4f}")
    lines.append("These: positiv — staerkere Lyapunov-Kontraktion (FTLE sinkt)")
    lines.append("=> groesserer Loss-Einbruch auf dem fruehen Task.")

    if args.frozen_arm:
        lines.append("\n--- Gegentest: lr=0-Arm (gradientenfrei) ---")
        for seed in seeds[:1]:
            snaps = run_sequence(seed, tasks, args.steps, args.lr, args.seq_len,
                                 device, frozen=True)
            for s in snaps:
                lines.append(f"phase {s['phase']} ({s['task']}): "
                             f"churn={s['churn']:.6f}  ftle={s['ftle_mean']:+.4f}  "
                             f"dorm={s['dormant_fraction']:.3f}")
        lines.append("Erwartung: churn≈0, FTLE konstant -> Metriken messen")
        lines.append("gradientengetriebene Plastizitaets-Veraenderung.")

    lines.append("")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
