"""
F14 — Fix it: Training nach dem F11/F12-Rezept vs. freies Training.

Fragen (ROADMAP F14):
  (1) Was macht FREIES SGD-Training mit kappa und dem getriebenen
      Kanten-Abstand? (F12-Vermutung: Training kann kappa in den
      ressourcen-zerstoerenden Bereich treiben — "das Issue der Praxis".)
  (2) Haelt der gradientenbasierte kappa-Cap (src/cgfn/kcap.py) das?
      Was kostet er an Task-Loss?
  (3) Welcher Arm gewinnt auf (Loss, Retention, MC, Kanten-Abstand)?
      => Verhaltens-Differenz "richtig gemacht" vs. "wie ueblich".

Design (end-to-end, W trainierbar — KEIN Reservoir-Setting mehr):
  Arme:
    free : plain Adam (Kontrolle)
    kcap : + kappa_weight * kappa_cap_penalty(W, kappa_target)
    pin  : FtleRegulator mit target=pin_target (default -0.2:
           subkritisch NAHE der getriebenen Kante — F12-Lektion,
           bewusst NICHT 0.0)
    both : kcap + pin
  Phase A: 'copy' trainieren; Phase B: 'parity' weitertrainieren
  (continual). Pro Arm/Seed: LossA/RetentA/LossB + MC (Jaeger) +
  Benettin-FTLE (getrieben) + SVD-FTLE + kappa-Anfang/Ende + RK2-Audit.
  task_dim=0 (sequenzielles Training, keine Task-Vektoren noetig) —
  damit memory_capacity/benettin direkt anwendbar sind.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.ftle import ftle_summary
from cgfn.regulator import FtleRegulator
from cgfn.kcap import kappa_of, kappa_cap_penalty
from cgfn.memory_capacity import memory_capacity
from cgfn.benettin import benettin_ftle_max, stability_audit

ARMS = ("free", "kcap", "pin", "both")


def evaluate(model, task, seq_len, batches=8):
    lossfn = torch.nn.BCEWithLogitsLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(task, batch_size=32, seq_len=seq_len, device="cpu")
            out, _ = model(x, None)
            total += lossfn(out, y).item()
    return total / batches


def train_phase(model, task, steps, lr, seq_len, arm, kappa_target,
                kappa_weight, regulator, track, eval_every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    loss_val = float("nan")
    for step in range(steps):
        x, y = make_batch(task, batch_size=32, seq_len=seq_len, device="cpu")
        out, states = model(x, None)
        loss = lossfn(out, y)
        if arm in ("kcap", "both"):
            loss = loss + kappa_weight * kappa_cap_penalty(model, kappa_target)
        opt.zero_grad(); loss.backward(); opt.step()
        if regulator is not None:
            regulator.step(step, model, states)
        loss_val = loss.item()
        if (step + 1) % eval_every == 0:
            track.append(kappa_of(model))
    return loss_val


def run_arm(arm, seed, args):
    torch.manual_seed(seed)
    model = ContinuousRNN(input_dim=1, task_dim=0,
                          hidden_dim=args.hidden_dim, tau=args.tau_init)
    regulator = None
    if arm in ("pin", "both"):
        regulator = FtleRegulator(target=args.pin_target, gain=args.gain,
                                  interval=args.interval)
    track = [kappa_of(model)]  # kappa vor dem Training

    train_phase(model, args.task_a, args.steps, args.lr, args.seq_len,
                arm, args.kappa_target, args.kappa_weight, regulator, track)
    loss_a_after = evaluate(model, args.task_a, args.seq_len)
    train_phase(model, args.task_b, args.steps, args.lr, args.seq_len,
                arm, args.kappa_target, args.kappa_weight, regulator, track)
    loss_a_retention = evaluate(model, args.task_a, args.seq_len)
    loss_b_final = evaluate(model, args.task_b, args.seq_len)

    # Ressourcen-/Dynamik-Messung am trainierten Netz
    mc, _ = memory_capacity(model, max_lag=args.max_lag, input_scale=0.5)
    x = torch.randn(4, 48, 1) * 0.5
    _, states = model(x, None)
    ftle_svd = float(ftle_summary(model, states)["ftle_max"])
    ftle_ben = benettin_ftle_max(model, x)
    audit = stability_audit(model)
    return {
        "arm": arm, "seed": seed,
        "loss_a": loss_a_after, "loss_b": loss_b_final,
        "retent_a": loss_a_retention,
        "mc": mc, "ben": ftle_ben, "svd": ftle_svd,
        "kappa0": track[0], "kappa1": track[-1],
        "kappa_track": track,
        "tau": float(model.tau), "rk2ok": audit["rk2_ok"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--task-a", default="copy")
    p.add_argument("--task-b", default="parity")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--tau-init", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=16)
    p.add_argument("--kappa-target", type=float, default=4.0)
    p.add_argument("--kappa-weight", type=float, default=0.02)
    p.add_argument("--pin-target", type=float, default=-0.2)
    p.add_argument("--gain", type=float, default=0.8)
    p.add_argument("--interval", type=int, default=100)
    p.add_argument("--out", default="results/f14_fix_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 120)
    lines.append("F14 FIX-IT: Training nach Rezept (kcap/pin/both) vs. freies Training — end-to-end, W trainierbar")
    lines.append(f"arms={list(ARMS)} seeds={seeds} steps={args.steps} lr={args.lr} tau_init={args.tau_init} H={args.hidden_dim}")
    lines.append(f"kappa_target={args.kappa_target} kappa_weight={args.kappa_weight} pin_target={args.pin_target} (F12-Lektion: subkritisch, nicht 0)")
    lines.append("MC nach Jaeger (Ridge, disjunkt); Benettin getrieben (Standard-Messgeraet); kappa = sv_max/|lam_max|")
    lines.append("=" * 120)
    lines.append(f"{'arm':>6s} {'seed':>4s} {'lossA':>7s} {'lossB':>7s} {'retentA':>8s} "
                 f"{'MC':>6s} {'ben':>8s} {'svd':>7s} {'kap0':>6s} {'kap1':>6s} {'tau':>5s} {'rk2ok':>5s}")

    results = []
    for seed in seeds:
        for arm in ARMS:
            r = run_arm(arm, seed, args)
            results.append(r)
            line = (f"{r['arm']:>6s} {r['seed']:4d} {r['loss_a']:7.4f} {r['loss_b']:7.4f} "
                    f"{r['retent_a']:8.4f} {r['mc']:6.2f} {r['ben']:+8.3f} {r['svd']:+7.3f} "
                    f"{r['kappa0']:6.2f} {r['kappa1']:6.2f} {r['tau']:5.2f} {str(r['rk2ok']):>5s}")
            lines.append(line)
            print(line, flush=True)

    # --- Aggregat (Mittel ueber Seeds) ---
    lines.append("")
    lines.append("--- Aggregat (Mittel pro Arm) ---")
    lines.append(f"{'arm':>6s} {'lossA':>8s} {'lossB':>8s} {'retentA':>8s} "
                 f"{'MC':>6s} {'ben':>8s} {'kap0':>6s} {'kap1':>6s}")
    agg = {}
    for arm in ARMS:
        rs = [r for r in results if r["arm"] == arm]
        def mean(key):
            vals = [r[key] for r in rs]
            return sum(vals) / len(vals) if vals else float("nan")
        agg[arm] = {k: mean(k) for k in ("loss_a", "loss_b", "retent_a", "mc", "ben", "kappa0", "kappa1")}
        a = agg[arm]
        lines.append(f"{arm:>6s} {a['loss_a']:8.4f} {a['loss_b']:8.4f} {a['retent_a']:8.4f} "
                     f"{a['mc']:6.2f} {a['ben']:+8.3f} {a['kappa0']:6.2f} {a['kappa1']:6.2f}")

    # --- (1) Was macht freies Training mit kappa? ---
    lines.append("")
    lines.append("--- (1) kappa-Drift unter freiem Training (F12-These: SGD treibt kappa nach oben?) ---")
    a = agg["free"]
    drift = a["kappa1"] - a["kappa0"]
    lines.append(f"free: kappa {a['kappa0']:.2f} -> {a['kappa1']:.2f} (Delta {drift:+.2f})")
    lines.append(f"kcap: kappa {agg['kcap']['kappa0']:.2f} -> {agg['kcap']['kappa1']:.2f} (Cap greift: {agg['kcap']['kappa1'] <= args.kappa_target + 1.0})")

    # --- (2) Was kostet der Cap? (3) Wer gewinnt? ---
    lines.append("")
    lines.append("--- (2/3) Rezept vs. Kontrolle ---")
    for arm in ARMS:
        if arm == "free":
            continue
        dl = agg[arm]["loss_a"] - agg["free"]["loss_a"]
        dm = agg[arm]["mc"] - agg["free"]["mc"]
        db = agg[arm]["ben"] - agg["free"]["ben"]
        lines.append(f"{arm:>6s} vs free: dlossA={dl:+.4f}  dMC={dm:+.2f}  dBenettin={db:+.3f}")
    best = min(ARMS, key=lambda arm: agg[arm]["loss_a"])
    lines.append("")
    lines.append(f"Bestes lossA: {best};  bestes MC: {max(ARMS, key=lambda arm: agg[arm]['mc'])}")
    lines.append("")
    lines.append("Lesart: (1) kap1(free) >> kap0(free) => freies Training treibt kappa in den")
    lines.append("schaedlichen Bereich — das Issue ist TRAININGSVERHALTEN, kein Initialisierungs-")
    lines.append("fehler. (2) kcap haelt kappa unter dem Ziel ohne lossA-Schaden => Cap ist ein")
    lines.append("kostenguenstiges Bau-Rezept. (3) pin/both mit MC- oder Retention-Gewinn =>")
    lines.append("Kanten-Pinning subkritisch (nicht auf 0!) ist die richtige Trainings-Zielvorgabe.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
