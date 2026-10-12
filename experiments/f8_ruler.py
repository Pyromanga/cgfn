"""
F8 — Ruler-Anwendung: Wo funktioniert der Spectral Ruler wirklich?

Checkpoint-Sweep ueber das Training: alle K Steps wird spectral_report()
gezogen — wir sehen, wie sich lambda_max(W), sv_max, kappa, die
Vorhersagen tau*_pred und die Regel-Guete (vs. gemessenes FTLE-Vorzeichen)
WAEHREND des Lernens entwickeln.

Zwei Diagnose-Fragen:
  (1) Regel-Guete: Welche Chaos-Regel (MF/Eig/SV) trifft das gemessene
      FTLE-Vorzeichen zu jedem Zeitpunkt? (Wirkungsbereich kartieren)
  (2) Kritikalitaets-Drift: Wandert lambda_max(W) im Training gegen
      1/tau (d.h. trainiert das Netz von selbst Richtung Regelrand)?
      => waere die staerkste Form des "Training findet Kritikalitaet".

Tasks: copy (binary, BCE) und sinus_memory (kontinuierlich, MSE) —
der Spektral-Range wird einmal pro Task mit anderem g-Skalierungsfaktor
variiert, um auch sub-/superkritische Zustaende im Sweep zu haben.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.continuous import make_continuous_batch
from cgfn.ftle import ftle_summary
from cgfn.model import ContinuousRNN
from cgfn.ruler import spectral_report
from cgfn.tasks import TASKS, make_batch


def make_model(seed, tau, g_scale, hidden_dim, task):
    torch.manual_seed(seed)
    task_dim = len(TASKS) if task == "copy" else 0
    m = ContinuousRNN(input_dim=1, task_dim=task_dim, hidden_dim=hidden_dim, tau=tau)
    with torch.no_grad():
        m.W.copy_(m.W * g_scale)
    return m


def train_block(model, task, steps, lr=2e-3):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    if task == "copy":
        lossfn = torch.nn.BCEWithLogitsLoss()
        idx = TASKS.index("copy")
        for _ in range(steps):
            x, y = make_batch("copy", seq_len=8)
            tv = torch.zeros(x.shape[0], len(TASKS))
            tv[:, idx] = 1.0
            out, _ = model(x, tv)
            loss = lossfn(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
    else:
        lossfn = torch.nn.MSELoss()
        for _ in range(steps):
            x, y = make_continuous_batch("sinus_memory", 16, 16)
            out, _ = model(x)
            loss = lossfn(out, y)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def rollout(model, task, batch=16, seq_len=16):
    if task == "copy":
        idx = TASKS.index("copy")
        x, _ = make_batch("copy", batch_size=batch, seq_len=seq_len)
        tv = torch.zeros(batch, len(TASKS))
        tv[:, idx] = 1.0
        _, states = model(x, tv)
    else:
        x, _ = make_continuous_batch("sinus_memory", batch, seq_len)
        _, states = model(x)
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", default="copy,sinus_memory")
    p.add_argument("--g-scales", default="0.8,1.6")
    p.add_argument("--taus", default="1.0")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--total-steps", type=int, default=600)
    p.add_argument("--ckpt-every", type=int, default=150)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--out", default="results/f8_ruler_results.txt")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    g_scales = [float(v) for v in args.g_scales.split(",")]
    taus = [float(v) for v in args.taus.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 112)
    lines.append("F8 RULER-ANWENDUNG: Chaos-Regeln ueber das Training verfolgt — wo gilt der Spectral Ruler?")
    lines.append(f"tasks={tasks}  g_scales={g_scales}  taus={taus}  seeds={seeds}  "
                 f"steps={args.total_steps}  ckpt={args.ckpt_every}  hidden={args.hidden_dim}")
    lines.append("=" * 112)
    lines.append(f"{'task':>12s} {'g_sc':>5s} {'tau':>5s} {'step':>5s} {'lam_max':>8s} "
                 f"{'sv_max':>7s} {'kappa':>6s} {'Q':>6s} {'tau*eig':>8s} {'tau_pred':>8s} "
                 f"{'ftle_max':>9s} {'pMF':>5s} {'pEig':>5s} {'pSV':>4s} {'hit_eig':>7s}")

    hits = {"mf": [0, 0], "eig": [0, 0], "sv": [0, 0]}
    drift = []  # (task, g, seed, lam_first, lam_last, tau)

    for task in tasks:
        for g_scale in g_scales:
            for tau in taus:
                for seed in seeds:
                    model = make_model(seed, tau, g_scale, args.hidden_dim, task)
                    lam_first = None
                    for step in range(0, args.total_steps + 1, args.ckpt_every):
                        if step > 0:
                            train_block(model, task, args.ckpt_every)
                        states = rollout(model, task)
                        rep = spectral_report(model, states)
                        ftle = ftle_summary(model, states)["ftle_max"]
                        meas = ftle > 0.0
                        for rule, pred in (("mf", rep["pred_mf"]),
                                          ("eig", rep["pred_eig"]),
                                          ("sv", rep["pred_sv"])):
                            hits[rule][1] += 1
                            if pred == meas:
                                hits[rule][0] += 1
                        if lam_first is None:
                            lam_first = rep["lam_max"]
                        line = (f"{task:>12s} {g_scale:5.2f} {tau:5.2f} {step:5d} "
                                f"{rep['lam_max']:8.3f} {rep['sv_max']:7.3f} "
                                f"{rep['kappa']:6.2f} {rep['Q']:6.3f} "
                                f"{tau * rep['lam_max']:8.3f} {rep['tau_pred_eig']:8.3f} "
                                f"{ftle:+9.4f} {str(rep['pred_mf']):>5s} "
                                f"{str(rep['pred_eig']):>5s} {str(rep['pred_sv']):>4s} "
                                f"{'Y' if rep['pred_eig'] == meas else 'N':>7s}")
                        lines.append(line)
                        print(line, flush=True)
                    drift.append((task, g_scale, seed, lam_first, rep["lam_max"], tau))

    lines.append("")
    lines.append("--- Regel-Guete über ALLE Checkpoints (vs. FTLE-Vorzeichen) ---")
    for rule, (h, t) in hits.items():
        lines.append(f"Regel {rule:>4s}: {h}/{t} = {h / max(1, t):.1%}")

    lines.append("")
    lines.append("--- Kritikalitaets-Drift: lambda_max(W) vor vs. nach Training ---")
    toward = 0
    for task, g, seed, lf, ll, tau in drift:
        target = 1.0 / tau
        moved = (abs(ll - target) < abs(lf - target))
        toward += moved
        lines.append(f"{task:>12s} g={g:.1f} seed={seed}: lam {lf:+.3f} -> {ll:+.3f} "
                     f"(Regelrand 1/tau = {target:.3f})  Richtung-Rand: {'JA' if moved else 'nein'}")
    lines.append(f"Netto: {toward}/{len(drift)} Nets wandern Richtung Regelrand "
                 f"(=> Training treibt Richtung Kritikalitaet).")
    lines.append("")
    lines.append("LESEAT: hit_eig-Spalte ~100% und hoher pEig-Anteil => Eigenwert-Regel")
    lines.append("ist der tragfaechste Ruler; Abweichungen konzentrieren sich dort,")
    lines.append("wo pred_eig=False aber pSV=True (transiente Nische).")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
