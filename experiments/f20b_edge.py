"""
F20b — STATISTIK-HAERTUNG des H4-Befunds: "Edge of Chaos" ist ein
Ressourcen-Optimum, keine Kritikalitaet. (Der AGI-relevante Claim.)

F20-Lauf-1: H1 widerlegt (kein Pull zur Kante — Starts nahe 0 druecken
WEG, 1/16 Endpunkte in Kanten-Naehe), H2 inkonsistent (Schwelle
unerreichbar in 400 Steps), H4 getroffen (MC-Optimum ben -0.33 vs
Trainings-Endpunkte -0.41, Abstand 0.08 < 0.1).

Bevor dieser Claim fest ist, haertet F20b:
  (1) MEHR SEEDS: 8 statt 4 pro Startpunkt
  (2) MEHR TASKS: sinus_memory UND copy (Antwort task-abhaengig?)
  (3) LAENGERES TRAINING: 1200 Steps — konvergieren die Endpunkte
      wirklich Richtung Optimum, oder stehen sie nur zufaellig da?
  (4) FEINERES MC-GITTER: Optimum-Position genauer (fuenf ben-Punkte
      um -0.3..-0.5), Regression Endpunkt ~ Optimum
  (5) H2 NACHMESSBAR: Schwelle relativ (Loss < 0.5 * anfaenglicher
      Loss), Zeitmessung bis dahin — Selektion sauber testen
  (6) KONTROLLE: Endpunkt-Verteilung gegen Nullmodell (Permutation
      der Endpunkte) — ist die Koinzidenz signifikant?

ENTSCHEIDUNG: Wenn H4 mit 8 Seeds x 2 Tasks haelt (Abstand < 0.1,
signifikant gegen Permutation), ist der Satz fest:
  "Trainings-Endpunkte und Ressourcen-Optimum koinzidieren; die
   scheinbare Kanten-Naehe ist eine Confound-Korrelation."
=> AGI-Konsequenz: Kritikalitaet ist KEINE Notwendigkeit —
   Architekturen brauchen Ressourcen-Optima, keine Kanten-Halter.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.design import make_designed_model
from cgfn.benettin import benettin_ftle_max
from cgfn.memory_capacity import memory_capacity
from cgfn.continuous import make_continuous_batch
from cgfn.field import drive_input

LAM_TARGETS = (-0.8, -0.4, -0.2, -0.1, 0.1)


def make_net_at_lambda(lam_target, seed, H=32):
    best = None
    for lam in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
        m = make_designed_model(H, 1.0, lam, 0.0, seed, task_dim=0)
        x = drive_input(4, 48, 0.5)
        ben = benettin_ftle_max(m, x)
        if best is None or abs(ben - lam_target) < best[2]:
            best = (lam, m, abs(ben - lam_target), ben)
    return best[1], best[0], best[3]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--out", default="results/f20b_edge_results.txt")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 110)
    lines.append("F20b STATISTIK-HAERTUNG: Endpunkte vs Ressourcen-Optimum (8 Seeds x 2 Tasks x 1200 Steps)")
    lines.append("=" * 110)

    # ── (3)(4) MC-Optimum fein + Regression ──
    lines.append("")
    lines.append("--- MC-Optimum (feines Gitter, s=0, 8 Seeds) ---")
    mc_curve = []
    for lam in (0.6, 0.8, 1.0, 1.2, 1.4):
        mcs, bs = [], []
        for seed in seeds:
            m = make_designed_model(32, 1.0, lam, 0.0, seed, task_dim=0)
            x = drive_input(4, 48, 0.5)
            bs.append(benettin_ftle_max(m, x))
            torch.manual_seed(seed + 2000)
            mc, _ = memory_capacity(m, max_lag=12, seq_len=64, input_scale=0.5)
            mcs.append(mc)
        ben = sum(bs) / len(bs)
        mc = sum(mcs) / len(mcs)
        mc_curve.append((ben, mc))
        lines.append(f"lam {lam:.1f}: ben {ben:+.3f} | MC {mc:5.2f}")
    opt_ben = max(mc_curve, key=lambda t: t[1])[0]
    lines.append(f"MC-Optimum bei ben ~ {opt_ben:+.3f}")

    # ── (1)(2)(3) Training + Endpunkte pro Task ──
    opt_dist_tasks = {}
    h2_times = {}
    for task in ("sinus_memory", "copy"):
        lines.append("")
        lines.append(f"--- Task {task}: Startpunkte -> Endpunkte (1200 Steps, 8 Seeds) ---")
        lines.append(f"{'start':>7s} {'seed':>4s} {'lam0':>7s} {'lam1':>7s} {'t_half':>7s}")
        ends = []
        for lam_t in LAM_TARGETS:
            for seed in seeds:
                m, _, ben0 = make_net_at_lambda(lam_t, seed)
                opt = torch.optim.Adam(m.parameters(), lr=args.lr)
                lossfn = torch.nn.MSELoss()
                loss_init = None
                t_half = None
                for step in range(args.steps):
                    x, y = make_continuous_batch(task, 16, 16)
                    out, _ = m(x)
                    loss = lossfn(out, y)
                    opt.zero_grad(); loss.backward(); opt.step()
                    lv = float(loss.item())
                    if loss_init is None:
                        loss_init = lv
                    if t_half is None and lv < 0.5 * loss_init:
                        t_half = step + 1
                xdr = drive_input(4, 48, 0.5)
                lam1 = benettin_ftle_max(m, xdr)
                ends.append(lam1)
                h2_times.setdefault(lam_t, []).append(t_half if t_half else args.steps + 1)
                lines.append(f"{lam_t:+7.1f} {seed:4d} {ben0:+7.3f} {lam1:+7.3f} {t_half if t_half else -1:7d}")
        ends_t = torch.tensor(ends)
        dist = (ends_t - opt_ben).abs().mean().item()
        opt_dist_tasks[task] = (dist, ends_t)
        near_edge = ((ends_t > -0.15) & (ends_t < 0.15)).float().mean().item()
        lines.append(f"=> Endpunkt-Mittel {ends_t.mean():+.3f} | |Endpunkt - Optimum| {dist:.3f} | in Kanten-Naehe {100*near_edge:.0f}%")

    # ── (6) Permutations-Kontrolle fuer sinus_memory ──
    lines.append("")
    lines.append("--- (6) Signifikanz: Permutations-Test (n=2000) ---")
    dist, ends_t = opt_dist_tasks["sinus_memory"]
    all_bens = torch.tensor([b for b, _ in mc_curve] + ends_t.tolist())
    n_over = 0
    for _ in range(2000):
        perm = all_bens[torch.randperm(len(all_bens))][:len(ends_t)]
        if (perm - opt_ben).abs().mean().item() <= dist:
            n_over += 1
    p_val = n_over / 2000
    lines.append(f"beobachteter Abstand {dist:.3f} | Permutations-p {p_val:.3f} (<0.05 => Koinzidenz signifikant)")

    # ── (5) H2 sauber: t_half vs Start ──
    lines.append("")
    lines.append("--- (5) H2 SELEKTION: t(halber Loss) vs Start-lambda (sinus_memory) ---")
    for lam_t in LAM_TARGETS:
        ts = h2_times[lam_t]
        lines.append(f"start {lam_t:+.1f}: t_half mittel {sum(ts)/len(ts):6.0f} (min {min(ts)}, max {max(ts)})")

    lines.append("")
    lines.append("ENTSCHEIDUNG: H4 haelt, wenn Abstand < 0.1 bei beiden Tasks UND")
    lines.append(f"Permutations-p < 0.05. Dann ist der Satz fest: Trainings-Endpunkte")
    lines.append("fallen mit dem RESSOURCEN-Optimum zusammen — scheinbare Kanten-")
    lines.append("Naehe ist Confound. AGI-Folge: Kritikalitaet ist keine Notwendigkeit.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
