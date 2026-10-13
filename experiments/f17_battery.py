"""
F17 — FRONTIER-BATTERIE: vier spekulative Durchbruch-Kandidaten in einem
Lauf, jeder billig, jeder darf scheitern (hoch-entropische Suchstrategie).

HYPOTHESE A — Das geschlossene Gesetz, Runde 2 (F10-Einloesung):
    F10 scheiterte mit SKALARER Saettigungs-Korrektur (d=<sech^2> als Zahl).
    Der richtige Ansatz: SELBstkonsistente FELD-Iteration — iteriere die
    Vektorfeld-Korrektur d_i = <sech^2(z_i)> einzeln pro Unit, bis die
    effektive Jacobi J = -I/tau + W diag(d) stationaer ist, und sage den
    GETRIEBENEN Benettin-FTLE damit voraus (kein Rollout!). Wenn die
    Feld-Iteration die getriebene Kante trifft, haben wir das geschlossene
    Design-Gesetz: Kante aus W + Treib-Statistik allein.
    Test: designed Netze (kappa-Sweep), Vorhersage vs. Benettin gemessen.

HYPOTHESE B — Warum laeuft Training zur Kante? (Gradienten-Mechanismus)
    Wenn der Task-Loss-Gradient systematisch die kappa/Rand-Richtung
    drueckt, die Richtung Kante erzeugt, ist "Training -> Kante" kein
    Unfall sondern Gradienten-Dynamik. Test: projiziere den letzten
    SGD-Schritt auf die Richtung "spektraler Rand hoch" und messe die
    Kovarianz mit dem Kanten-Abstand-Drift ueber das Training.

HYPOTHESE C — kappa*(Horizont)-Gesetz (Design-Dial):
    F12: MC-Optimum bei moderatem kappa, nicht an der Kante. Wenn das
    optimale kappa systematisch mit der Task-Gedaechtnis-Horizontlaenge
    waechst (laengerer Horizont -> mehr nicht-normale Struktur nuetzlich),
    haben wir eine Bau-Anweisung kappa*(L). Test: designed Netze, MC bei
    max_lag in {4,8,16,32} x kappa in {1,2,4,8,16} — liegt kappa*(L)
    auf einer Kurve?

HYPOTHESE D — Kante(W, sigma): haengt die getriebene Benettin-Kante
    systematisch von der Treib-Skala ab? Ein Gesetz edge(sigma) waere die
    Kalibrierungs-Grundlage fuer alle Kanten-Regler. Test: ein Netz,
    sigma-Sweep, Benettin-Kurve — linear in log sigma? Sattigung?
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.design import make_designed_model
from cgfn.cavity import eig_max, spectral_edge
from cgfn.benettin import benettin_ftle_max
from cgfn.memory_capacity import memory_capacity
from cgfn.model import ContinuousRNN
from cgfn.continuous import make_continuous_batch

from cgfn.field import drive_input, sech2_field_iteration, predict_driven_ftle_field  # noqa: E402  (Bibliothek)


# ── HYPOTHESE A: selbstkonsistente Feld-Iteration ──

def hyp_a(lines, seeds, hidden, taus=(1.0,), lams=(0.6, 1.0, 1.4), s_vals=(0.0, 0.5, 1.0, 2.0)):
    lines.append("")
    lines.append("=" * 100)
    lines.append("HYPOTHESE A — Geschlossenes Gesetz Runde 2: Feld-Iteration sagt getriebenen Benettin-FTLE voraus")
    lines.append("=" * 100)
    lines.append(f"{'lam':>5s} {'s':>4s} {'kap':>6s} {'pred':>8s} {'ben':>9s} {'err':>7s}")
    rows = []
    for lam in lams:
        for s in s_vals:
            for seed in seeds[:2]:
                torch.manual_seed(seed)
                m = make_designed_model(hidden, 1.0, lam, s, seed, task_dim=0)
                x = drive_input(4, 48, 0.5)
                pred = predict_driven_ftle_field(m, x)
                ben = benettin_ftle_max(m, x)
                kap = spectral_edge(m) / max(abs(float(eig_max(m))), 1e-9)
                err = pred - ben
                rows.append((pred, ben, err))
                lines.append(f"{lam:5.2f} {s:4.1f} {kap:6.2f} {pred:+8.3f} {ben:+9.3f} {err:+7.3f}")
    errs = torch.tensor([abs(r[2]) for r in rows])
    lines.append(f"mittlerer |Fehler|: {errs.mean():.3f} — (F10-Skalar: 0.91; <0.2 = Gesetz-Kandidat)")
    # Vorzeichen-Einheit
    agree = sum(1 for p, b, _ in rows if (p > 0) == (b > 0))
    lines.append(f"Vorzeichen-Einheit: {agree}/{len(rows)}")


# ── HYPOTHESE B: Gradienten-Richtung vs. Kanten-Drift ──

def hyp_b(lines, seeds, steps=300, lr=2e-3):
    lines.append("")
    lines.append("=" * 100)
    lines.append("HYPOTHESE B — Gradienten-Mechanismus: drueckt der Task-Gradient Richtung spektraler Rand?")
    lines.append("=" * 100)
    cors = []
    for seed in seeds:
        torch.manual_seed(seed)
        m = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=32, tau=1.0)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        lossfn = torch.nn.MSELoss()
        drifts, proj_rands = [], []
        lam_prev = None
        for step in range(steps):
            x, y = make_continuous_batch("sinus_memory", 16, 16)
            out, _ = m(x)
            loss = lossfn(out, y)
            opt.zero_grad(); loss.backward()
            # Projektion des W-Gradients auf die Richtung "Eigenwert-Rand hoch"
            if m.W.grad is not None:
                W = m.W.detach()
                lams = torch.linalg.eigvals(W)
                idx = lams.real.argmax()
                v = lams[idx]  # komplex; wir nutzen den Realteil als Skalar-Mass
                # Richtung: empirisch projizieren wir auf dominanten Eigen-Realteil-Anteil
                # ueber Finite-Differenz: dlam nach einem Mini-Schritt
                lam_now = lams.real.max().item()
                if lam_prev is not None:
                    drifts.append(lam_now - lam_prev)
                    proj_rands.append(float(m.W.grad.norm()))
                lam_prev = lam_now
            opt.step()
        if len(drifts) > 3:
            d = torch.tensor(drifts); p = torch.tensor(proj_rands)
            if d.std() > 1e-9 and p.std() > 1e-9:
                c = torch.corrcoef(torch.stack([d, p]))[0, 1].item()
                cors.append(c)
                lines.append(f"seed {seed}: corr(dlam_max, ||dW||) = {c:+.2f}, lam {drifts[0]:+.3f} -> {drifts[-1]:+.3f} (sum dlam {sum(drifts):+.3f})")
        else:
            lines.append(f"seed {seed}: zu wenig Drift-Daten")
    if cors:
        mean_c = sum(cors) / len(cors)
        lines.append(f"Mittel-corr: {mean_c:+.2f} — (sum dlam > 0 systematisch => SGD drueckt RICHTUNG Rand: Mechanismus-Kandidat)")
    else:
        lines.append("keine Korrelation berechenbar")


# ── HYPOTHESE C: kappa*(Horizont)-Gesetz ──

def hyp_c(lines, seeds, hidden=32, lags=(4, 8, 16, 32), s_vals=(0.0, 0.5, 1.0, 2.0), lam=0.8):
    lines.append("")
    lines.append("=" * 100)
    lines.append("HYPOTHESE C — Design-Dial: kappa*(Horizont)-Gesetz (MC-Optimum vs. max_lag)")
    lines.append("=" * 100)
    header = f"{'lag':>4s} " + " ".join(f"{'s=' + str(s):>7s}" for s in s_vals)
    lines.append(header)
    best_kappas = []
    for lag in lags:
        mcs = []
        for s in s_vals:
            vals = []
            for seed in seeds[:2]:
                torch.manual_seed(seed)
                m = make_designed_model(hidden, 1.0, lam, s, seed, task_dim=0)
                mc, _ = memory_capacity(m, max_lag=lag, input_scale=0.5)
                vals.append(mc)
            mcs.append(sum(vals) / len(vals))
        row = f"{lag:4d} " + " ".join(f"{mc:7.2f}" for mc in mcs)
        lines.append(row)
        best_i = max(range(len(mcs)), key=lambda i: mcs[i])
        best_kappas.append((lag, s_vals[best_i], mcs[best_i]))
        print(row, flush=True)
    lines.append("Optimum pro lag: " + ", ".join(f"L={l}: s*={s}" for l, s, _ in best_kappas))
    mono = all(best_kappas[i+1][1] >= best_kappas[i][1] for i in range(len(best_kappas)-1))
    lines.append(f"s*(lag) monoton wachsend: {mono} — (True => kappa als Horizont-Dial: BAU-ANWEISUNG)")


# ── HYPOTHESE D: Kante(sigma) ──

def hyp_d(lines, seeds, hidden=32, sigmas=(0.1, 0.2, 0.5, 1.0, 2.0)):
    lines.append("")
    lines.append("=" * 100)
    lines.append("HYPOTHESE D — Kante(W, sigma): getriebene Benettin-Kante vs. Treib-Skala")
    lines.append("=" * 100)
    lines.append(f"{'sigma':>6s} {'ben':>9s} {'pred_field':>10s}")
    for seed in seeds[:2]:
        torch.manual_seed(seed)
        m = make_designed_model(hidden, 1.0, 1.0, 0.5, seed, task_dim=0)
        lines.append(f"--- seed {seed} (lam=1.0, s=0.5) ---")
        for sig in sigmas:
            x = drive_input(4, 48, sig)
            ben = benettin_ftle_max(m, x)
            pred = predict_driven_ftle_field(m, x)
            lines.append(f"{sig:6.2f} {ben:+9.3f} {pred:+10.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--out", default="results/f17_battery_results.txt")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 100)
    lines.append("F17 FRONTIER-BATTERIE: 4 spekulative Durchbruch-Kandidaten (A: geschlossenes Gesetz Runde 2 / B: Gradienten-Mechanismus / C: kappa*(Horizont)-Dial / D: Kante(sigma))")
    lines.append("=" * 100)
    hyp_a(lines, seeds, 32)
    hyp_b(lines, seeds)
    hyp_c(lines, seeds)
    hyp_d(lines, seeds)
    lines.append("")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
