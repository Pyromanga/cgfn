"""
F18 — DAS GESCHLOSSENE GESETZ: Feld-Iteration + Transienten-Korrektur.

F17-A zeigte: Die Pro-Unit-sech^2-Feld-Iteration (sech2_field_iteration)
sagt die getriebene Benettin-Kante aus W + Treib-Statistik OHNE Rollout
voraus — mittlerer |Fehler| 0.172 (F10-Skalar: 0.91), bei kappa<5 fast
exakt. Der EINZIGE systematische Fehler: bei kappa>=8 wird zu STARK
negativ vorhergesagt (Feld-Jacobi = Linearisierung am Punkt verpasst das
TRANSMIENTE Wachstum nicht-normaler Matrizen — die F11/F13-Physik).

F18 prueft drei Transienten-Korrekturen auf demselben Gitter:
  (T0) reines Feld-Gesetz (F17-A, Baseline)
  (T1) + Kreiss-Konstante: addiere K(W) = ||W||*||W^{-1}|| - 1 als
       Scharnier auf den Praeexponenten des transienten Anteils
  (T2) + Pseudospektren-Schranke (Trefethen): nutze die eps-Pseudo-
       Spektral-Unterschranke fuer das transient-maximale Wachstum:
       growth_transient <= max_s sv((J - sI)^{-1})^{-1}
       (wir evaluieren auf einem 1D-Gitter im Spektralbereich)

Zusaetzlich Statistik-Aufstockung: 4 Seeds, feineres Gitter bei kappa
4-10 (der F17-Fehler-Bereich), H-Skala 16/32.

FRAGE: Bringt eine der Korrekturen den mittleren |Fehler| unter 0.1 und
die Vorzeichen-Einheit ueber 90%? Dann ist das GESETZ geschlossen:
edge(W, Statistik) — Design ohne Rollout, mit quantifizierter Grenze.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.design import make_designed_model
from cgfn.cavity import eig_max, spectral_edge
from cgfn.benettin import benettin_ftle_max
from experiments.f17_battery import sech2_field_iteration, predict_driven_ftle, drive_input


def kreiss_constant(J):
    """K(J) = ||J||_2 * ||J^{-1}||_2 - 1 (Scharnier fuer Transienten)."""
    sv = torch.linalg.svdvals(J)
    sv_min = sv[-1].item() if sv[-1].item() > 1e-12 else 1e-12
    return (sv[0].item() / sv_min) - 1.0


def transient_bound_pseudo(J, n_grid=16):
    """Trefethen-Unterschranke: max_s 1/sv((J-sI)^{-1})...
    Genauer: transient growth <= max_{s in grid} sv((J - sI)^{-1})^{-1}
    (approximiert; wir nutzen ein grobes 1D-Gitter entlang des
    Spektral-Realteils von lambda_min bis lambda_max)."""
    H = J.shape[0]
    lams = torch.linalg.eigvals(J).real
    grid = torch.linspace(lams.min().item(), lams.max().item(), n_grid)
    best = 0.0
    I = torch.eye(H)
    for s in grid:
        sv_min = torch.linalg.svdvals((J - s * I).inverse())[0].item() if False else None
        # sv_min of M = sv_max of M^{-1} -> wir nutzen: kleinste Singulaerwert
        M = J - s * I
        sv = torch.linalg.svdvals(M)
        if sv[-1].item() < 1e-12:
            return float("inf")  # s im Spektrum: transientes Wachstum beliebig
        sv_min_inv = 1.0 / sv[-1].item()
        best = max(best, sv_min_inv)
    return best


@torch.no_grad()
def predict_with_transient(model, x_probe, mode="T1"):
    """Feld-FTLE + Transienten-Korrektur.

    T1: pred = field + alpha * log(1 + K(J_field))  (Kreiss-Scharnier)
    T2: pred = field + beta * log(1 + transient_bound_pseudo(J_field))
    alpha/beta fest 0.1 (kein Fitting auf dem Test — erst F18-Statistik
    darf kalibrieren; hier: qualitative Pruefung, ob die RICHTUNG stimmt).
    """
    d = sech2_field_iteration(model, x_probe)
    H = model.hidden_dim
    J = -torch.eye(H) / model.tau + model.W * d.unsqueeze(0)
    base = torch.linalg.eigvals(J).real.max().item()
    if mode == "T0":
        return base
    if mode == "T1":
        return base + 0.1 * torch.log1p(torch.tensor(kreiss_constant(J))).item()
    if mode == "T2":
        tb = transient_bound_pseudo(J)
        if tb == float("inf"):
            return base  # degeneriert — nur Baseline
        return base + 0.1 * torch.log1p(torch.tensor(tb)).item()
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--hidden-dims", default="32,16")
    p.add_argument("--lams", default="0.8,1.2")
    p.add_argument("--s-values", default="0.0,0.4,0.8,1.2,1.6,2.0")
    p.add_argument("--out", default="results/f18_law_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    hiddens = [int(h) for h in args.hidden_dims.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 118)
    lines.append("F18 GESCHLOSSENES GESETZ: Feld-Iteration + Transienten-Korrektur (T0 Baseline / T1 Kreiss / T2 Pseudospektral)")
    lines.append(f"seeds={seeds} H={hiddens} lams={lams} s={s_vals} — alpha/beta fest 0.1, kein Fitting")
    lines.append("FRAGE: |Fehler| < 0.1 und Vorzeichen-Einheit > 90%?")
    lines.append("=" * 118)
    lines.append(f"{'H':>4s} {'lam':>5s} {'s':>4s} {'kap':>6s} {'ben':>8s} {'T0':>7s} {'T1':>7s} {'T2':>7s}")

    errs = {"T0": [], "T1": [], "T2": []}
    signs = {"T0": 0, "T1": 0, "T2": 0}
    n = 0
    for H in hiddens:
        for lam in lams:
            for s in s_vals:
                for seed in seeds:
                    torch.manual_seed(seed)
                    m = make_designed_model(H, 1.0, lam, s, seed, task_dim=0)
                    x = drive_input(4, 48, 0.5)
                    ben = benettin_ftle_max(m, x)
                    kap = spectral_edge(m) / max(abs(float(eig_max(m))), 1e-9)
                    row_err = []
                    for mode in ("T0", "T1", "T2"):
                        pred = predict_with_transient(m, x, mode)
                        err = pred - ben
                        errs[mode].append(abs(err))
                        row_err.append(pred)
                        if (pred > 0) == (ben > 0):
                            signs[mode] += 1
                    n += 1
                    line = (f"{H:4d} {lam:5.2f} {s:4.1f} {kap:6.2f} {ben:+8.3f} "
                            f"{row_err[0]:+7.3f} {row_err[1]:+7.3f} {row_err[2]:+7.3f}")
                    lines.append(line)
                    if seed == seeds[0]:
                        print(line, flush=True)

    lines.append("")
    lines.append("--- Bewertung ---")
    for mode in ("T0", "T1", "T2"):
        e = torch.tensor(errs[mode])
        lines.append(f"{mode}: mittlerer |Fehler| {e.mean():.3f} | Vorzeichen-Einheit {signs[mode]}/{n} ({100.0*signs[mode]/n:.0f}%)")
    lines.append("")
    lines.append("Lesart: Wenn T1 oder T2 den mittleren |Fehler| gegen T0 signifikant")
    lines.append("senkt (insbesondere im kappa>=8-Bereich) OHNE bei kappa<5 zu schaden,")
    lines.append("ist das transiente Wachstum die richtige Korrektur und das GESETZ mit")
    lines.append("quantifizierter Grenze geschlossen: edge(W, Statistik) ohne Rollout.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
