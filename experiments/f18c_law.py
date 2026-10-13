"""
F18c — ZWEI-REGIME-GESETZ: κ-Schalter statt globaler Ridge.

F18b-Befund: Das Feld-Gesetz ist bei kappa<5 EXZELLENT (0.057) und
die Transienten-Korrektur hilft bei kappa>=10 REAL (0.329 -> 0.145)
— aber die globale Ridge verschmiert beides (held-out 0.163):
ein Modell fuer zwei verschiedene Physik-Regime.

F18c testet die natuerliche Struktur: ZWEI REGIME mit κ-Schalter.
  Regime A (kappa < kappa_c): T0 pur — die Linearisierung IST das Gesetz.
  Regime B (kappa >= kappa_c): T0 + gelernte Transienten-Korrektur
    (Ridge auf [1, log1p(K), log1p(pseudo)] — OHNE T0-Feature, denn
    dort ist die Linearisierung ja gerade unzulaenglich; die Korrektur
    lernt den FEHLER, nicht das Ziel neu).
  kappa_c und die Ridge-Koeffizienten werden NUR auf dem Train-Split
  gefittet (Grid-Search ueber kappa_c, Ridge geschlossen), bewertet auf
  dem HELD-OUT-Split. Keine Rueckkopplung.

FRAGE: Held-out |err| < 0.10 und Vorzeichen > 85% — GESCHLOSSEN?
Zusatz: learnbare Schwellen-Interpretation — wenn kappa_c ~ 5-8
rauskommt, ist der Schalter genau die F11/F13-Transienten-Schwelle.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.design import make_designed_model
from cgfn.cavity import eig_max, spectral_edge
from cgfn.benettin import benettin_ftle_max
from cgfn.field import (sech2_field_iteration, drive_input,
                        kreiss_constant, transient_bound_pseudo)


@torch.no_grad()
def features(model, x_probe):
    d = sech2_field_iteration(model, x_probe)
    H = model.hidden_dim
    J = -torch.eye(H) / model.tau + model.W * d.unsqueeze(0)
    t0 = torch.linalg.eigvals(J).real.max().item()
    k = kreiss_constant(J)
    tb = transient_bound_pseudo(J)
    tb = 10.0 if tb == float("inf") else tb
    return {"t0": t0,
            "logK": torch.log1p(torch.tensor(k)).item(),
            "logP": torch.log1p(torch.tensor(tb)).item()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--hidden-dims", default="32,16")
    p.add_argument("--lams", default="0.8,1.2")
    p.add_argument("--s-values", default="0.0,0.4,0.8,1.2,1.6,2.0")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--kappa-grid", default="3,4,5,6,7,8,9,10,12")
    p.add_argument("--out", default="results/f18c_law_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    hiddens = [int(h) for h in args.hidden_dims.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]
    kgrid = [float(v) for v in args.kappa_grid.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    for H in hiddens:
        for lam in lams:
            for s in s_vals:
                for seed in seeds:
                    torch.manual_seed(seed)
                    m = make_designed_model(H, 1.0, lam, s, seed, task_dim=0)
                    x = drive_input(4, 48, 0.5)
                    ben = benettin_ftle_max(m, x)
                    kap = spectral_edge(m) / max(abs(float(eig_max(m))), 1e-9)
                    f = features(m, x)
                    rows.append({"t0": f["t0"], "logK": f["logK"],
                                 "logP": f["logP"], "ben": ben, "kap": kap})
    n = len(rows)

    train = [r for i, r in enumerate(rows) if i % 2 == 0]
    test = [r for i, r in enumerate(rows) if i % 2 == 1]

    def fit_correction(subset, ridge):
        """Ridge: ben - t0 ~ [1, logK, logP] auf Regime-B-Subset."""
        X = torch.tensor([[1.0, r["logK"], r["logP"]] for r in subset])
        y = torch.tensor([r["ben"] - r["t0"] for r in subset])
        A = X.T @ X + ridge * torch.eye(3)
        return torch.linalg.solve(A, X.T @ y)

    def evaluate(subset, kc, w):
        errs, signs = [], 0
        for r in subset:
            if r["kap"] < kc:
                pred = r["t0"]
            else:
                z = torch.tensor([1.0, r["logK"], r["logP"]])
                pred = r["t0"] + float(z @ w)
            errs.append(abs(pred - r["ben"]))
            if (pred > 0) == (r["ben"] > 0):
                signs += 1
        return sum(errs) / len(errs), signs / len(subset)

    # Grid-Search kappa_c auf TRAIN
    best = None
    for kc in kgrid:
        sub = [r for r in train if r["kap"] >= kc]
        if len(sub) < 8:
            continue
        w = fit_correction(sub, args.ridge)
        e, sg = evaluate(train, kc, w)
        if best is None or e < best[1]:
            best = (kc, e, sg, w)
    kc, e_tr, sg_tr, w = best

    e_te, sg_te = evaluate(test, kc, w)

    # Referenzen auf demselben Test-Split
    e_t0 = sum(abs(r["t0"] - r["ben"]) for r in test) / len(test)
    sg_t0 = sum(1 for r in test if (r["t0"] > 0) == (r["ben"] > 0)) / len(test)

    lines = []
    lines.append("=" * 100)
    lines.append("F18c ZWEI-REGIME-GESETZ: T0 pur (kappa<kc) + gelernte Transienten-Korrektur (kappa>=kc)")
    lines.append(f"n={n} (train {len(train)}, held-out {len(test)}) seeds={seeds} H={hiddens}")
    lines.append(f"kc-Grid-Search auf TRAIN; Ridge-Korrektur lernt (ben - t0) aus [1, log1p(K), log1p(pseudo)]")
    lines.append("=" * 100)
    lines.append(f"gewaehlter Schalter: kappa_c = {kc:.1f} (train |err| {e_tr:.3f}, Vorzeichen {100*sg_tr:.0f}%)")
    lines.append(f"Korrektur-Koeffizienten: bias={w[0]:+.3f}, log1p(K)={w[1]:+.3f}, log1p(pseudo)={w[2]:+.3f}")
    lines.append("")
    lines.append(f"{'Modell':>24s} {'|Fehler|':>9s} {'Vorzeichen':>11s}")
    lines.append(f"{'T0 pur (global)':>24s} {e_t0:9.3f} {100*sg_t0:10.0f}%")
    lines.append(f"{'ZWEI-REGIME (HELD-OUT)':>24s} {e_te:9.3f} {100*sg_te:10.0f}%")
    lines.append("")
    lines.append("--- Held-out nach Regime ---")
    for name, cond in (("A: kappa<kc", lambda r: r["kap"] < kc),
                       ("B: kappa>=kc", lambda r: r["kap"] >= kc)):
        sub = [r for r in test if cond(r)]
        if not sub:
            continue
        e_a = sum(abs(r["t0"] - r["ben"]) for r in sub) / len(sub)
        lines.append(f"{name}: n={len(sub)} | T0 {e_a:.3f} | (Regime-Modell: siehe oben)")
    lines.append("")
    lines.append("Lesart: Held-out |err| < 0.10 bei >85% Vorzeichen => GESCHLOSSEN.")
    lines.append(f"Interpretation kappa_c={kc:.1f}: (F11/F13-Transienten-Schwelle erwartet ~5-8)")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
