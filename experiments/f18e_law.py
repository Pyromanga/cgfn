"""
F18e — NUMERISCHE ABSZISSE: das fehlende Physik-Feature fuer das Gesetz.

F18d-Befunde: (1) Die +0.3-Korrektur ist NICHT flach (linear/quadratisch
schlagen die Konstante) — das Gesetz braucht ein physikalisches Feature.
(2) GRU-Test inkonklusiv (keine GRU erreichte kappa>=6). (3) D(sigma)
nicht-monoton mit Peak — Deckel-Physik am Werk.

THEORIE-IDEE (F18d-Folgerung): Das Benettin-Fenster mit Re-Normierung
misst eine WACHSTUMSRATE. Fuer nicht-normaler J gilt: die anfaengliche
Wachstumsrate einer Störung ist d/dt (1/2) ln||delta||^2 =
delta^T (J+J^T) delta / 2||delta||^2 <= omega(J), mit
omega(J) = lambda_max((J+J^T)/2) — der NUMERISCHEN ABSZISSE.
Regime-Struktur-Vorhersage: bei kappa~1 ist J fast normal =>
omega ≈ spektral => T0 reicht. Bei kappa>=6 dominiert transientes
Wachstum => FTLE naeher an omega als an T0.

FEATURE-TEST (held-out wie F18b/c):
  P1: T0 pur (Baseline)
  P2: omega pur
  P3: Ridge [1, T0, omega] (die Kopplung lernen)
  P4: max(T0, omega) (Regel-Schalter, kein Fit!)
  P5: max(T0, omega - 0.5*omega_0) mit omega_0 = Norm-Mass? NEIN —
      P4 ist der saubere fit-freie Kandidat; P5 entfaellt.

ZIEL: Held-out |err| < 0.10 bei >85% Vorzeichen mit P4 (OHNE FIT)
waere der Durchbruch: das Gesetz ist dann max(spektral, transient) —
zwei Zeilen Mathematik, keine gelernten Parameter.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.design import make_designed_model
from cgfn.cavity import eig_max, spectral_edge
from cgfn.benettin import benettin_ftle_max
from cgfn.field import sech2_field_iteration, drive_input


@torch.no_grad()
def field_t0_omega(model, x_probe):
    d = sech2_field_iteration(model, x_probe)
    H = model.hidden_dim
    J = -torch.eye(H) / model.tau + model.W * d.unsqueeze(0)
    t0 = torch.linalg.eigvals(J).real.max().item()
    omega = torch.linalg.eigvalsh((J + J.T) / 2)[-1].item()
    return t0, omega


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--hidden-dims", default="32,16")
    p.add_argument("--lams", default="0.8,1.2")
    p.add_argument("--s-values", default="0.0,0.4,0.8,1.2,1.6,2.0")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--out", default="results/f18e_law_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    hiddens = [int(h) for h in args.hidden_dims.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]
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
                    t0, om = field_t0_omega(m, x)
                    rows.append({"t0": t0, "om": om, "ben": ben, "kap": kap})
    n = len(rows)
    train = [r for i, r in enumerate(rows) if i % 2 == 0]
    test = [r for i, r in enumerate(rows) if i % 2 == 1]

    lines = []
    lines.append("=" * 110)
    lines.append("F18e NUMERISCHE ABSZISSE omega(J)=lam_max((J+J^T)/2) als Gesetz-Feature (held-out)")
    lines.append(f"n={n} (train {len(train)}, held-out {len(test)}) seeds={seeds} H={hiddens}")
    lines.append("P4 = max(T0, omega) ist FIT-FREI — kein gelernter Parameter")
    lines.append("=" * 110)

    def stats(sub, pred_fn):
        errs = [abs(pred_fn(r) - r["ben"]) for r in sub]
        signs = sum(1 for r in sub if (pred_fn(r) > 0) == (r["ben"] > 0))
        return sum(errs) / len(errs), signs / len(sub)

    p1 = stats(test, lambda r: r["t0"])
    p2 = stats(test, lambda r: r["om"])
    p4 = stats(test, lambda r: max(r["t0"], r["om"]))

    # P3: Ridge [1, T0, omega] auf Train, Skalierung aus Train
    Xtr = torch.tensor([[1.0, r["t0"], r["om"]] for r in train])
    ytr = torch.tensor([r["ben"] for r in train])
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    Ztr = (Xtr - mu) / sd
    A = Ztr.T @ Ztr + args.ridge * torch.eye(3)
    w = torch.linalg.solve(A, Ztr.T @ ytr)
    Xte = torch.tensor([[1.0, r["t0"], r["om"]] for r in test])
    Zte = (Xte - mu) / sd
    p3_err = ((Zte @ w) - torch.tensor([r["ben"] for r in test])).abs().mean().item()
    p3_pred = (Zte @ w).tolist()
    p3_sign = sum(1 for pv, r in zip(p3_pred, test) if (pv > 0) == (r["ben"] > 0)) / len(test)

    lines.append(f"{'Praediktor':>26s} {'|Fehler|':>9s} {'Vorzeichen':>11s}")
    lines.append(f"{'P1 T0 pur (Baseline)':>26s} {p1[0]:9.3f} {100*p1[1]:10.0f}%")
    lines.append(f"{'P2 omega pur':>26s} {p2[0]:9.3f} {100*p2[1]:10.0f}%")
    lines.append(f"{'P3 Ridge [1,T0,omega]':>26s} {p3_err:9.3f} {100*p3_sign:10.0f}%")
    lines.append(f"{'P4 max(T0, omega) FIT-FREI':>26s} {p4[0]:9.3f} {100*p4[1]:10.0f}%")
    lines.append(f"{'(Ridge-Koeff.: T0 {w[1]:+.3f}, omega {w[2]:+.3f})':>26s}")
    lines.append("")
    lines.append("--- Held-out nach kappa-Band: P4 vs T0 ---")
    for lo, hi in ((1.0, 5.0), (5.0, 10.0), (10.0, 99.0)):
        sub = [r for r in test if lo <= r["kap"] < hi]
        if not sub:
            continue
        e_t0 = sum(abs(r["t0"] - r["ben"]) for r in sub) / len(sub)
        e_p4 = sum(abs(max(r["t0"], r["om"]) - r["ben"]) for r in sub) / len(sub)
        lines.append(f"kappa [{lo:.0f},{hi:.0f}): n={len(sub):3d} | T0 {e_t0:.3f} | P4 {e_p4:.3f}")
    lines.append("")
    lines.append("Lesart: P4 fit-frei unter 0.10/85% => GESETZ GESCHLOSSEN als max(spektral,")
    lines.append("numerische Abszisse): zwei Zeilen Mathematik, keine gelernten Parameter.")
    lines.append("P3-Koeffizienten zeigen die Kopplung (omega-Gewicht > 0 erwartet).")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
