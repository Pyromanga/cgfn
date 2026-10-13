"""
F18b — Gesetz-Kalibrierung mit HELD-OUT-Test: lernt die Transienten-
Kopplung statt sie zu raten.

F18-Befund: T0 (Feld, |err| 0.174) bleibt Best; T1 (Kreiss, fest 0.1)
dreht bei kappa>=8 in die richtige Richtung, ueberschiesst aber bei
kappa<5; T2 (Pseudospektrum, fest 0.1) ueberschiesst ueberall. Die
Korrekturen sind nicht falsch, sondern UNGEKOPPELT.

F18b macht die erlaubte Kalibrierung — aber ehrlich:
  - Features pro Zeile: T0 (Feld-LLE), log1p(K(J)), log1p(pseudo_bound),
    plus Konstante.
  - Ridge-Regression ben ~ features auf der HAELFTE des Gitters
    (gerade Indizes), Bewertung auf der gehaltenen Haelfte (ungerade).
  - KEINE Rueckkopplung auf den Test — der Held-out-Fehler ist die
    ehrliche Gesetzes-Guete.

FRAGE: Held-out |Fehler| < 0.12 und Vorzeichen-Einheit > 85%?
Dann ist edge(W, Statistik) mit gelernter Kopplung geschlossen —
und die gelernten Koeffizienten SAGEN UNS, wie die Transienten-Physik
in das Gesetz eingeht (interpretables Nebenresultat).
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
    return [1.0, t0, torch.log1p(torch.tensor(k)).item(),
            torch.log1p(torch.tensor(tb)).item()]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--hidden-dims", default="32,16")
    p.add_argument("--lams", default="0.8,1.2")
    p.add_argument("--s-values", default="0.0,0.4,0.8,1.2,1.6,2.0")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--out", default="results/f18b_law_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    hiddens = [int(h) for h in args.hidden_dims.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []  # (features, ben, kap)
    for H in hiddens:
        for lam in lams:
            for s in s_vals:
                for seed in seeds:
                    torch.manual_seed(seed)
                    m = make_designed_model(H, 1.0, lam, s, seed, task_dim=0)
                    x = drive_input(4, 48, 0.5)
                    ben = benettin_ftle_max(m, x)
                    kap = spectral_edge(m) / max(abs(float(eig_max(m))), 1e-9)
                    rows.append((features(m, x), ben, kap))
    n = len(rows)
    X = torch.tensor([r[0] for r in rows])          # (n, 4)
    y = torch.tensor([r[1] for r in rows])           # (n,)

    # Split: gerade Indizes train, ungerade held-out
    idx_tr = torch.arange(n)[torch.arange(n) % 2 == 0]
    idx_te = torch.arange(n)[torch.arange(n) % 2 == 1]
    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xte, yte = X[idx_te], y[idx_te]

    # Feature-Skalierung (nur aus Train)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    # Ridge (geschlossene Loesung)
    A = Ztr.T @ Ztr + args.ridge * torch.eye(Ztr.shape[1])
    w = torch.linalg.solve(A, Ztr.T @ ytr)

    def predict(Z):
        return Z @ w

    # Baseline: reines T0 (Feature 1, unskaliert benutzen wir Original-T0)
    t0_tr = X[idx_tr, 1]
    t0_te = X[idx_te, 1]
    err_t0 = (t0_te - yte).abs().mean().item()
    sign_t0 = ((t0_te > 0) == (yte > 0)).float().mean().item()

    pred_te = predict(Zte)
    err_lr = (pred_te - yte).abs().mean().item()
    sign_lr = ((pred_te > 0) == (yte > 0)).float().mean().item()
    pred_tr = predict(Ztr)
    err_tr = (pred_tr - ytr).abs().mean().item()

    lines = []
    lines.append("=" * 100)
    lines.append("F18b GESETZ-KALIBRIERUNG (Ridge auf [1, T0, log1p(K), log1p(pseudo)]) — HELD-OUT")
    lines.append(f"n={n} (train {len(idx_tr)}, held-out {len(idx_te)}) seeds={seeds} H={hiddens} ridge={args.ridge}")
    lines.append("=" * 100)
    lines.append(f"gelernte Koeffizienten (skalierte Features): bias={w[0]:+.3f}, T0={w[1]:+.3f}, log1p(K)={w[2]:+.3f}, log1p(pseudo)={w[3]:+.3f}")
    lines.append("")
    lines.append(f"{'Modell':>22s} {'|Fehler|':>9s} {'Vorzeichen':>11s}")
    lines.append(f"{'T0 (Feld pur)':>22s} {err_t0:9.3f} {100*sign_t0:10.0f}%")
    lines.append(f"{'Ridge (train)':>22s} {err_tr:9.3f} {'--':>11s}")
    lines.append(f"{'Ridge (HELD-OUT)':>22s} {err_lr:9.3f} {100*sign_lr:10.0f}%")
    lines.append("")
    lines.append("--- Fehler nach kappa-Band (held-out) ---")
    for lo, hi in ((1.0, 5.0), (5.0, 10.0), (10.0, 99.0)):
        mask = [(lo <= rows[i][2] < hi) for i in idx_te.tolist()]
        if not any(mask):
            continue
        sel = [i for i, m_ in zip(idx_te.tolist(), mask) if m_]
        e_t0 = sum(abs(rows[i][0][1] - rows[i][1]) for i in sel) / len(sel)
        e_lr = sum(abs(float(predict(Zte)[j]) - rows[i][1])
                   for j, i in enumerate(idx_te.tolist()) if i in sel) / len(sel)
        lines.append(f"kappa in [{lo:.0f},{hi:.0f}): n={len(sel):3d} | T0 {e_t0:.3f} | Ridge {e_lr:.3f}")
    lines.append("")
    lines.append("Lesart: Held-out-Fehler < 0.12 bei >85% Vorzeichen-Einheit =>")
    lines.append("edge(W, Statistik) mit gelernter Transienten-Kopplung GESCHLOSSEN;")
    lines.append("die Koeffizienten quantifizieren, wie die F11-Transientenphysik in")
    lines.append("das Gesetz eingeht (erwartet: positives Gewicht auf log1p(K)).")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
