"""
F18d — DIE KONSTANTEN-FRAGE: Warum ist die Regime-B-Korrektur +0.3?
Und: Ist die Konstante ARCHITEKTUR-UNIVERSELL (der Durchbruchs-Test)?

F18c: Das Zwei-Regime-Gesetz (FTLE ~ T0 bei kappa<6, T0+0.3 bei kappa>=6)
verfehlt das Ziel knapp (0.114/83%). Die gelernte Korrektur degenerierte
zu einer Konstanten — offen ist WARUM, und ob die Konstante universell ist.

Teil 1 — Struktur der Korrektur (designed Netze):
  Plot der Residuen (ben - T0) gegen kappa im Band kappa>=5:
  Ist das Defizit wirklich FLACH (konstant) oder leicht strukturiert?
  Fit: residuum ~ a + b*kappa + c*kappa^2 auf Train, Held-out-Bewertung.
  Vergleich gegen den F18c-Konstanten-Fit.

Teil 2 — ARCHITEKTUR-UNIVERSELLITAET (der entscheidende Test):
  Dasselbe Zwei-Regime-Gesetz auf den F13-Zoo-Netzen (gru, cornn —
  Initialisierung UND nach 200 Trainings-Steps), gemessen mit deren
  eigenen T0 (architekturspezifische Jacobi-Feld-Rechnung ist nicht
  verfuegbar — stattdessen: ben vs. (freies) T0-Netz-Mass: wir nutzen
  den CT-Proxy rho/W-Schur-Mass... NEIN: Ehrlich bleiben — fuer Zoo-
  Netze haben wir kein Feld-T0. Stattdessen messen wir das, was das
  Gesetz MOEGLICHERWEISE verallgemeinert: der DRIVEN-Benettin minus
  Benettin auf WEISSEM Drive (sigma=0 vs sigma=0.5) — das Treib-Defizit
  D(sigma) = ben(sigma=0.5) - ben(sigma=0). Wenn D auf Zoo-Netzen
  ebenfalls ~ +0.3 bei kappa>=6 und ~ 0 darunter liegt, ist die
  Konstante eine EIGENSCHAFT DER TREIBUNG x KAPPA, nicht unserer CT-
  Architektur — das waere der Universell-Befund.

Teil 3 — Sattigungs-Hypothese:
  Wenn die Konstante aus dem sech^2-Deckel kommt, sollte sie mit der
  Treib-Skala sigma wachsen und saettigen: D(sigma) auf Sigma-Sweep
  (0.25/0.5/1.0/2.0) bei festem Netz mit kappa>=6. Saettigungskurve
  => Deckel-Physik; lineares Wachstum => andere Mechanik.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.design import make_designed_model
from cgfn.cavity import eig_max, spectral_edge
from cgfn.benettin import benettin_ftle_max
from cgfn.field import sech2_field_iteration, drive_input
from cgfn.zoo import make_zoo_model


@torch.no_grad()
def field_t0(model, x_probe):
    d = sech2_field_iteration(model, x_probe)
    H = model.hidden_dim
    J = -torch.eye(H) / model.tau + model.W * d.unsqueeze(0)
    return torch.linalg.eigvals(J).real.max().item()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--lams", default="0.8,1.2")
    p.add_argument("--s-values", default="0.0,0.8,1.2,1.6,2.0")
    p.add_argument("--out", default="results/f18d_law_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 110)
    lines.append("F18d KONSTANTEN-FRAGE: Struktur der Regime-B-Korrektur + Architektur-Universalitaet + Sattigung")
    lines.append("=" * 110)

    # ── Teil 1: Residuen-Struktur auf designed Netzen ──
    lines.append("")
    lines.append("--- Teil 1: Residuum (ben - T0) gegen kappa (designed, kappa>=5) ---")
    rows = []
    for lam in lams:
        for s in s_vals:
            for seed in seeds:
                torch.manual_seed(seed)
                m = make_designed_model(32, 1.0, lam, s, seed, task_dim=0)
                x = drive_input(4, 48, 0.5)
                ben = benettin_ftle_max(m, x)
                kap = spectral_edge(m) / max(abs(float(eig_max(m))), 1e-9)
                t0 = field_t0(m, x)
                if kap >= 5.0:
                    rows.append((kap, ben - t0))
    tr = [r for i, r in enumerate(rows) if i % 2 == 0]
    te = [r for i, r in enumerate(rows) if i % 2 == 1]
    # Konstanten-Fit vs. linearer Fit vs. quadratisch (auf Train)
    ktr = torch.tensor([r[0] for r in tr]); rtr = torch.tensor([r[1] for r in tr])
 kte = torch.tensor([r[0] for r in te]); rte = torch.tensor([r[1] for r in te])
    c_const = rtr.mean().item()
    def fit(deg):
        X = torch.stack([ktr ** d for d in range(deg + 1)], dim=1)
        A = X.T @ X + 1e-6 * torch.eye(deg + 1)
        return torch.linalg.solve(A, X.T @ rtr)
    def evalfit(w, deg):
        X = torch.stack([kte ** d for d in range(deg + 1)], dim=1)
        pred = X @ w
        return (pred - rte).abs().mean().item()
    w1, w2 = fit(1), fit(2)
    lines.append(f"n kappa>=5: train {len(tr)}, held-out {len(te)}")
    lines.append(f"Konstanten-Fit: c = {c_const:+.3f} | held-out |err| {evalfit(torch.tensor([c_const]), 0):.3f}")
    lines.append(f"Linear-Fit:  a={w1[0]:+.3f} b={w1[1]:+.4f} | held-out |err| {evalfit(w1, 1):.3f}")
    lines.append(f"Quadrat-Fit: a={w2[0]:+.3f} b={w2[1]:+.4f} c={w2[2]:+.5f} | held-out |err| {evalfit(w2, 2):.3f}")

    # ── Teil 2: Architektur-Universalitaet (Zoo: Treib-Defizit D) ──
    lines.append("")
    lines.append("--- Teil 2: Treib-Defizit D = ben(sigma=0.5) - ben(sigma=0) auf Zoo-Netzen ---")
    lines.append("(Wenn D ~ +0.3 bei kappa>=6 und ~0 darunter: die Konstante ist Treib-x-kappa-Physik, nicht CT-spezifisch)")
    lines.append(f"{'model':>8s} {'phase':>7s} {'seed':>4s} {'kappa':>6s} {'ben0':>8s} {'ben.5':>8s} {'D':>7s}")
    for name in ("gru", "cornn"):
        for phase in ("init", "trained"):
            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                m = make_zoo_model(name, hidden_dim=32, seed=seed)
                if phase == "trained":
                    from cgfn.continuous import make_continuous_batch
                    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
                    torch.manual_seed(seed + 1000)
                    for _ in range(200):
                        xb, yb = make_continuous_batch("sinus_memory", 16, 16)
                        out, _ = m(xb)
                        loss = torch.nn.MSELoss()(out, yb)
                        opt.zero_grad(); loss.backward(); opt.step()
                # kappa des rekurrenten Gewichts (wie F16)
                W = m.core.weight_hh if name == "gru" else None
                if W is None:
                    continue  # cornn-J-Matrix hier nicht als einfaches W — nur gru bewerten
                sv = torch.linalg.svdvals(W.detach())[0].item()
                lam = max(torch.linalg.eigvals(B).abs().max().item() for B in W.detach().chunk(3, dim=0))
                kap = sv / max(abs(lam), 1e-9)
                gen = torch.Generator().manual_seed(777)
                x0 = torch.zeros(4, 48, 1)
                xh = torch.randn(4, 48, 1, generator=gen) * 0.5
                ben0 = benettin_ftle_max(m, x0)
                benh = benettin_ftle_max(m, xh)
                lines.append(f"{name:>8s} {phase:>7s} {seed:4d} {kap:6.2f} {ben0:+8.3f} {benh:+8.3f} {benh-ben0:+7.3f}")

    # ── Teil 3: Sattigung D(sigma) ──
    lines.append("")
    lines.append("--- Teil 3: Sattigungs-Kurve D(sigma) bei festem designed Netz (lam=1.2, s=1.6 => kappa~11) ---")
    torch.manual_seed(0)
    m = make_designed_model(32, 1.0, 1.2, 1.6, 0, task_dim=0)
    lines.append(f"{'sigma':>6s} {'ben':>8s} {'D':>7s}")
    ben0 = benettin_ftle_max(m, torch.zeros(4, 48, 1))
    for sig in (0.25, 0.5, 1.0, 2.0):
        x = drive_input(4, 48, sig)
        ben = benettin_ftle_max(m, x)
        lines.append(f"{sig:6.2f} {ben:+8.3f} {ben-ben0:+7.3f}")

    lines.append("")
    lines.append("Lesart: (1) Konstanten-Fit schlaegt strukturierte Fits => Korrektur ist wirklich flach;")
    lines.append("(2) D ~ +0.3 auf gru bei kappa>=6 => Konstante architektur-universell;")
    lines.append("(3) D(sigma) saettigt => sech^2-Deckel-Physik bestaetigt die Herkunft.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
