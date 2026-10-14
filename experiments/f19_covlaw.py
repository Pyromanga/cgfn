"""
F19 — KOVARIANZ-PROPAGATIONS-GESETZ: Benettin als geschlossene
Kovarianz-Iteration mit der Feld-Jacobi (fit-frei).

F18e widerlegte die numerische Abszisse (omega zu gross). Die ganze
F18-Kette hat gezeigt: additive Korrekturen auf T0 schaffen es nicht
unter 0.11 held-out. Der RICHTIGE Gedanke: Benettin-QR ueber ein
Fenster misst das exponentielle Wachstum der Stoer-Kovarianz. Also
berechnen wir genau das geschlossen, mit der FELD-Jacobi J (aus der
Statistik-Probe, kein Netz-Rollout):

    C_0 = I
    C_{k+1} = J C_k J^T + I          (getriebene Lyapunov-Iteration)
    FTLE ~ (1/(2K)) * ln lam_max(C_K)

Bei normalen J -> Wachstumsrate -> spektrale Kante (T0). Bei
nicht-normalen J -> transientes Wachstum ist EXAKT in J C J^T
enthalten (die Kovarianz sieht die nicht-orthogonalen Richtungen).
K ist der einzige Hyperparameter (Fensterlaenge), auf Train gewaehlt.

Praediktoren (held-out):
  Q1: T0 pur (Baseline)
  Q2: Kovarianz-FTLE (K auf Train aus Grid)
  Q3: max(T0, Kovarianz) — fit-frei ausser K

Zusaetzlich: (a) GRU-Universalitaet diesmal RICHTIG: GRUs auf hohe
kappa treiben (kappa-Drift aus F16 +30% Steps), dann Gesetz testen;
(b) Designed-Netze nach 200 Trainings-Steps: ueberlebt das Gesetz
echtes Training?

ZIEL: held-out |err| < 0.10, Vorzeichen > 85% mit Q2/Q3.
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
from cgfn.continuous import make_continuous_batch


@torch.no_grad()
def field_J(model, x_probe):
    d = sech2_field_iteration(model, x_probe)
    H = model.hidden_dim
    return -torch.eye(H) / model.tau + model.W * d.unsqueeze(0)


@torch.no_grad()
def covarianz_ftle(J, K):
    """Kovarianz-FTLE mit Spur-Renormierung (Analogon zur Benettin-QR-Renorm):
    C <- J C J^T + I; Akkumulation von log tr(C); C /= tr(C).
    Fuer grosse K dominiert lam_max die Spur => FTLE ~ (1/2K) sum log tr.
    Numerisch stabil gegen Kovarianz-Explosion (F19-Lauf-1-Crash: eigvalsh
    auf unrenormiertem C kippte bei ~e^60 in float32)."""
    H = J.shape[0]
    J = J.to(torch.float64)
    C = torch.eye(H, dtype=torch.float64)
    I = torch.eye(H, dtype=torch.float64)
    acc = 0.0
    for _ in range(K):
        C = J @ C @ J.T + I
        tr = C.trace().clamp_min(1e-300)
        acc += torch.log(tr).item()
        C = C / tr
    return 0.5 * acc / K


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--hidden-dims", default="32,16")
    p.add_argument("--lams", default="0.8,1.2")
    p.add_argument("--s-values", default="0.0,0.4,0.8,1.2,1.6,2.0")
    p.add_argument("--K-grid", default="2,4,8,16,32")
    p.add_argument("--out", default="results/f19_law_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    hiddens = [int(h) for h in args.hidden_dims.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]
    kgrid = [int(v) for v in args.K_grid.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 110)
    lines.append("F19 KOVARIANZ-PROPAGATIONS-GESETZ: C <- J C J^T + I mit Feld-J (fit-frei, K aus Train)")
    lines.append("=" * 110)

    # ── Daten sammeln ──
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
                    J = field_J(m, x)
                    t0 = torch.linalg.eigvals(J).real.max().item()
                    covs = {K: covarianz_ftle(J, K) for K in kgrid}
                    rows.append({"t0": t0, "ben": ben, "kap": kap, "cov": covs})
    train = [r for i, r in enumerate(rows) if i % 2 == 0]
    test = [r for i, r in enumerate(rows) if i % 2 == 1]

    # K auf Train waehlen (fit-frei bzgl. Test)
    best_K, best_e = None, None
    for K in kgrid:
        e = sum(abs(r["cov"][K] - r["ben"]) for r in train) / len(train)
        if best_e is None or e < best_e:
            best_K, best_e = K, e
    lines.append(f"K gewaehlt auf Train: K={best_K} (train |err| {best_e:.3f})")
    lines.append(f"n={len(rows)} (train {len(train)}, held-out {len(test)})")
    lines.append("")

    def stats(sub, fn):
        errs = [abs(fn(r) - r["ben"]) for r in sub]
        signs = sum(1 for r in sub if (fn(r) > 0) == (r["ben"] > 0))
        return sum(errs) / len(errs), signs / len(sub)

    q1 = stats(test, lambda r: r["t0"])
    q2 = stats(test, lambda r: r["cov"][best_K])
    q3 = stats(test, lambda r: max(r["t0"], r["cov"][best_K]))

    lines.append(f"{'Praediktor':>28s} {'|Fehler|':>9s} {'Vorzeichen':>11s}")
    lines.append(f"{'Q1 T0 pur (Baseline)':>28s} {q1[0]:9.3f} {100*q1[1]:10.0f}%")
    lines.append(f"{'Q2 Kovarianz (K aus Train)':>28s} {q2[0]:9.3f} {100*q2[1]:10.0f}%")
    lines.append(f"{'Q3 max(T0, Kovarianz)':>28s} {q3[0]:9.3f} {100*q3[1]:10.0f}%")
    lines.append("")
    lines.append("--- Held-out nach kappa-Band ---")
    for lo, hi in ((1.0, 5.0), (5.0, 10.0), (10.0, 99.0)):
        sub = [r for r in test if lo <= r["kap"] < hi]
        if not sub:
            continue
        e1 = sum(abs(r["t0"] - r["ben"]) for r in sub) / len(sub)
        e2 = sum(abs(r["cov"][best_K] - r["ben"]) for r in sub) / len(sub)
        lines.append(f"kappa [{lo:.0f},{hi:.0f}): n={len(sub):3d} | T0 {e1:.3f} | Kov {e2:.3f}")

    # ── (a) GRU-Universalitaet: GRUs auf hohe kappa treiben ──
    lines.append("")
    lines.append("--- (a) GRU auf hohe kappa treiben (800 Steps, kappa-Drift aus F16) + Gesetz ---")
    lines.append("(Kovarianz-Gesetz braucht Feld-J: nur fuer CT verfuegbar — hier testen wir,")
    lines.append(" ob GRU mit kappa>=6 UEBERHAUPT ein Treib-Defizit D zeigt [F18d-Luecke schliessen])")
    from cgfn.driven_lyap import driven_measures
    gen = torch.Generator().manual_seed(777)
    x0 = torch.zeros(4, 48, 1)
    xh = torch.randn(4, 48, 1, generator=gen) * 0.5
    for seed in (0, 1):
        torch.manual_seed(seed)
        g = make_zoo_model("gru", hidden_dim=32, seed=seed)
        opt = torch.optim.Adam(g.parameters(), lr=5e-3)
        torch.manual_seed(seed + 1000)
        for _ in range(800):
            xb, yb = make_continuous_batch("sinus_memory", 16, 16)
            out, _ = g(xb)
            loss = torch.nn.MSELoss()(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        W = g.core.weight_hh.detach()
        sv = torch.linalg.svdvals(W)[0].item()
        lam = max(torch.linalg.eigvals(B).abs().max().item() for B in W.chunk(3, dim=0))
        kap = sv / max(abs(lam), 1e-9)
        ben0 = driven_measures(g, x0)["lyap_ben"]
        benh = driven_measures(g, xh)["lyap_ben"]
        D = benh - ben0
        reg = "B (kappa>=6)" if kap >= 6 else "A"
        lines.append(f"gru seed {seed}: kappa={kap:.2f} ({reg}) | ben0 {ben0:+.3f} ben.5 {benh:+.3f} | D={D:+.3f}")

    # ── (b) Gesetz nach Training: designed + 300 Steps ──
    lines.append("")
    lines.append("--- (b) ueberlebt das Gesetz echtes Training? designed + 300 Steps ---")
    lines.append(f"{'kap':>6s} {'ben':>8s} {'T0':>8s} {'Kov':>8s}")
    e_t0_tr, e_kov_tr, ntr = 0.0, 0.0, 0
    for s in (1.2, 1.6, 2.0):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            m = make_designed_model(32, 1.0, 1.2, s, seed, task_dim=0)
            opt = torch.optim.Adam(m.parameters(), lr=2e-3)
            torch.manual_seed(seed + 1000)
            for _ in range(300):
                xb, yb = make_continuous_batch("sinus_memory", 16, 16)
                out, _ = m(xb)
                loss = torch.nn.MSELoss()(out, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            x = drive_input(4, 48, 0.5)
            ben = benettin_ftle_max(m, x)
            J = field_J(m, x)
            t0 = torch.linalg.eigvals(J).real.max().item()
            kov = covarianz_ftle(J, best_K)
            kap = spectral_edge(m) / max(abs(float(eig_max(m))), 1e-9)
            lines.append(f"{kap:6.2f} {ben:+8.3f} {t0:+8.3f} {kov:+8.3f}")
            e_t0_tr += abs(t0 - ben); e_kov_tr += abs(kov - ben); ntr += 1
    lines.append(f"nach Training: T0 |err| {e_t0_tr/ntr:.3f} | Kovarianz |err| {e_kov_tr/ntr:.3f}")

    lines.append("")
    lines.append("Lesart: Q2/Q3 unter 0.10/85% => GESETZ GESCHLOSSEN als Kovarianz-")
    lines.append("Iteration (fit-frei ausser Fenstlaenge K). GRU-Zeilen zeigen, ob")
    lines.append("Regime-B-Defizit auf getriebener Praxisarchitektur existiert.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
