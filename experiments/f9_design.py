"""
F9 — Generatives Pendant zum Spectral Ruler: Kritikalitaet by construction.

Die Umkehrung von F8: Statt den Ruler auf gewachsene Gewichte anzuwenden,
KONSTRUIEREN wir W mit vorgeschriebenem Spektrum (Schur-Bau, s.
cgfn.design) und pruefen, ob der Ruler das Verhalten VORHERSAGT,
bevor es gemessen wird — Validierungs-Dreieck: Messung/Theorie/Konstruktion.

Pro Konfiguration (lam_max x s_nonnormal):
  1. Ruler-Vorhersage OHNE Rollout: tau_pred_eig = 1/lam_max (per Bau exakt),
     tau_pred_sv = 1/(sv_max*sqrt(Q)).
  2. Empirisches FTLE-Crossing: tau-Grid, FTLE pro tau, Vorzeichenwechsel
     interpoliert => tau_cross (das "wahre" Kritikalitaets-Mass).
  3. Tau*(loss): Training auf sinus_memory pro tau (freeze_w! nur
     U/Readout lernen => Spektrum bleibt ueber Training gueltig),
     Argmin des Losses.

Kernfragen:
  - Naehert tau_cross eher tau_pred_eig (normale Matrix, s=0) oder
    tau_pred_sv (starke Nicht-Normalitaet, s>>1)?
  - Sitzt das Loss-Optimum (Memory-Task) auf dem Crossing (F5-Befund
    tau*(loss)=tau*(|FTLE|) generalisiert)?
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.continuous import make_continuous_batch
from cgfn.design import make_designed_model
from cgfn.ftle import ftle_summary
from cgfn.ruler import spectral_report


def measure_ftle(lam, s, tau, hidden_dim, seed, batch=16, seq_len=24):
    model = make_designed_model(hidden_dim, tau, lam, s, seed, task_dim=0)
    x, _ = make_continuous_batch("sinus_memory", batch, seq_len)
    _, states = model(x)
    return ftle_summary(model, states)["ftle_max"]


def tau_crossing(lam, s, taus, hidden_dim, seed):
    """FTLE ueber tau-Grid; interpoliertes Crossing (Vorzeichenwechsel)."""
    pts = [(tau, measure_ftle(lam, s, tau, hidden_dim, seed)) for tau in taus]
    for (t1, f1), (t2, f2) in zip(pts, pts[1:]):
        if (f1 <= 0.0 <= f2) or (f2 <= 0.0 <= f1):
            if f2 == f1:
                return t1, pts
            frac = (0.0 - f1) / (f2 - f1)
            return t1 + frac * (t2 - t1), pts
    return None, pts


def train_tau(lam, s, tau, hidden_dim, seed, steps, lr=2e-3, batch=16, seq_len=16):
    """Training auf sinus_memory mit freeze_w=True (Reservoir-Setting)."""
    model = make_designed_model(hidden_dim, tau, lam, s, seed, task_dim=0)
    assert not model.W.requires_grad, "freeze_w muss gesetzt sein (Reservoir-Setting)"
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossfn = torch.nn.MSELoss()
    loss = None
    for _ in range(steps):
        x, y = make_continuous_batch("sinus_memory", batch, seq_len)
        out, _ = model(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lams", default="0.8,1.15,1.6")
    p.add_argument("--svals", default="0.0,1.0,3.0")
    p.add_argument("--taus", default="0.4,0.7,1.0,1.5,2.2,3.2")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--out", default="results/f9_design_results.txt")
    args = p.parse_args()

    lams = [float(v) for v in args.lams.split(",")]
    svals = [float(v) for v in args.svals.split(",")]
    base_taus = [float(v) for v in args.taus.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 116)
    lines.append("F9 GENERATIVES PENDANT: Designed W (Schur-Bau) — sagt der Ruler das Verhalten voraus?")
    lines.append(f"lam_max={lams}  s_nonnormal={svals}  taus={base_taus}  seeds={seeds}  "
                 f"steps={args.train_steps}  hidden={args.hidden_dim}")
    lines.append("Vorhersage per Bau: tau_pred_eig = 1/lam_max (exakt); Frage: folgt das gemessene")
    lines.append("FTLE-Crossing der Eigenwert- oder der SVD-Kante — und sitzt tau*(loss) darauf?")
    lines.append("=" * 116)
    lines.append(f"{'lam':>5s} {'s':>4s} {'kappa':>6s} {'tEig':>6s} {'tSv':>6s} "
                 f"{'tCross':>7s} {'|dEig|':>7s} {'|dSv|':>7s} {'tau*loss':>8s} {'|star-X|':>8s} {'winner':>7s}")

    errs_eig, errs_sv, errs_star = [], [], []
    wins = {"eig": 0, "sv": 0, "none": 0}

    for lam in lams:
        for s in svals:
            # Ruler-Vorhersage (states=None => Ursprungs-Naeherung, kein Rollout)
            model0 = make_designed_model(args.hidden_dim, 1.0, lam, s, seeds[0], task_dim=0)
            rep = spectral_report(model0)
            tau_pred_eig, tau_pred_sv, kappa = rep["tau_pred_eig"], rep["tau_pred_sv"], rep["kappa"]

            # Crossing auf erweitertem Grid (Vorhersagen mit in den Grid)
            grid = sorted(set(base_taus) | {round(tau_pred_eig, 4), round(tau_pred_sv, 4)})
            grid = [t for t in grid if 0.2 <= t <= 6.0]
            cross, pts = tau_crossing(lam, s, grid, args.hidden_dim, seeds[0])

            # tau*(loss): Argmin ueber Basis-Grid, gemittelt ueber Seeds
            star_by_seed = []
            for seed in seeds:
                losses = [(tau, train_tau(lam, s, tau, args.hidden_dim, seed, args.train_steps))
                          for tau in base_taus]
                star_by_seed.append(min(losses, key=lambda kv: kv[1])[0])
            tau_star = sum(star_by_seed) / len(star_by_seed)

            if cross is None:
                d_eig = d_sv = d_star = float("nan")
                wins["none"] += 1
                sign = "sub" if pts[-1][1] < 0 else "sup"
                lines.append(f"{lam:5.2f} {s:4.1f} {kappa:6.2f} {tau_pred_eig:6.3f} {tau_pred_sv:6.3f} "
                              f"{'kein Wechsel (' + sign + ')':>18s}   tau*loss={tau_star:.2f}")
                continue
            d_eig, d_sv = abs(cross - tau_pred_eig), abs(cross - tau_pred_sv)
            d_star = abs(tau_star - cross)
            errs_eig.append(d_eig); errs_sv.append(d_sv); errs_star.append(d_star)
            winner = "eig" if d_eig <= d_sv else "sv"
            wins[winner] += 1
            line = (f"{lam:5.2f} {s:4.1f} {kappa:6.2f} {tau_pred_eig:6.3f} {tau_pred_sv:6.3f} "
                    f"{cross:7.3f} {d_eig:7.3f} {d_sv:7.3f} {tau_star:8.3f} {d_star:8.3f} {winner:>7s}")
            lines.append(line)
            print(line, flush=True)

    lines.append("")
    lines.append("--- Aggregat ---")
    if errs_eig:
        n = len(errs_eig)
        lines.append(f"mean |cross - tau_pred_eig| = {sum(errs_eig)/n:.3f}   "
                     f"mean |cross - tau_pred_sv| = {sum(errs_sv)/n:.3f}")
        lines.append(f"Sieger pro Konfiguration: eig={wins['eig']}, sv={wins['sv']}, kein Crossing={wins['none']}")
        lines.append(f"mean |tau*(loss) - tau_cross| = {sum(errs_star)/n:.3f}")
    lines.append("")
    lines.append("LESEAT: Bei s=0 (kappa~1) MUSS tau_cross ~ tau_pred_eig sein (Konsistenz-Check).")
    lines.append("Wenn tau_cross mit wachsendem s zur SVD-Kante (tau_pred_sv) wandert,")
    lines.append("legt die SVD-basierte FTLE-Messung die nicht-normale Kante frei —")
    lines.append("dann ist kappa der Schalter zwischen den Regeln: Ruler = f(kappa).")
    lines.append("Wenn tau*(loss) ~ tau_cross bleibt, generalisiert der F5-Befund")
    lines.append("(Memory-Optimum auf Kritikalitaet) auf designed Netzwerke.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
