"""
F15 — Doom-Kriterium: Welches Messgeraet im Reglerkreis ist ein System
zum Scheitern verurteilt — VORHERSAGBAR, ohne Training?

F14-Befund zerpflueckt: Der SVD-Regler-Fail war eine dreistellige Gleichung:
  (1) Systemeigenschaft kappa -> Artefakt A(kappa) = FTLE_svd - FTLE_ben
      (transientes Wachstum blaeht das Kurzfenster auf; F11/F13: monoton).
  (2) Regler-Polaritaet + Gain: Artefakt wirkt als Dauer-Fehler => tau
      driftet systematisch (hier: runter, Richtung Clamp).
  (3) Clamp-Wahl: tau_min kann das numerisch sichere Regime verlassen
      (RK2: sv(J0)*dt < 2; bei tau=0.05 ist 1/tau=20 => ausserhalb).

F15 macht daraus einen VORHERSAGE-Test: Auf EINGEFRORENEN designed Netzen
(kappa-Sweep per s_nonnormal) wird der Regler-Drift OHNE Training simuliert
(nur Mess-Update-Schleife). Kandidaten-Kriterium: Ein System ist "verurteilt",
wenn der gemessene Artefakt den Regler bis zum Clamp treibt — und der
Benettin-Regler auf demselben Netz stabil bleibt (A/B-Verifikation).

Doom-Masse:
  drift = |log(tau_end / tau_start)|      (wie weit lief tau weg)
  doomed = tau-toerer-Clamp erreicht ODER drift > DOOM_DRIFT
Ausgabe: kappa vs. Artefakt vs. Doom beider Regler + Vorhersage-Tabelle.
"""
import argparse
import math
import os
from pathlib import Path

import torch

from cgfn.design import make_designed_model
from cgfn.cavity import eig_max, spectral_edge
from cgfn.ftle import ftle_summary
from cgfn.regulator import FtleRegulator, BenettinRegulator
from cgfn.benettin import benettin_ftle_max, stability_audit

DOOM_DRIFT = 1.0  # tau mehr als Faktor e vom Start => ausserhalb jedes
                  # sinnvollen Regelbereichs ("verurteilt")


def regulator_doom(model, regulator_cls, target, gain, n_updates,
                   probe_x):
    """Simuliert den Regelkreis auf dem EINGEFRORENEN Netz: n Mess-Updates.

    Rueckgabe: (tau_end, drift, doomed, log der tau-Trajektorie).
    Kein Training — der Effekt ist rein Regelkreis + Messgeraet.
    """
    model.tau = 1.0
    reg = regulator_cls(target=target, gain=gain, interval=1)
    tau0 = float(model.tau)
    taus = [tau0]
    for i in range(n_updates):
        reg.update(i, model, None)
        taus.append(float(model.tau))
    tau_end = taus[-1]
    drift = abs(math.log(max(tau_end, 1e-9) / tau0))
    doomed = (tau_end <= reg.tau_min * 1.01 or tau_end >= reg.tau_max * 0.99
              or drift > DOOM_DRIFT)
    return tau_end, drift, doomed, taus


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--lams", default="0.6")
    p.add_argument("--s-values", default="0.0,0.5,1.0,2.0,3.0")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--gain", type=float, default=0.8)
    p.add_argument("--target", type=float, default=-0.2)
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--out", default="results/f15_doom_results.txt")
    args = p.parse_args()

    seeds = [int(v) for v in args.seeds.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 110)
    lines.append("F15 DOOM-KRITERIUM: Artefakt A(kappa) sagt Regler-Doom vorher — eingefrorene Netze, kein Training")
    lines.append(f"seeds={seeds} lams={lams} s={s_vals} H={args.hidden_dim} gain={args.gain} target={args.target} updates={args.updates}")
    lines.append("A(kappa) = FTLE_svd - FTLE_ben (gemessen auf derselben Treib-Probe); Doom: tau-Clamp erreicht oder |log(tau_end/tau_0)| > %.1f" % DOOM_DRIFT)
    lines.append("=" * 110)
    lines.append(f"{'lam':>5s} {'s':>4s} {'kappa':>6s} {'A(kap)':>7s} {'svdtau':>7s} {'svd_drift':>9s} {'svDOOM':>6s} "
                 f"{'bentau':>7s} {'ben_drift':>9s} {'benDOOM':>7s} {'rk2ok@clamp':>11s}")

    rows = []
    for lam in lams:
        for s_nn in s_vals:
            for seed in seeds:
                torch.manual_seed(seed)
                model = make_designed_model(args.hidden_dim, 1.0, lam, s_nn,
                                             seed, task_dim=0)
                kappa = (spectral_edge(model)
                         / max(abs(float(eig_max(model))), 1e-9))
                # Artefakt auf identischer Treib-Probe
                gen = torch.Generator().manual_seed(20261013)
                x = torch.randn(4, 48, 1, generator=gen) * 0.5
                _, states = model(x, None)
                ftle_svd = float(ftle_summary(model, states)["ftle_mean"])
                ftle_ben = benettin_ftle_max(model, x)
                artifact = ftle_svd - ftle_ben

                # Beide Regler auf demselben eingefrorenen Netz
                st, sd, sdoom, _ = regulator_doom(
                    model, FtleRegulator, args.target, args.gain,
                    args.updates, x)
                bt, bd, bdoom, _ = regulator_doom(
                    model, BenettinRegulator, args.target, args.gain,
                    args.updates, x)

                # Ist der SVD-Clamp numerisch unsicher? (RK2-Fenster)
                model.tau = 0.05
                clamp_audit = stability_audit(model)
                model.tau = 1.0

                rows.append((lam, s_nn, kappa, artifact, st, sd, sdoom,
                             bt, bd, bdoom))
                line = (f"{lam:5.2f} {s_nn:4.1f} {kappa:6.2f} {artifact:+7.3f} "
                        f"{st:7.2f} {sd:9.3f} {str(sdoom):>6s} "
                        f"{bt:7.2f} {bd:9.3f} {str(bdoom):>7s} "
                        f"{str(clamp_audit['rk2_ok']):>11s}")
                lines.append(line)
                print(line, flush=True)

    # --- Vorhersage-Qualitaet: sagt der Artefakt Doom voraus? ---
    lines.append("")
    lines.append("--- Vorhersage: Artefakt A vs. SVD-Regler-Doom ---")
    art = torch.tensor([r[3] for r in rows])
    doo = torch.tensor([1.0 if r[6] else 0.0 for r in rows])
    n_doom = int(doo.sum().item())
    if n_doom > 0 and n_doom < len(rows) and art.std() > 1e-6:
        corr = torch.corrcoef(torch.stack([art, doo]))[0, 1].item()
        lines.append(f"corr(A, doomed) = {corr:+.2f} — Doom-Faelle: {n_doom}/{len(rows)}")
    else:
        lines.append(f"Doom-Faelle: {n_doom}/{len(rows)} (Korrelation nur bei gemischtem Ausgang informativ)")
    if n_doom > 0:
        a_min_doom = min(r[3] for r in rows if r[6])
        a_max_ok = max(r[3] for r in rows if not r[6])
        lines.append(f"kleinstes A unter Doom: {a_min_doom:+.3f} | groesstes A ohne Doom: {a_max_ok:+.3f} "
                     f"=> Schwellen-Kandidat in diesem Bereich" if a_min_doom <= a_max_ok else
                     f"=> lueckenfreie Trennung! Schwellen: A > {a_max_ok:+.3f}")
    lines.append("")
    lines.append(f"Benettin-Regler: {int(sum(1 for r in rows if r[9]))}/{len(rows)} doomed "
                 f"(Erwartung: 0 — artefaktfreies Messgeraet kennt keinen Phantom-Fehler)")
    lines.append("")
    lines.append("Lesart: (1) Doom nur bei hohem A(kappa) und 0 beim Benettin-Regler => das")
    lines.append("Kriterium ist das MESSGERAET x SYSTEM-KAPPA, nicht die Regel-Idee — Systeme")
    lines.append("sind VORHERSAGBAR verurteilt: Canary = Artefakt-Probe auf eingefrorenem Netz,")
    lines.append("kein Training noetig. (2) Clamp 0.05 mit rk2ok=False zeigt: der Tod ist teils")
    lines.append("NUMERISCH — Tau-Grenzen muessen das RK2-Fenster respektieren.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
