"""
F12 — Nicht-Normalitaet als Speicher-Ressource: der Weg vom Verstaendnis
zum Bau-Rezept (ROADMAP: "wie nutzen wir das, um starke KI zu bauen?").

F11 zeigte: Spektralregeln messen die ungetriebene Strecke; die WAHRE
getriebene Kante (Benettin) liegt tiefer, und SVD-FTLE war bei kappa>>1
ein Messartefakt. F12 dreht die Frage um: Wenn transientes Wachstum
(nicht-normales W) die SVD-Messung aufblaest — ist dasselbe transiente
Wachstum dann eine SPEICHER-RESSOURCE? Hypothese:

    Bei festem sub-kritischem lam_max steigt die Memory Capacity (MC)
    mit kappa (nicht-Normalitaet) — langes Input-Gedaechtnis OHNE Chaos.

Design (Reservoir-Setting, saubere Attribution):
  - Designed W (Schur, s_nonnormal als kappa-Drehregler), W FIX
  - lam in {0.6, 1.0, 1.4} x tau=1.0: klar unter/auf/ueber der
    LINEAREN Kante — aber F11 lehrt: entscheidend ist die GETRIEBENE.
  - Pro Netz: MC (Jaeger, Ridge-Readout, weisses Rauschen — dieselben
    Input-Statistiken wie beim Benettin-Rollout), Benettin-FTLE
    (getrieben), SVD-FTLE, kappa, RK2-Audit.

F12-Fragen:
  (1) MC vs. kappa bei fixem lam — Ressourcen-These (steigt es?)
  (2) MC vs. getriebener Kante — liegt das MC-Optimum nahe der
      BENETTIN-Kante (nicht der Spektral-Kante)?
  (3) Chaos-Zahl: SVD-FTLE koennte "Ressource" vortaeuschen (Wachstum,
      das nicht dekodierbar ist) — korrigiert Benettin das Bild?
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max, stability_audit
from cgfn.cavity import eig_max, spectral_edge
from cgfn.design import make_designed_model
from cgfn.ftle import ftle_summary
from cgfn.memory_capacity import memory_capacity


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--taus", default="1.0")
    p.add_argument("--lams", default="0.6,1.0,1.4")
    p.add_argument("--s-values", default="0.0,0.5,1.0,2.0")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=16)
    p.add_argument("--out", default="results/f12_resource_results.txt")
    args = p.parse_args()

    seeds = [int(v) for v in args.seeds.split(",")]
    taus = [float(v) for v in args.taus.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 120)
    lines.append("F12 RESSOURCEN-EXPERIMENT: Memory Capacity vs. Nicht-Normalitaet kappa — langes Gedaechtnis ohne Chaos?")
    lines.append(f"seeds={seeds} taus={taus} lams={lams} s={s_vals} hidden={args.hidden_dim} max_lag={args.max_lag}")
    lines.append("MC = sum_{lag=1..L} corr^2(Ridge(z_t), x_{t-lag}) — Jaeger-Standard, deterministisch, disjunktes Testset")
    lines.append("=" * 120)
    lines.append(f"{'config':>24s} {'lam':>5s} {'kappa':>6s} {'MC':>6s} {'benettin':>9s} {'svdftle':>8s} "
                 f"{'tau*lam':>8s} {'svJ0dt':>7s} {'rk2ok':>5s}")

    rows = []  # (lam, s, kappa, mc, ben, svd, tau_lam, rk2ok)

    for lam in lams:
        for s_nn in s_vals:
            for tau in taus:
                mcs, bens, svds, kappas = [], [], [], []
                for seed in seeds:
                    torch.manual_seed(seed)
                    model = make_designed_model(args.hidden_dim, tau, lam,
                                                 s_nn, seed, task_dim=0)
                    sv_w = spectral_edge(model)
                    lam_w = eig_max(model)
                    kappa = sv_w / max(abs(lam_w), 1e-9)

                    # MC-Probe: weisses Rauschen, dieselben Statistiken
                    # wie beim Benettin-Rollout unten (input_scale=0.5).
                    mc, _ = memory_capacity(model, max_lag=args.max_lag,
                                            input_scale=0.5)

                    # Getriebene Messgeraete auf identischem Antrieb.
                    x = torch.randn(4, 48, 1) * 0.5
                    _, states = model(x, None)
                    ftle_svd = ftle_summary(model, states)["ftle_max"]
                    ftle_ben = benettin_ftle_max(model, x)

                    mcs.append(mc); bens.append(ftle_ben)
                    svds.append(ftle_svd); kappas.append(kappa)

                audit = stability_audit(model)
                mc_mean = sum(mcs) / len(mcs)
                ben_mean = sum(bens) / len(bens)
                svd_mean = sum(svds) / len(svds)
                kap_mean = sum(kappas) / len(kappas)
                rows.append((lam, s_nn, kap_mean, mc_mean, ben_mean,
                             svd_mean, tau * lam, audit["rk2_ok"]))
                line = (f"{'lam=%.2f s=%.1f' % (lam, s_nn):>24s} {lam:5.2f} "
                         f"{kap_mean:6.2f} {mc_mean:6.2f} {ben_mean:+9.4f} "
                         f"{svd_mean:+8.3f} {tau * lam:8.2f} "
                         f"{audit['svJ0_times_dt']:7.2f} {str(audit['rk2_ok']):>5s}")
                lines.append(line)
                print(line, flush=True)

    # --- Aggregate: Ressourcen-These (1) MC vs kappa bei fixem lam ---
    lines.append("")
    lines.append("--- (1) MC vs. kappa bei fixem lam (Ressourcen-These) ---")
    for lam in lams:
        sub = [r for r in rows if r[0] == lam and r[7]]  # nur rk2-gueltige
        if len(sub) < 2:
            continue
        kaps = torch.tensor([r[2] for r in sub])
        mcs = torch.tensor([r[3] for r in sub])
        tag = f"lam={lam:4.2f}: MC {mcs[0]:.2f} (kappa={kaps[0]:.2f}) -> {mcs[-1]:.2f} (kappa={kaps[-1]:.2f})"
        if kaps.numel() > 2 and kaps.std() > 1e-6 and mcs.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([kaps, mcs]))[0, 1].item()
            tag += f"  corr(kappa, MC)={corr:+.2f}"
        lines.append(tag)

    # --- (2) Liegt das MC-Optimum nahe der GETRIEBENEN Kante? ---
    lines.append("")
    lines.append("--- (2) MC vs. getriebener Kante (Benettin-FTLE ~ 0?) ---")
    for s_nn in s_vals:
        sub = [r for r in rows if r[1] == s_nn and r[7]]
        if not sub:
            continue
        best = max(sub, key=lambda r: r[3])
        lines.append(f"s={s_nn:3.1f}: bestes MC={best[3]:.2f} bei lam={best[0]:.2f} "
                     f"(Benettin {best[4]:+.3f}, Spektral-Rand tau*lam={best[6]:.2f})")

    lines.append("")
    lines.append("--- (3) SVD-Chaos-Illusion: SVD-FTLE > 0 waehrend Benettin < 0 ---")
    illu = [r for r in rows if r[7] and r[5] > 0 and r[4] < 0]
    lines.append(f"{len(illu)}/{len([r for r in rows if r[7]])} rk2-gueltige Konfigurationen: "
                 f"SVD-FTLE vortaeuscht Expansions-Ressource, Benettin sagt subkritisch.")
    lines.append("")
    lines.append("LESEAT: (1) steigt MC mit kappa => transientes Wachstum ist dekodierbare")
    lines.append("Speicherressource, kein Messfehler. (2) MC-Optimum nahe Benettin-Kante =>")
    lines.append("getriebene (nicht spektrale) Kante ist die Bau-Anweisung. (3) hoher Anteil =>")
    lines.append("SVD-basierte Reservoir-Regeln uebertreiben die nutzbare Ressource.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
