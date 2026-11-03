"""
F24 — TRANSIENTES WACHSTUM ALS MEMORY-RESSOURCE: der Kollisionstest.

Literatur-Lage (Tiefenanalyse I-IV):
  Ganguli/Huh/Sompolinsky 2008: Fisher-Memory-Kurve — transiente
    nicht-normale Amplifikation ist SNR-Speicher, bis O(N).
  Orhan/Pitkow 2019 (1905.13715): sequentiell nicht-normale RNNs
    schlagen orthogonale deutlich — kappa=1 toetet die Ressource.
  Schuecker/Hellgoth/Geist 2016 (1603.01880): MC-Peak im 'expandiv,
    nicht-chaotischen' Regime; getriebener Lyapunov < 0 = Echo-State.
  Unsere F12: MC-Optimum bei getriebenem ben ~ -0.33 (subkritisch!).
  Unsere F18c: Regime B (kappa>=5) ist genau das lokal-expandive,
    asymptotisch stabile Regime (T0+Transienten-Korrektur).

HYPOTHESE (F24, scharf): Bei FESTER getriebener Stabilitaet (Benettin
auf Ziel lam_target eingestellt per lam-Bisektion) STEIGT die Memory
Capacity mit kappa. Dann ist Nicht-Normalitaet eine freie Resource:
mehr Speicher OHNE mehr Chaos — die Bau-Anweisung fuer subkritische
Netze mit langem Gedaechtnis.

DESIGN:
  - make_designed_model (Schur, s als kappa-Drehregler), W FIX.
  - Pro (target, s, seed): lam-Bisektion (14 Iter.) auf
    benettin_ftle_max(m, drive) ~ target. Zwei Targets:
    -0.30 (F12-Ressourcenfenster) und -0.05 (Naehe Kante, Kontrolle).
  - Messung: kappa, t0, log1p(K), log1p(pseudo), ben, MC (Jaeger).

VORHERSAGE: MC steigt mit kappa bei beiden Targets (Regime-B-These).
Gegenbild: MC flach/faellt => Ressource ist Artefakt der lam-Kovariation
(F12-Kritik) und orthogonale Architekturen waeren gerechtfertigt.

Run-35-Erweiterung (VOR dem Lauf registriert, 2026-11-03):
  Hintergrund: F27 Run 34/35 — Adam erodiert kappa_T(W) von ~5.6e7
  auf Endpunkte ~165..380 (5 Groessenordnungen) und VERBESSERT dabei
  die Memory-Regression (loss 0.43 -> 0.15). F31-P4: bei hohem s
  explodiert der Exzess E (mean 1.17 bei s=0 vs 9.7 bei s=0.8) —
  extreme Nicht-Normalitaet ist Rauschverstaerkung, kein SNR-Speicher.
  SEKUNDAER-VORHERSAGE S1: MC(kappa) hat ein INNERES Optimum. Ein
  monotoner Anstieg kann im Extrembereich nicht gelten — dort frisst
  der Exzess den SNR-Gewinn. (Traefte MC bis s=2.0 monoton zu, waere
  S1 gefallen UND der F27-Befund eroerterungsbeduerftig.)
  SEKUNDAER-VORHERSAGE S2: Das MC-optimale kappa_T-Band liegt in der
  Naehe der F27-Adam-Endpunkte (kappa_T ~ 100..500, Run 34:
  165.8/189.5/380.7). Dann haette der Optimierer widerwillig genau
  in die Ressourcen-Landschaft 'gefahren', die F24 vermisst.
  ZOO-ERWEITERUNG: kappa_T, kappa_cond, delta_H auf W (dasselbe
  Instrument wie F27) plus Exzess E auf dem linearisierten Feld
  (F31-Instrument). Die Auswertung sagt, WELCHES kappa die Memory-
  Ressource traegt — F31/F27 haben gezeigt, dass die Zoo-Vertreter
  verschiedenen Geschichten erzaehlen.
  DIE ORIGINAL-SCHWELLE (corr > +0.5, Monotonie >= 3/4 bei -0.30)
  BLEIBT UNANGETASTET: sie faellt oder sie steht. S1/S2 sind zusaetz-
  liche registrierte Messungen, kein Rettungsanker.
"""
import argparse
import math
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max
from cgfn.cavity import eig_max, spectral_edge
from cgfn.design import make_designed_model
from cgfn.field import (drive_input, sech2_field_iteration,
                        kreiss_constant, transient_bound_pseudo)
from cgfn.memory_capacity import memory_capacity
from cgfn.resource import kappa_zoo, normal_excess

IS = 0.5  # input_scale, konsistent mit drive_input- und MC-Aufrufen

# F27-Referenzband (Run 34, Adam, seeds 0/1/2): kappa_T-Endpunkte.
F27_ADAM_KT = (165.811, 189.452, 380.668)
F27_ADAM_BAND = (100.0, 500.0)


@torch.no_grad()
def lin_features(model, x_probe):
    d = sech2_field_iteration(model, x_probe)
    H = model.hidden_dim
    J = -torch.eye(H) / model.tau + model.W * d.unsqueeze(0)
    t0 = torch.linalg.eigvals(J).real.max().item()
    k = kreiss_constant(J)
    tb = transient_bound_pseudo(J)
    tb = 10.0 if tb == float("inf") else tb
    return t0, torch.log1p(torch.tensor(k)).item(), torch.log1p(torch.tensor(tb)).item()


def field_excess(model, x_probe):
    """Exzess E auf dem linearisierten getriebenen Feld (F31-Instrument).

    E = Tr(Sigma)/Tr(Sigma_norm) mit J aus sech2-Felditeration und
    Q = IS^2 * Delta * B B^T (wie dmft_fixed_point). None bei
    instabilem J oder Solver-Fehler — ehrlich statt garbage.
    """
    try:
        d = sech2_field_iteration(model, x_probe)
        H = model.hidden_dim
        J = (-torch.eye(H) / model.tau + model.W * d.unsqueeze(0)).to(torch.float64)
        if torch.linalg.eigvals(J).real.max().item() >= -1e-9:
            return None
        B = model.U.weight.detach().to(torch.float64)
        Delta = float(model.substeps) * float(model.dt)
        Q = (IS ** 2) * Delta * (B @ B.T)
        E, _S, _Sn = normal_excess(J, Q)
        return float(E)
    except Exception:
        return None


def tune_lam(hidden, tau, s, seed, target, xdr, lo=0.1, hi=2.5, iters=14):
    """Bisektion: spectral lam so, dass getriebener Benettin ~ target."""
    lam, model, ben = 0.5 * (lo + hi), None, None
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        torch.manual_seed(seed)
        model = make_designed_model(hidden, tau, lam, s, seed, task_dim=0)
        ben = benettin_ftle_max(model, xdr)
        if ben < target:
            lo = lam
        else:
            hi = lam
    return lam, model, ben


def corr2(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--s-values", default="0.0,0.4,0.8,1.2,1.6,2.0")
    p.add_argument("--targets", default="-0.30,-0.05")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=16)
    p.add_argument("--bisect-iters", type=int, default=14)
    p.add_argument("--out", default="results/f24_memory_kappa_results.txt")
    args = p.parse_args()

    seeds = [int(v) for v in args.seeds.split(",")]
    s_vals = [float(v) for v in args.s_values.split(",")]
    targets = [float(v) for v in args.targets.split(",")]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, IS)
    lines = []
    lines.append("=" * 118)
    lines.append("F24 MEMORY-RESSOURCE: MC vs kappa bei FESTEM getriebenem Benettin (lam-Bisektion)")
    lines.append(f"targets={targets} s={s_vals} seeds={seeds} hidden={args.hidden_dim} max_lag={args.max_lag}")
    lines.append("Vorhersage (Ganguli/Orhan/Schuecker + F18c-Regime B): MC STEIGT mit kappa.")
    lines.append("REGISTRIERTE SEKUNDAER-VORHERSAGEN (vor dem Lauf, 2026-11-03):")
    lines.append("  S1: MC(kappa) hat ein INNERES Optimum (Exzess frisst im Extrembereich den SNR).")
    lines.append(f"  S2: MC-optimales kappa_T-Band ueberlappt F27-Adam-Endpunkte kappa_T in [{F27_ADAM_BAND[0]:.0f}, {F27_ADAM_BAND[1]:.0f}]"
                 f" (Run 34: {F27_ADAM_KT[0]:.1f}/{F27_ADAM_KT[1]:.1f}/{F27_ADAM_KT[2]:.1f}).")
    lines.append("  Original-Schwelle unangetastet: corr > +0.5 UND Monotonie >= 3/4 Seeds bei target=-0.30.")
    lines.append("=" * 118)

    rows = []
    for target in targets:
        lines.append("")
        lines.append(f"########## Ziel: getriebenes lambda = {target:+.2f} ##########")
        lines.append(f"{'s':>5s} {'seed':>4s} {'lam*':>6s} {'kappa':>6s} {'t0':>6s} {'logK':>5s} {'ben':>7s} {'MC':>6s} {'kT':>9s} {'kcond':>9s} {'dH':>6s} {'E':>7s}")
        for s_nn in s_vals:
            for seed in seeds:
                lam, m, ben = tune_lam(args.hidden_dim, 1.0, s_nn, seed, target,
                                       xdr, iters=args.bisect_iters)
                kappa = spectral_edge(m) / max(abs(float(eig_max(m))), 1e-9)
                t0, logk, _ = lin_features(m, xdr)
                E_exc = field_excess(m, xdr)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=args.max_lag,
                                        seq_len=64, input_scale=IS)
                kz = kappa_zoo(m.W.detach())
                lines.append(f"{s_nn:5.1f} {seed:4d} {lam:6.3f} {kappa:6.2f} {t0:+6.2f} "
                             f"{logk:5.2f} {ben:+7.3f} {mc:6.2f} "
                             f"{kz['kappa_T']:9.2f} {kz['kappa_cond']:9.2f} "
                             f"{kz['delta_H']:6.3f} "
                             f"{(f'{E_exc:7.2f}' if E_exc is not None else '      -')}")
                rows.append(dict(target=target, s=s_nn, seed=seed, lam=lam,
                                 kappa=kappa, t0=t0, logk=logk, ben=ben, mc=mc,
                                 kT=kz['kappa_T'], kcond=kz['kappa_cond'],
                                 dH=kz['delta_H'], E=E_exc))
        # Run-21/33-Lehre: Snapshot nach jedem Target-Block.
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- Aggregat: MC vs kappa bei fixem Target ---
    lines.append("")
    for target in targets:
        sub = [r for r in rows if r['target'] == target]
        lines.append(f"--- Ziel lambda={target:+.2f}: Mittelwerte pro s ---")
        lines.append(f"{'s':>5s} {'kappa':>6s} {'MC':>6s} {'kT':>9s} {'kcond':>9s} {'E':>7s}")
        mc_by_s, kap_by_s, kt_by_s, kc_by_s, e_by_s = {}, {}, {}, {}, {}
        for r in sub:
            mc_by_s.setdefault(r['s'], []).append(r['mc'])
            kap_by_s.setdefault(r['s'], []).append(r['kappa'])
            kt_by_s.setdefault(r['s'], []).append(r['kT'])
            kc_by_s.setdefault(r['s'], []).append(r['kcond'])
            if r['E'] is not None:
                e_by_s.setdefault(r['s'], []).append(r['E'])
        for s_nn in s_vals:
            if s_nn in mc_by_s:
                mm = sum(mc_by_s[s_nn]) / len(mc_by_s[s_nn])
                kk = sum(kap_by_s[s_nn]) / len(kap_by_s[s_nn])
                tt = sum(kt_by_s[s_nn]) / len(kt_by_s[s_nn])
                cc = sum(kc_by_s[s_nn]) / len(kc_by_s[s_nn])
                ee = (sum(e_by_s[s_nn]) / len(e_by_s[s_nn])) if s_nn in e_by_s else float('nan')
                lines.append(f"{s_nn:5.1f} {kk:6.2f} {mm:6.2f} {tt:9.2f} {cc:9.2f} {ee:7.2f}")
        # Pearson (Original-Instrument: spectral kappa) ueber alle Punkte
        kaps = [r['kappa'] for r in sub]
        mcs = [r['mc'] for r in sub]
        corr = corr2(kaps, mcs)
        if not math.isnan(corr):
            kaps_t = torch.tensor(kaps, dtype=torch.float64)
            mcs_t = torch.tensor(mcs, dtype=torch.float64)
            slope = ((kaps_t * mcs_t).mean() - kaps_t.mean() * mcs_t.mean())
            slope = slope / kaps_t.var().clamp_min(1e-9)
            lines.append(f"corr(kappa, MC) = {corr:+.3f} | OLS-Steigung MC/kappa = {slope:+.3f}")
        # Per-Seed Monotonie: MC am groessten vs kleinsten kappa
        up = 0
        for seed in seeds:
            ss = [r for r in sub if r['seed'] == seed]
            if len(ss) >= 2 and ss[-1]['mc'] > ss[0]['mc']:
                up += 1
        lines.append(f"Monotonie: {up}/{len(seeds)} Seeds mit MC(kappa_max) > MC(kappa_min)")
        # --- Zoo-Korrelationen: WELCHES kappa traegt die Memory-Ressource? ---
        lines.append("Zoo-Korrelationen (alle (log Mass, MC)-Punkte dieses Targets):")
        for key, label in [('kT', 'log kappa_T'), ('kcond', 'log kappa_cond'),
                           ('dH', 'log delta_H'), ('E', 'log E')]:
            pairs = [(r[key], r['mc']) for r in sub if r[key] is not None]
            if len(pairs) >= 4:
                c = corr2([math.log(max(v, 1e-9)) for v, _ in pairs],
                          [mm for _, mm in pairs])
                lines.append(f"  corr({label}, MC) = {c:+.3f}  (n={len(pairs)})")
        # --- S1: inneres Optimum? Per-Seed Argmax ueber s ---
        argmax_s = {}
        for seed in seeds:
            ss = [r for r in sub if r['seed'] == seed]
            if ss:
                best = max(ss, key=lambda r: r['mc'])
                argmax_s[seed] = best['s']
        interior = [1 for v in argmax_s.values()
                    if v not in (s_vals[0], s_vals[-1])]
        lines.append(f"S1 Argmax-s pro Seed: {[f'seed {k}: {v:.1f}' for k, v in sorted(argmax_s.items())]}")
        lines.append(f"S1 inneres Optimum (Argmax nicht am Rand) bei {sum(interior)}/{len(argmax_s)} Seeds"
                     f" — Rand = s in {{{s_vals[0]:.1f}, {s_vals[-1]:.1f}}}")
        # --- S2: MC-optimales kappa_T-Band vs F27-Adam-Endpunkte ---
        if argmax_s:
            kt_at_best = []
            for seed, s_best in argmax_s.items():
                cand = [r['kT'] for r in sub if r['seed'] == seed and r['s'] == s_best]
                if cand:
                    kt_at_best.append(cand[0])
            if kt_at_best:
                kt_min = min(kt_at_best)
                kt_max = max(kt_at_best)
                lo, hi = F27_ADAM_BAND
                overlap = kt_min <= hi and kt_max >= lo
                lines.append(f"S2 kappa_T am MC-Optimum pro Seed: "
                             f"{[f'{v:.1f}' for v in kt_at_best]}")
                lines.append(f"S2 Band-Ueberlappung mit F27-Adam [{lo:.0f}, {hi:.0f}]: "
                             f"{'JA' if overlap else 'NEIN'} "
                             f"(kappa_T-Band hier: [{kt_min:.1f}, {kt_max:.1f}])")

    lines.append("")
    lines.append("LESEAT: corr > +0.5 UND Monotonie >= 3/4 Seeds bei target=-0.30 =>")
    lines.append("Nicht-Normalitaet ist freie Memory-Ressource unter fest getriebener")
    lines.append("Stabilitaet (Regime-B/Orhan/Schuecker-Kollision BESTAETIGT).")
    lines.append("corr ~ 0 => F12-Kovariations-Kritik trifft: Ressource war lam, nicht kappa.")
    lines.append("S1 trifft => MC(kappa) hat inneres Optimum: extremes kappa ist Exzess,")
    lines.append("  kein SNR-Speicher — verfeinert Orhan (kappa=1 toetet) zu einem Fenster.")
    lines.append("S2 trifft => Adams Erosion fuhr (widerwillig) ins MC-optimale Band:")
    lines.append("  Antagonist und Ressourcen-Optimum haenden zusammen.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
