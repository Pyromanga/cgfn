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
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max
from cgfn.cavity import eig_max, spectral_edge
from cgfn.design import make_designed_model
from cgfn.field import (drive_input, sech2_field_iteration,
                        kreiss_constant, transient_bound_pseudo)
from cgfn.memory_capacity import memory_capacity


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
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    xdr = drive_input(4, 48, 0.5)
    lines = []
    lines.append("=" * 118)
    lines.append("F24 MEMORY-RESSOURCE: MC vs kappa bei FESTEM getriebenem Benettin (lam-Bisektion)")
    lines.append(f"targets={targets} s={s_vals} seeds={seeds} hidden={args.hidden_dim} max_lag={args.max_lag}")
    lines.append("Vorhersage (Ganguli/Orhan/Schuecker + F18c-Regime B): MC STEIGT mit kappa.")
    lines.append("=" * 118)

    rows = []  # (target, s, seed, lam, kappa, t0, logK, ben, mc)
    for target in targets:
        lines.append("")
        lines.append(f"########## Ziel: getriebenes lambda = {target:+.2f} ##########")
        lines.append(f"{'s':>5s} {'seed':>4s} {'lam*':>6s} {'kappa':>6s} {'t0':>6s} {'logK':>5s} {'ben':>7s} {'MC':>6s}")
        for s_nn in s_vals:
            for seed in seeds:
                lam, m, ben = tune_lam(args.hidden_dim, 1.0, s_nn, seed, target,
                                       xdr, iters=args.bisect_iters)
                kappa = spectral_edge(m) / max(abs(float(eig_max(m))), 1e-9)
                t0, logk, _ = lin_features(m, xdr)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=args.max_lag,
                                        seq_len=64, input_scale=0.5)
                lines.append(f"{s_nn:5.1f} {seed:4d} {lam:6.3f} {kappa:6.2f} {t0:+6.2f} {logk:5.2f} {ben:+7.3f} {mc:6.2f}")
                rows.append((target, s_nn, seed, lam, kappa, t0, logk, ben, mc))

    # --- Aggregat: MC vs kappa bei fixem Target ---
    lines.append("")
    for target in targets:
        sub = [r for r in rows if r[0] == target]
        lines.append(f"--- Ziel lambda={target:+.2f}: Mittelwerte pro s ---")
        mc_by_s, kap_by_s = {}, {}
        for r in sub:
            mc_by_s.setdefault(r[1], []).append(r[8])
            kap_by_s.setdefault(r[1], []).append(r[4])
        smean, kmean = [], []
        for s_nn in s_vals:
            if s_nn in mc_by_s:
                mm = sum(mc_by_s[s_nn]) / len(mc_by_s[s_nn])
                kk = sum(kap_by_s[s_nn]) / len(kap_by_s[s_nn])
                smean.append(mm); kmean.append(kk)
                lines.append(f"s={s_nn:3.1f}: kappa {kk:5.2f} | MC {mm:6.2f}")
        # Pearson ueber alle (kappa, MC)-Punkte
        kaps = torch.tensor([r[4] for r in sub])
        mcs = torch.tensor([r[8] for r in sub])
        if kaps.std() > 1e-6 and mcs.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([kaps, mcs]))[0, 1].item()
            slope = (kaps * mcs).mean() - kaps.mean() * mcs.mean()
            slope = slope / kaps.var().clamp_min(1e-9)
            lines.append(f"corr(kappa, MC) = {corr:+.3f} | OLS-Steigung MC/kappa = {slope:+.3f}")
        # Per-Seed Monotonie: MC am groessten vs kleinsten kappa
        up = 0
        for seed in seeds:
            ss = [r for r in sub if r[2] == seed]
            if len(ss) >= 2 and ss[-1][8] > ss[0][8]:
                up += 1
        lines.append(f"Monotonie: {up}/{len(seeds)} Seeds mit MC(kappa_max) > MC(kappa_min)")

    lines.append("")
    lines.append("LESEAT: corr > +0.5 UND Monotonie >= 3/4 Seeds bei target=-0.30 =>")
    lines.append("Nicht-Normalitaet ist freie Memory-Ressource unter fest getriebener")
    lines.append("Stabilitaet (Regime-B/Orhan/Schuecker-Kollision BESTAETIGT).")
    lines.append("corr ~ 0 => F12-Kovariations-Kritik trifft: Ressource war lam, nicht kappa.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
