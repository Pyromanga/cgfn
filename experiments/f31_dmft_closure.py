'''
F31 — RESSOURCEN-KERN: DMFT-Selbstschluss, Sigma-Buchhaltung,
Kappa-Zoo, Optimierer-Kokkyklus  (Tiefenanalyse XI — die Ratte im Schwanz).

Vier Zirkularitaeten, vier Schnitte, vorab festgenagelte Kriterien:

  P1 (Schluss-d): d_dmft vs d_emp (Rollout). Im linearen Regime fallen
     sie zusammen; die relative Abweichung ist das Nichtlinearitaets-
     Meter. Erwartung: d_gap wachst mit s (kappa) und Antrieb.
  P2 (Schluss-lambda): |lam_dmft - lam_benettin| vs
     |lam_open - lam_benettin| im Mittel uebers Gitter. VORAB: DMFT
     gewinnt. Kill bei Gegenteil = DMFT-Geschichte falsch herum lesen
     (d_emp enthaelt jenseits-MF-Information).
  P3 (H2- und MC-Konsistenz): corr(MC_opt_linear, MC_gemessen) >= 0.8
     und Tr(Sigma)-Schluss (h2_emp vs h2_dmft). HINWEIS: Das ist ein
     Konsistenz-Test des LINEAREN Regimes — keine Evidenz fuer die
     Ressourcen-These (die lebt und stirbt mit F30, Transfer).
  P4 (Exzess): E = Tr(Sigma)/Tr(Sigma_norm) wachst mit s; log E
     korreliert mit log g und dem besten kappa-Mass (Erwartung:
     kappa_T / Kreiss vor kappa_cond).
  P5 (Optimierer-Kokkyklus): g_opt(beta) des Heavy-Ball-Kompagnons
     monoton in beta (EoS-'Sharpness' ist Update-Operator-Fixpunkt,
     kein Landschafts-Eigentum).

Design: designed models (Schur, s als kappa-Regler), lam-Sweep ueber
die getriebene Kante, kein Training, freeze_w. Alles CPU-torch,
deterministische Seeds.
'''
import argparse
import math
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max
from cgfn.design import make_designed_model
from cgfn.field import drive_input, predict_driven_ftle_field
from cgfn.memory_capacity import memory_capacity
from cgfn.resource import (dmft_fixed_point, empirical_sigma,
                           kappa_zoo, mc_linear_prediction,
                           normal_excess, transient_growth,
                           update_cocycle_growth)


def fmt(v, w=7, p=3):
    if v is None or v != v:
        return ' ' * (w - 1) + '-'
    return f'{v:+{w}.{p}f}'


def corr2(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    p = argparse.ArgumentParser(description='F31 DMFT closure battery')
    p.add_argument('--seeds', default='0,1')
    p.add_argument('--lam-values', default='0.6,1.0,1.4,1.8')
    p.add_argument('--s-values', default='0.0,0.8,1.6')
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--T', type=int, default=2500)
    p.add_argument('--max-lag', type=int, default=16)
    p.add_argument('--input-scale', type=float, default=0.5)
    p.add_argument('--out', default='results/f31_dmft_results.txt')
    args = p.parse_args()

    torch.set_num_threads(2)
    seeds = [int(v) for v in args.seeds.split(',')]
    lams = [float(v) for v in args.lam_values.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    L = []
    L.append('=' * 128)
    L.append('F31 RESSOURCEN-KERN: DMFT-Selbstschluss (d<->Sigma), H2/MC-Theorie, Kappa-Zoo, Optimierer-Kokkyklus')
    L.append(f'lam={lams} s={s_vals} seeds={seeds} H={args.hidden} T={args.T} max_lag={args.max_lag} input_scale={args.input_scale}')
    L.append('P1 d-Schluss | P2 lambda-Schluss (DMFT vs offen, Benettin als Wahrheit) | P3 MC/H2-Konsistenz |')
    L.append('P4 Exzess E=TrS/TrS_norm vs kappa-Zoo | P5 g_opt(beta) Heavy-Ball-Kompagnon')
    L.append('=' * 128)

    rows = []
    for s in s_vals:
        L.append('')
        L.append(f'########## s (kappa-Regler) = {s} ##########')
        L.append(f"{'lam':>5s} {'seed':>4s} {'ben':>7s} {'open':>7s} {'dmft':>7s} {'dgap':>7s} {'h2emp':>7s} {'h2pre':>7s} {'MCmes':>6s} {'MCpre':>6s} {'E':>6s} {'g':>6s} {'t*':>5s} {'kcon':>6s} {'kT':>6s} {'dH':>6s} {'kK':>6s} {'it':>4s} {'ok':>3s}")
        for lam in lams:
            for seed in seeds:
                torch.manual_seed(seed)
                model = make_designed_model(args.hidden, 1.0, lam, s, seed, task_dim=0)
                Delta = float(model.substeps) * float(model.dt)

                xdr = drive_input(4, 48, args.input_scale)
                lam_ben = benettin_ftle_max(model, xdr)
                x_probe = drive_input(1, 200, args.input_scale)
                lam_open = predict_driven_ftle_field(model, x_probe)

                Sig_emp, d_emp = empirical_sigma(model, args.T, args.input_scale, 200, seed + 1234)
                sig2 = args.input_scale ** 2
                h2_emp = Sig_emp.diagonal().sum().item() / sig2

                dmft = dmft_fixed_point(model, args.input_scale)
                r = dict(lam=lam, s=s, seed=seed, lam_ben=lam_ben, lam_open=lam_open,
                         dmft_ok=dmft is not None)
                if dmft is None:
                    rows.append(r)
                    L.append(f'{lam:5.2f} {seed:4d} {fmt(lam_ben)} {fmt(lam_open)} ' + ' ' * 7 + '-' +
                             '   -    -    -    -    -    -    -     -    -    -    -    -    -    -  0   NO')
                    continue

                lam_dmft = dmft['lam_field']
                d_gap = ((dmft['d'].double() - d_emp.double()).abs()
                         / d_emp.double().clamp_min(1e-6)).max().item()
                B = model.U.weight.detach()
                pred = mc_linear_prediction(dmft['J'], B, args.input_scale, Delta, args.max_lag)
                if pred is None:
                    rows.append(r)
                    continue
                mc_meas, r2_lag = memory_capacity(model, max_lag=args.max_lag,
                                                  seq_len=64, input_scale=args.input_scale)
                kz = kappa_zoo(dmft['J'])
                g, t_star = transient_growth(dmft['J'])
                E_exc, _S, _Sn = normal_excess(dmft['J'], dmft['Q'])

                r.update(lam_dmft=lam_dmft, d_gap=d_gap,
                         h2_emp=h2_emp, h2_pred=pred['h2'],
                         mc_meas=mc_meas, mc_pred=pred['mc_opt'],
                         per_lag=pred['mc_per_lag'], r2_lag=r2_lag,
                         E=E_exc, g=g, t_star=t_star,
                         kcond=kz['kappa_cond'], kT=kz['kappa_T'],
                         dH=kz['delta_H'], klb=kz['kreiss_lb'],
                         iters=dmft['iters'], conv=dmft['converged'])
                rows.append(r)
                L.append(f'{lam:5.2f} {seed:4d} {fmt(lam_ben)} {fmt(lam_open)} {fmt(lam_dmft)} '
                         f'{fmt(d_gap)} {fmt(h2_emp)} {fmt(r["h2_pred"])} {mc_meas:6.2f} '
                         f'{r["mc_pred"]:6.2f} {E_exc:6.2f} {g:6.2f} {t_star:5.1f} '
                         f'{r["kcond"]:6.2f} {r["kT"]:6.2f} {r["dH"]:6.2f} {r["klb"]:6.2f} '
                         f'{r["iters"]:4d} {"YES" if r["conv"] else "NO":>3s}')

    ok = [r for r in rows if r.get('dmft_ok') and r.get('mc_pred') is not None]
    n_div = sum(1 for r in rows if not r['dmft_ok'])

    L.append('')
    L.append('=' * 128)
    L.append(f'AUSWERTUNG ({len(ok)} DMFT-ok Konfigurationen, {n_div} ohne stabilen Fixpunkt)')
    if ok:
        dg = [r['d_gap'] for r in ok]
        L.append(f'P1 Schluss-d: mean d_gap = {sum(dg)/len(dg):.4f}, max = {max(dg):.4f}')
        for s in s_vals:
            sub = [r['d_gap'] for r in ok if r['s'] == s]
            if sub:
                L.append(f'  s={s}: mean d_gap = {sum(sub)/len(sub):.4f}')
        e_dm = [abs(r['lam_dmft'] - r['lam_ben']) for r in ok]
        e_op = [abs(r['lam_open'] - r['lam_ben']) for r in ok]
        wins = sum(1 for a, b in zip(e_dm, e_op) if a < b)
        L.append(f'P2 Schluss-lambda: |dmft-ben| mean {sum(e_dm)/len(e_dm):+.4f} vs '
                 f'|open-ben| mean {sum(e_op)/len(e_op):+.4f} | DMFT besser in {wins}/{len(ok)}')
        L.append('   (Kill bei Gegenteil: d_emp enthaelt jenseits-MF-Information — DMFT falsch herum lesen)')
        c_mc = corr2([r['mc_pred'] for r in ok], [r['mc_meas'] for r in ok])
        ratio = sum(r['mc_meas'] / max(r['mc_pred'], 1e-9) for r in ok) / len(ok)
        c_h2 = corr2([r['h2_pred'] for r in ok], [r['h2_emp'] for r in ok])
        rh2 = sum(r['h2_emp'] / max(r['h2_pred'], 1e-9) for r in ok) / len(ok)
        L.append(f'P3 MC-Konsistenz: corr(MC_opt_linear, MC_gemessen) = {c_mc:+.3f} (>= 0.8 noetig), '
                 f'mean ratio = {ratio:.3f}')
        L.append(f'   Tr(Sigma)-Schluss: corr(h2_emp, h2_dmft) = {c_h2:+.3f}, mean ratio = {rh2:.3f}')
        L.append('   (Hinweis: Konsistenz-Test des linearen Regimes — KEINE Evidenz fuer die These; die lebt mit F30)')
        for s in s_vals:
            sub = [r['E'] for r in ok if r['s'] == s]
            if sub:
                L.append(f'P4 Exzess s={s}: mean E = {sum(sub)/len(sub):.3f}')
        lg = [math.log(max(r['E'], 1e-9)) for r in ok]
        L.append(f'   corr(log E, MC)    = {corr2(lg, [r["mc_meas"] for r in ok]):+.3f}')
        L.append(f'   corr(log E, log g) = {corr2(lg, [math.log(max(r["g"], 1e-9)) for r in ok]):+.3f}')
        for key, label in [('kcond', 'kappa_cond'), ('kT', 'kappa_T'),
                           ('dH', 'delta_H'), ('klb', 'kreiss_lb')]:
            lk = [math.log(max(r[key], 1e-9)) for r in ok]
            L.append(f'   corr(log {label}, log E) = {corr2(lk, lg):+.3f}')

        anchor = None
        best_d = 1e9
        for r in ok:
            d = abs(r['lam'] - 1.0) + abs(r['s'] - 0.8)
            if r['seed'] == seeds[0] and d < best_d:
                anchor, best_d = r, d
        if anchor is not None:
            L.append('')
            L.append(f'--- Lag-Profil Anker (lam={anchor["lam"]}, s={anchor["s"]}, seed={anchor["seed"]}) ---')
            L.append(f"{'lag':>4s} {'r2_gemes':>9s} {'theorie':>9s}")
            for k in range(args.max_lag):
                meas = anchor['r2_lag'][k + 1] if k + 1 < len(anchor['r2_lag']) else float('nan')
                pred = anchor['per_lag'][k]
                L.append(f'{k+1:4d} {meas:9.4f} {pred:9.4f}')
    else:
        L.append('[w] keine DMFT-ok Konfigurationen — Gitter liegt supra-linear? lam-Grid senken.')

    L.append('')
    L.append('--- P5: Optimierer-Kokkyklus (Heavy-Ball-Kompagnon, quadratisches Demo) ---')
    h = torch.logspace(0, 2, 16)
    L.append(f'Hessian-Spread: {h[0].item():.1f}..{h[-1].item():.1f} (log-spaced, n=16), eta=0.019')
    g_prev = None
    mono = True
    for beta in [0.0, 0.5, 0.9, 0.99]:
        g_opt = update_cocycle_growth(h, eta=0.019, beta=beta, t_max=600, n_probe=4)
        if g_prev is not None and g_opt < g_prev - 1e-9:
            mono = False
        L.append(f'beta={beta:4.2f}: g_opt = {g_opt:.3f}')
        g_prev = g_opt
    L.append(f'P5 monoton in beta: {"JA" if mono else "NEIN"} '
             '(EoS-Sharpness ist Fixpunkt des Update-Operators, kein Landschafts-Eigentum)')

    L.append('')
    L.append('LESEAT:')
    L.append('P1 ok => DMFT-Schluss tragfaehig, d_gap ist das Nichtlinearitaets-Meter.')
    L.append('P2 ok => Homeostat darf lambda_dmft regeln (geschlossenes Modell);')
    L.append('     P2 faellt => offenes Feld NICHT durch DMFT ersetzen — Fluktuationen tragen Information.')
    L.append('P3 ok => lineare Theorie beschreibt das Reservoir; E/h2 sind kontrollierte Koordinaten.')
    L.append('P4 ok => Exzess E ist die messbare kappa-Achse (SSM: E=1 strukturell).')
    L.append('P5 ok => Optimierer-Antagonist ist der Kompagnon-Kokkyklus (g_opt), nicht das Hessian.')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
