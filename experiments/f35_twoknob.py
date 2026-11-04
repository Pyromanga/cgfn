'''
F35 — ZWEI-KNOPF-TEST: DELTA_H-BETRIEBSZUSTAND ODER GARTEN DER VERZWEIGTEN PFADE
(registriert 2026-11-04, VOR dem Lauf, keine Nachverhandlung).

Vorgeschichte:
  F34 Phase 1: delta_H mean corr +0.790, 10/10 Zellen positiv.
  F34 Phase 2 (f34_confirmation): TEIL-PASS — 4/10 Zellen (>= 8 noetig).
  Forensik des Teil-Pass: KEIN Transfer-Versagen, sondern Range-Restriktion.
  delta_H = s * lam* * ||upper||_F (Schur-Konstruktion), und die Bisektion
  auf ben=-0.30 handelt lam* gegen s so, dass das PRODUKT s*lam* bei H=128
  nahezu konstant bleibt (dH ~ 15-17 ueber alle s). Der s-Knopf ist bei
  H=128 unter Stabilitaets-Matching TOT — keine Praediktor-Varianz, keine
  Korrelation per Konstruktion. Aufgabenseite dasselbe Muster umgekehrt:
  trend/prod an der R2-Decke zeigen keine Korrelation.

F35-FRAGE (neu, sauber): Wenn die delta_H-Varianz bei festem ben-Ziel
  WIEDERHERGESTELLT wird — praediziert delta_H dann Aufgabenleistung
  auch bei H=128? Ein Ein-Knopf-Design kann diese Zwei-Knopf-Frage nicht
  entscheiden.

ZWEITER KNOPF: Bandbreite b des streng-obren Rauschens.
  b klein (b=1)  => Jordan-artige Ketten: maximales transientes
                   Wachstum pro Frobenius-Einheit Nicht-Normalitaet.
  b gross (b=8) => diffuse Nicht-Normalitaet, gleiche Henrici-Masse
                   brauchen mehr lam*-Abzug fuer gleiches ben.
  Die Band-Struktur aendert die ben<->delta_H-Handelskurve, nicht die
  Eigenwerte (Dreiecksform bleibt) — dH-Varianz bei festem ben-Ziel.

REGISTRIERUNG (VOR DEM LAUF GENAGELT):
  Gitter: H in {32,128} x b in {1,2,4,8} x s in {0.6,1.2} x seeds
          {10,11,12,13} (frisch) = 32 Modelle je H, 64 gesamt.
          Bisektion lam* auf ben=-0.30 wie gehabt (lo=0.02).
  Zellen: 2 H x 5 Aufgaben = 10, n=32 je Zelle.
  PRIMAER:   corr(delta_H, TASK_R2) >= +0.4 in >= 6/10 Zellen
             -> PASS: delta_H validiert (mit restaurierter Varianz);
                Homeostat-Ziel F32 offen.
             < 6/10 und Varianz-Gate erfuellt
             -> KILL: Betriebszustands-Programm endgueltig tot,
                Rueckstufung auf F27-Phaenomenologie. Kein dritter
                Versuch, kein neues Gitter, keine Nachverhandlung.
  SEKUNDAER: corr(MC) >= +0.4 in >= 6/10 Zellen.
  VARIANZ-GATE (Guetigkeits-Bedingung, KEINE Schwelle zum Verhandeln):
             delta_H-Range (max-min) je H >= 2.0. Verletzung bei
             H=128 -> UNENTSCHEIDEN (Knopf erneut tot: Design, nicht
             Schwelle). Report der Range ungeachtet des Verdikts.
  MANIPULATIONS-CHECK: |corr(ben1, TASK_R2)| < 0.4 in ALLEN Zellen,
             sonst UNENTSCHEIDEN (Matching durchbrochen).
  Instrument: observables/tasks/corr2 IDENTISCH via Import aus
             f34_exploration — kein Messpfad-Drift.
'''
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import (observables, corr2, task_sinus, task_narma10,
                             task_white_delay8, task_prod, task_trend,
                             nlm, BEN_TARGET, IS)

from cgfn.benettin import benettin_spectrum
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity
from cgfn.model import ContinuousRNN


def recurrent_matrix_band(H, lam_max, s_nn, seed, band):
    '''Schur-Konstruktion wie cgfn.design, aber obres Rauschen nur auf
    den ersten "band" Nebendiagonalen (band=1: Jordan-artige Kette).
    Eigenwerte unveraendert (Dreiecksform), delta_H-Struktur veraendert.'''
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(H, H, generator=gen)
    Q, _ = torch.linalg.qr(A)
    others = lam_max * 0.9 * (2.0 * torch.rand(H - 1, generator=gen) - 1.0)
    eigs = torch.cat([torch.tensor([lam_max], dtype=torch.float32), others])
    T = torch.diag(eigs)
    if s_nn > 0 and band > 0:
        upper = torch.triu(torch.randn(H, H, generator=gen), diagonal=1)
        keep = (torch.triu(torch.ones(H, H), diagonal=1) -
                torch.triu(torch.ones(H, H), diagonal=band + 1))
        T = T + s_nn * lam_max * upper * keep
    return Q @ T @ Q.T


def make_band_model(H, s_nn, lam, seed, band):
    model = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
    W = recurrent_matrix_band(H, lam, s_nn, seed, band)
    with torch.no_grad():
        model.W.copy_(W)
    model.W.requires_grad_(False)
    return model


def tune_lam_band(H, s_nn, seed, band, xdr, lo=0.02, hi=2.5, iters=10):
    lam = 0.5 * (lo + hi)
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        m = make_band_model(H, s_nn, lam, seed, band)
        spec, _ = benettin_spectrum(m, xdr)
        ben = float(spec[0])
        if ben < BEN_TARGET:
            lo = lam
        else:
            hi = lam
    m = make_band_model(H, s_nn, lam, seed, band)
    return lam, m


def main():
    p = argparse.ArgumentParser(description='F35 two-knob delta_H test')
    p.add_argument('--seeds', default='10,11,12,13')
    p.add_argument('--bands', default='1,2,4,8')
    p.add_argument('--s-values', default='0.6,1.2')
    p.add_argument('--out', default='results/f35_twoknob_results.txt')
    p.add_argument('--csv', default='results/f35_twoknob_table.csv')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    bands = [int(v) for v in args.bands.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    tasks = {'sinus': task_sinus, 'narma': task_narma10,
             'wdelay8': task_white_delay8, 'prod': task_prod,
             'trend': task_trend}

    xdr = drive_input(4, 48, IS)
    L = []
    L.append('=' * 118)
    L.append('F35 — ZWEI-KNOPF-TEST (registriert: delta_H >= +0.4 in >= 6/10 Zellen = PASS,')
    L.append('sonst KILL; Varianz-Gate dH-Range >= 2.0 je H; Manipulations-Check ben1 < 0.4)')
    L.append(f'Gitter: H in {{32,128}} x b={bands} x s={s_vals} x seeds={seeds}, Bisektion auf ben={BEN_TARGET}')
    L.append('=' * 118)

    rows = []
    for H in (32, 128):
        L.append('')
        L.append(f'########## H={H} ##########')
        for band in bands:
            for s_nn in s_vals:
                for seed in seeds:
                    lam, m = tune_lam_band(H, s_nn, seed, band, xdr)
                    obs = observables(m, xdr)
                    torch.manual_seed(seed + 9000)
                    mc, _ = memory_capacity(m, max_lag=16, seq_len=64,
                                             input_scale=IS)
                    obs['MC'] = mc
                    obs['NLM'] = nlm(m, seed)
                    obs['lam'] = lam
                    t2s = {tname: tfn(m, seed) for tname, tfn in tasks.items()}
                    rows.append(dict(H=H, band=band, s=s_nn, seed=seed,
                                     obs=obs, t2=t2s))
                    L.append(f"H={H} b={band} s={s_nn:.1f} seed={seed} "
                             f"lam*={lam:.3f} ben1={obs['ben1']:+.3f} "
                             f"dH={obs['delta_H']:.3f} MC={mc:.2f} "
                             f"sinus={t2s['sinus']:+.3f} narma={t2s['narma']:+.3f} "
                             f"wdelay8={t2s['wdelay8']:+.3f} prod={t2s['prod']:+.3f} "
                             f"trend={t2s['trend']:+.3f}")
        Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    # --- Breite CSV ---
    obs_keys = sorted(rows[0]['obs'].keys())
    with open(args.csv, 'w', encoding='utf-8') as fh:
        fh.write('H,band,s,seed,' + ','.join(obs_keys) + ',' +
                 ','.join('t2_' + t for t in tasks) + '\n')
        for r in rows:
            ovals = [f"{r['obs'][k]:.6g}" if r['obs'][k] == r['obs'][k] else ''
                     for k in obs_keys]
            vals = [str(r['H']), str(r['band']), str(r['s']), str(r['seed'])] + \
                   ovals + [f"{r['t2'][t]:.6g}" for t in tasks]
            fh.write(','.join(vals) + '\n')

    # --- Registrierte Auswertung ---
    L.append('')
    L.append('--- F35-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    dh_ranges = {}
    for H in (32, 128):
        sub = [r['obs']['delta_H'] for r in rows if r['H'] == H]
        dh_ranges[H] = max(sub) - min(sub)
        L.append(f'Varianz-Gate H={H}: delta_H-Range = {dh_ranges[H]:.3f} '
                 f'(>= 2.0 gefordert)')
    variance_ok = all(v >= 2.0 for v in dh_ranges.values())

    hits = {'delta_H': 0, 'MC': 0}
    ben_violations = []
    per_cell = []
    for H in (32, 128):
        for tname in tasks:
            sub = [r for r in rows if r['H'] == H]
            tvals = [r['t2'][tname] for r in sub]
            cell = dict(tag=f'H={H} {tname}')
            for key in ('delta_H', 'MC', 'ben1'):
                c = corr2([r['obs'][key] for r in sub], tvals)
                cell[key] = c
                if c == c:
                    if key in hits and c >= 0.4:
                        hits[key] += 1
                    if key == 'ben1' and abs(c) >= 0.4:
                        ben_violations.append(f'H={H} {tname}: {c:+.3f}')
            per_cell.append(cell)
            L.append(f"Zelle H={H} {tname}: corr(delta_H)={cell['delta_H']:+.3f} "
                     f"corr(MC)={cell['MC']:+.3f} corr(ben1)={cell['ben1']:+.3f}")

    n_cells = len(per_cell)
    L.append('')
    L.append(f'PRIMAER delta_H: {hits["delta_H"]}/{n_cells} Zellen >= +0.4   (>= 6 noetig)')
    L.append(f'SEKUNDAER MC:   {hits["MC"]}/{n_cells} Zellen >= +0.4   (>= 6 noetig)')
    if ben_violations:
        L.append('MANIPULATIONS-CHECK VERLETZT: ' + '; '.join(ben_violations))
    else:
        L.append('MANIPULATIONS-CHECK: ben1 ueberall < 0.4 — Matching haelt.')

    if ben_violations:
        verdict = ('UNENTSCHEIDEN — Stabilitaets-Matching durchbrochen '
                   '(Design-Problem, Schwelle unangetastet).')
    elif not variance_ok:
        verdict = ('UNENTSCHEIDEN — Varianz-Gate verletzt: auch der Band-Knopf '
                   'stellt bei H=128 keine delta_H-Varianz her (Range < 2.0). '
                   'Design-Frage erneut offen, KEINE neue Schwelle.')
    elif hits['delta_H'] >= 6:
        verdict = ('PASS — delta_H praediziert Aufgabenleistung auch bei '
                   'restaurierter Varianz und beiden H: Betriebszustand '
                   'validiert. Homeostat-Ziel (F32) offen: Regler auf delta_H.')
    else:
        verdict = ('KILL — delta_H praediziert NICHT, sobald die Varianz '
                   'da ist und die Decke weg ist. Betriebszustands-Programm '
                   'endgueltig tot: Rueckstufung auf F27-Phaenomenologie, '
                   'kein dritter Versuch.')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} und {args.csv} geschrieben.')


if __name__ == '__main__':
    main()
