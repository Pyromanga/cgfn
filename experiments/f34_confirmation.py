'''
F34 — ZWEIPHASEN-SUCHE, PHASE 2: BESTAETIGUNG (registriert 2026-11-04).

Phase 1 (Run: f34_exploration, 2026-11-04, 40 Modelle, 10 Zellen)
fand deskriptiv:
  delta_H  mean +0.790 (in ALLEN 10 Zellen positiv, min +0.633)
  lam      -0.752 | t_star -0.742 | NLM +0.724 | MC +0.679 (10/10
  positiv, min +0.500) | ben1 -0.013 (Matching haelt) | E tot.

REGISTRIERUNG PHASE 2 (VOR dem Lauf, keine Nachverhandlung):
  PRIMAER:   delta_H — corr(delta_H, TASK_R2) >= +0.4 in
             >= 8 von 10 Zellen (2 H x 5 Aufgaben).
  SEKUNDAER: MC — >= +0.4 in >= 6/10 Zellen.
             NLM — >= +0.4 in >= 6/10 Zellen.
  MANIPULATIONS-CHECK: |corr(ben1, TASK_R2)| < 0.4 in ALLEN Zellen
             (Stabilitaets-Matching darf in den frischen Seeds nicht
             durchbrochen sein).
  VERDICT:
    >= 8/10 delta-H-Treffer  -> PASS: delta_H ist die validierte
        Ressourcenachse; Homeostat-Ziel (F32) wieder offen.
    4-7/10                   -> TEIL-PASS: schwach generalisiert,
        Fundament zu duenn fuer Regler-Bau.
    < 4/10                   -> KILL: Phase-1-Signal war Garten der
        verzweigten Pfade; Ressourcen-Wette endgueltig tot,
        Projekt-Rueckstufung auf F27-Phaenomenologie.
    Manipulations-Check verletzt -> UNENTSCHEIDEN (Design, nicht
        Schwelle).

DESIGN (identisches Instrument wie Phase 1 — Import aus
f34_exploration, kein Code-Drift):
  - FRISCHE Seeds {4,5,6,7,8,9} (Phase 1: 0-3), gleiche s-Werte,
    gleiche Bisektion auf ben=-0.30, gleiche 5 Aufgaben.
  - Zellen: H in {32,128} x 5 Aufgaben, n=30 je Zelle.
  - Alle Observablen geloggt (CSV), Auswertung NUR gegen die
    registrierten Groessen; der Rest ist deskriptiv.
'''
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import (observables, tune_lam, corr2,
                             task_sinus, task_narma10, task_white_delay8,
                             task_prod, task_trend, nlm,
                             BEN_TARGET, IS)

from cgfn.benettin import benettin_spectrum
from cgfn.design import make_designed_model
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity


def main():
    p = argparse.ArgumentParser(description='F34 phase 2 confirmation')
    p.add_argument('--seeds', default='4,5,6,7,8,9')
    p.add_argument('--s-values', default='0.0,0.4,0.8,1.2,1.6')
    p.add_argument('--out', default='results/f34_confirmation_results.txt')
    p.add_argument('--csv', default='results/f34_confirmation_table.csv')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    tasks = {'sinus': task_sinus, 'narma': task_narma10,
             'wdelay8': task_white_delay8, 'prod': task_prod,
             'trend': task_trend}

    xdr = drive_input(4, 48, IS)
    L = []
    L.append('=' * 118)
    L.append('F34 PHASE 2 — BESTAETIGUNG (registriert: delta_H PRIMAER >= +0.4 in >= 8/10 Zellen;')
    L.append('SEKUNDAER MC/NLM >= 6/10; Manipulations-Check ben1 < 0.4 ueberall; frische Seeds)')
    L.append(f'Gitter: H in {{32,128}} x s={s_vals} x seeds={seeds} (frisch), Bisektion auf ben={BEN_TARGET}')
    L.append('=' * 118)

    rows = []
    for H in (32, 128):
        L.append('')
        L.append(f'########## H={H} ##########')
        for s_nn in s_vals:
            for seed in seeds:
                lam, m = tune_lam(H, s_nn, seed, xdr)
                obs = observables(m, xdr)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=IS)
                obs['MC'] = mc
                obs['NLM'] = nlm(m, seed)
                obs['lam'] = lam
                t2s = {tname: tfn(m, seed) for tname, tfn in tasks.items()}
                rows.append(dict(H=H, s=s_nn, seed=seed, obs=obs, t2=t2s))
                L.append(f"H={H} s={s_nn:.1f} seed={seed} lam*={lam:.3f} "
                         f"ben1={obs['ben1']:+.3f} dH={obs['delta_H']:.3f} "
                         f"MC={mc:.2f} NLM={obs['NLM']:+.3f} "
                         f"sinus={t2s['sinus']:+.3f} narma={t2s['narma']:+.3f} "
                         f"wdelay8={t2s['wdelay8']:+.3f} prod={t2s['prod']:+.3f} "
                         f"trend={t2s['trend']:+.3f}")
        Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    # --- Breite CSV (alle Observablen, fuer spaetere Analysen) ---
    obs_keys = sorted(rows[0]['obs'].keys())
    with open(args.csv, 'w', encoding='utf-8') as fh:
        fh.write('H,s,seed,' + ','.join(obs_keys) + ',' +
                 ','.join('t2_' + t for t in tasks) + '\n')
        for r in rows:
            ovals = [f"{r['obs'][k]:.6g}" if r['obs'][k] == r['obs'][k] else ''
                     for k in obs_keys]
            vals = [str(r['H']), str(r['s']), str(r['seed'])] + ovals + \
                   [f"{r['t2'][t]:.6g}" for t in tasks]
            fh.write(','.join(vals) + '\n')

    # --- Registrierte Auswertung ---
    L.append('')
    L.append('--- PHASE-2-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    hits = {'delta_H': 0, 'MC': 0, 'NLM': 0}
    ben_violations = []
    per_cell = []
    for H in (32, 128):
        for tname in tasks:
            sub = [r for r in rows if r['H'] == H]
            tvals = [r['t2'][tname] for r in sub]
            cell = dict(tag=f'H={H} {tname}')
            for key in ('delta_H', 'MC', 'NLM', 'ben1'):
                c = corr2([r['obs'][key] for r in sub], tvals)
                cell[key] = c
                if c == c:
                    if key in hits and c >= 0.4:
                        hits[key] += 1
                    if key == 'ben1' and abs(c) >= 0.4:
                        ben_violations.append(f'H={H} {tname}: {c:+.3f}')
            per_cell.append(cell)
            L.append(f"Zelle H={H} {tname}: corr(delta_H)={cell['delta_H']:+.3f} "
                     f"corr(MC)={cell['MC']:+.3f} corr(NLM)={cell['NLM']:+.3f} "
                     f"corr(ben1)={cell['ben1']:+.3f}")

    n_cells = len(per_cell)
    L.append('')
    L.append(f'PRIMAER delta_H: {hits["delta_H"]}/{n_cells} Zellen >= +0.4   (>= 8 noetig)')
    L.append(f'SEKUNDAER MC:   {hits["MC"]}/{n_cells} Zellen >= +0.4   (>= 6 noetig)')
    L.append(f'SEKUNDAER NLM:  {hits["NLM"]}/{n_cells} Zellen >= +0.4   (>= 6 noetig)')
    if ben_violations:
        L.append('MANIPULATIONS-CHECK VERLETZT: ' + '; '.join(ben_violations))
    else:
        L.append('MANIPULATIONS-CHECK: ben1 ueberall < 0.4 — Matching haelt.')
    means = {k: sum(c[k] for c in per_cell) / n_cells
             for k in ('delta_H', 'MC', 'NLM')}
    L.append(f"deskriptiv: mean corr delta_H={means['delta_H']:+.3f}, "
             f"MC={means['MC']:+.3f}, NLM={means['NLM']:+.3f}")

    if ben_violations:
        verdict = ('UNENTSCHEIDEN — Stabilitaets-Matching in frischen Seeds '
                   'durchbrochen (Design-Problem, Schwelle unangetastet).')
    elif hits['delta_H'] >= 8:
        verdict = ('PASS — delta_H ist die validierte Ressourcenachse: '
                   'Nicht-Normalitaet (Henrici) praediziert Aufgabenleistung '
                   'unter kontrollierter Stabilitaet, bestaetigt auf frischen '
                   'Seeds. Homeostat-Ziel (F32) wieder offen: Regler auf delta_H.')
    elif hits['delta_H'] >= 4:
        verdict = ('TEIL-PASS — delta_H generalisiert schwaecher als in Phase 1. '
                   'Fundament zu duenn fuer Regler-Bau; nur dokumentieren.')
    else:
        verdict = ('KILL — Phase-1-Signal war Garten der verzweigten Pfade. '
                   'Ressourcen-Wette endgueltig tot; Rueckstufung auf '
                   'F27-Antagonisten-Phaenomenologie (deskriptiv).')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} und {args.csv} geschrieben.')


if __name__ == '__main__':
    main()
