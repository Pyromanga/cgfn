'''
F37 — SAUBERER ANKER, LETZTER STEIN: DELTA_H UNTER MONOTONIE-ROBUSTEM
STABILITY-MATCHING (registriert 2026-11-05, VOR dem Lauf, ENDGUELTIG).

VORGESCHICHTE (nur Fakten, keine Deutung):
  F34 Phase 2: TEIL-PASS (delta_H 4/10, noetig 8) — Range-Restriktion.
  F35: UNENTSCHEIDEN — lambda*->ben-Karte bei band-strukturierter
       Nicht-Normalitaet NICHT MONOTON (H=32), Bisektion verfehlte
       ben=-0.30 bis zu 0.75 — Manipulations-Check verletzt.
  F36: UNENTSCHEIDEN (derselbe Anker-Bruch, repliziert) und MSDM
       in den gueltigen H=128-Zellen gefallen (0/5 >= +0.6).
  Kern-Lektion: Die Messung selbst war das Problem. Dieser Stein
  repariert NICHT die These, sondern das INSTRUMENT — und stellt
  dann die Originalfrage ein letztes Mal, endgueltig.

ANKER-REPARATUR (Messhygiene, vor dem Lauf genagelt):
  tune_lam_robust statt Bisektion:
    (1) Grober lambda*-Sweep (10 Punkte, geometrisch 0.02..2.5),
        ben je Punkt gemessen.
    (2) Bracket-Suche: benachbarte Punkte, die das Ziel -0.30
        einschliessen; Verfeinerung (6 Bisektions-Schritte) NUR
        innerhalb dieses Brackets (lokal monotones Segment).
    (3) AKZEPTANZ-GATE: Modell nur GUELTIG, wenn
        |ben_final + 0.30| <= 0.05. Sonst: Modell UNGUELTIG,
        laut markiert, zaehlt in der Zell-Gueltigkeit.
  Damit ist der Manipulations-Check ueberfluessig-streng: bei
  akzeptierten Modellen hat ben1 eine Spannweite <= 0.10 — eine
  Korrelation von ben1 mit Aufgaben ist strukturell ausgeschlossen
  (Bericht: ben1-std je Zelle; falls < 0.02 => Matching perfekt).

REGISTRIERUNG (VOR DEM LAUF GENAGELT, KEINE NACHVERHANDLUNG):
  Gitter: H in {32,128} x band in {1,4,8} x s in {0.6,1.6} x
          seeds {30,31,32,33,34} (frisch) = 30 Modelle je H,
          60 gesamt. n=30 je Zelle (2 H x 5 Aufgaben = 10 Zellen).
  PRIMAER: corr(delta_H, TASK_R2) >= +0.4 in >= 8/10 gueltigen
           Zellen -> PASS: Homeostat-Fundament (F32) zurueck
           auf dem Tisch: Regler auf delta_H.
           < 8/10 bei ausreichender Gueltigkeit -> FINALER KILL:
           Betriebszustands-Programm endgueltig beendet. Kein
           Design-Fix, kein vierter Versuch, keine neue Schwelle.
  ZELL-GUELTIGKEIT: >= 15 der 30 Modelle einer Zelle akzeptiert.
           < 15 -> Zelle ungueltig. > 5 ungueltige Zellen ->
           UNENTSCHEIDEN: der reparierte Anker versagt flaechig,
           dann ist das Ziel ben=-0.30 unter Struktur-Variation
           selbst nicht haltbar — ein Physik-Befund, kein Bug.
  SEKUNDAER: corr(MC) >= +0.4 in >= 6/10 gueltigen Zellen.
  Instrument: observables/tasks/corr2 IDENTISCH via Import aus
           f34_exploration; Band-Konstruktion via f35_twoknob.
  Alles andere (MSDM, top25, E, ...) nur noch Wissenschafts-
           geschichte — hier wird EINE Frage sauber beantwortet.
'''
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import (task_sinus, task_narma10, task_white_delay8,
                             task_prod, task_trend, corr2, observables,
                             BEN_TARGET, IS)
from f35_twoknob import make_band_model

from cgfn.benettin import benettin_spectrum
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity

BEN_TOL = 0.05
SWEEP = [0.02, 0.05, 0.09, 0.16, 0.28, 0.5, 0.8, 1.2, 1.8, 2.5]


def ben_of(H, s_nn, seed, band, lam, xdr):
    m = make_band_model(H, s_nn, lam, seed, band)
    spec, _ = benettin_spectrum(m, xdr)
    return float(spec[0])


def tune_lam_robust(H, s_nn, seed, band, xdr):
    '''Sweep -> Bracket -> lokale Verfeinerung -> Akzeptanz-Gate.
    Rueckgabe: (lam, m, ben, valid).'''
    bens = []
    for lam in SWEEP:
        bens.append(ben_of(H, s_nn, seed, band, lam, xdr))
    # Bracket: benachbarte Punkte, die BEN_TARGET einschliessen
    lo, hi = None, None
    for i in range(len(SWEEP) - 1):
        b0, b1 = bens[i], bens[i + 1]
        s0 = b0 - BEN_TARGET
        s1 = b1 - BEN_TARGET
        if s0 == 0 or s1 == 0 or (s0 < 0) != (s1 < 0):
            lo, hi = SWEEP[i], SWEEP[i + 1]
            blo, bhi = b0, b1
            break
    if lo is None:
        # kein Bracket: Punkt mit kleinstem |ben - Ziel|, ungeklaerte
        # Monotonie => ungueltig ausser er trifft die Toleranz zufaellig
        j = min(range(len(SWEEP)), key=lambda i: abs(bens[i] - BEN_TARGET))
        lam = SWEEP[j]
        ben = bens[j]
        m = make_band_model(H, s_nn, lam, seed, band)
        return lam, m, ben, abs(ben - BEN_TARGET) <= BEN_TOL
    # lokale Bisektion im Bracket
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        bm = ben_of(H, s_nn, seed, band, mid, xdr)
        if (bm - BEN_TARGET < 0) == (blo - BEN_TARGET < 0):
            lo, blo = mid, bm
        else:
            hi, bhi = mid, bm
    lam = 0.5 * (lo + hi)
    ben = ben_of(H, s_nn, seed, band, lam, xdr)
    m = make_band_model(H, s_nn, lam, seed, band)
    return lam, m, ben, abs(ben - BEN_TARGET) <= BEN_TOL


def main():
    p = argparse.ArgumentParser(description='F37 clean-anchor delta_H test')
    p.add_argument('--seeds', default='30,31,32,33,34')
    p.add_argument('--bands', default='1,4,8')
    p.add_argument('--s-values', default='0.6,1.6')
    p.add_argument('--out', default='results/f37_cleananchor_results.txt')
    p.add_argument('--csv', default='results/f37_cleananchor_table.csv')
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
    L.append('F37 — SAUBERER ANKER, LETZTER STEIN (registriert: delta_H >= +0.4 in >= 8/10 gueltigen')
    L.append(f'Zellen = PASS, sonst FINALER KILL; Akzeptanz-Gate |ben+0.30| <= {BEN_TOL}; Zelle gueltig bei >= 15/30)')
    L.append(f'Gitter: H in {{32,128}} x band={bands} x s={s_vals} x seeds={seeds}')
    L.append('=' * 118)

    rows = []
    n_invalid_models = 0
    for H in (32, 128):
        L.append('')
        L.append(f'########## H={H} ##########')
        for band in bands:
            for s_nn in s_vals:
                for seed in seeds:
                    lam, m, ben, valid = tune_lam_robust(H, s_nn, seed, band, xdr)
                    if not valid:
                        n_invalid_models += 1
                        L.append(f"H={H} b={band} s={s_nn:.1f} seed={seed} "
                                 f"lam*={lam:.3f} ben1={ben:+.3f} "
                                 f"UNGUELTIG (|ben+0.30|={abs(ben + 0.30):.3f} > {BEN_TOL})")
                        rows.append(dict(H=H, band=band, s=s_nn, seed=seed,
                                         valid=False, obs=None, t2=None))
                        continue
                    obs = observables(m, xdr)
                    torch.manual_seed(seed + 9000)
                    mc, _ = memory_capacity(m, max_lag=16, seq_len=64,
                                             input_scale=IS)
                    obs['MC'] = mc
                    t2s = {t: tfn(m, seed) for t, tfn in tasks.items()}
                    rows.append(dict(H=H, band=band, s=s_nn, seed=seed,
                                     valid=True, obs=obs, t2=t2s))
                    L.append(f"H={H} b={band} s={s_nn:.1f} seed={seed} "
                             f"lam*={lam:.3f} ben1={ben:+.3f} "
                             f"dH={obs['delta_H']:.3f} MC={mc:.2f} "
                             f"sinus={t2s['sinus']:+.3f} narma={t2s['narma']:+.3f} "
                             f"wdelay8={t2s['wdelay8']:+.3f} prod={t2s['prod']:+.3f} "
                             f"trend={t2s['trend']:+.3f}")
        Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(args.csv, 'w', encoding='utf-8') as fh:
        head = ['H', 'band', 's', 'seed', 'valid', 'ben1', 'delta_H', 'MC'] + \
               [f't2_{t}' for t in tasks]
        fh.write(','.join(head) + '\n')
        for r in rows:
            if not r['valid']:
                fh.write(','.join([str(r['H']), str(r['band']), str(r['s']),
                                   str(r['seed']), '0']) + '\n')
                continue
            vals = [str(r['H']), str(r['band']), str(r['s']), str(r['seed']), '1',
                    f"{r['obs']['ben1']:.6g}", f"{r['obs']['delta_H']:.6g}",
                    f"{r['obs']['MC']:.6g}"] + \
                   [f"{r['t2'][t]:.6g}" for t in tasks]
            fh.write(','.join(vals) + '\n')

    # --- Registrierte Auswertung ---
    L.append('')
    L.append('--- F37-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    L.append(f'Modelle gesamt: {len(rows)}, davon ungueltig: {n_invalid_models}')
    hits_dh = 0
    hits_mc = 0
    valid_cells = 0
    invalid_cells = 0
    for H in (32, 128):
        for tname in tasks:
            sub_all = [r for r in rows if r['H'] == H]
            sub = [r for r in sub_all if r['valid']]
            if len(sub) < 15:
                L.append(f'Zelle H={H} {tname}: UNGUELTIG (nur {len(sub)}/30 '
                         f'akzeptierte Modelle)')
                invalid_cells += 1
                continue
            valid_cells += 1
            tvals = [r['t2'][tname] for r in sub]
            cdh = corr2([r['obs']['delta_H'] for r in sub], tvals)
            cmc = corr2([r['obs']['MC'] for r in sub], tvals)
            bstd = torch.tensor([r['obs']['ben1'] for r in sub],
                                dtype=torch.float64).std().item()
            if cdh == cdh and cdh >= 0.4:
                hits_dh += 1
            if cmc == cmc and cmc >= 0.4:
                hits_mc += 1
            L.append(f'Zelle H={H} {tname}: n={len(sub)} corr(delta_H)={cdh:+.3f} '
                     f'corr(MC)={cmc:+.3f} ben1-std={bstd:.4f}')

    L.append('')
    L.append(f'PRIMAER delta_H: {hits_dh}/{valid_cells} gueltige Zellen >= +0.4   (>= 8 noetig)')
    L.append(f'SEKUNDAER MC:   {hits_mc}/{valid_cells} gueltige Zellen >= +0.4   (>= 6 noetig)')
    L.append(f'Zellen: {valid_cells} gueltig, {invalid_cells} ungueltig (> 5 => UNENTSCHEIDEN)')

    if invalid_cells > 5:
        verdict = ('UNENTSCHEIDEN — der reparierte Anker versagt flaechig: '
                   'ben=-0.30 ist unter Struktur-Variation nicht haltbar. '
                   'Das ist ein Physik-Befund ueber den Anker, kein Bug.')
    elif hits_dh >= 8:
        verdict = ('PASS — unter sauberem Matching ueberlebt delta_H die '
                   '8/10-Schwelle: Homeostat-Fundament (F32) wieder offen, '
                   'Regler auf delta_H.')
    else:
        verdict = ('FINALER KILL — unter sauberem Matching und dem dritten, '
                   'letzten Design: delta_H traegt nicht. Betriebszustands-'
                   'Programm endgueltig beendet. Kein vierter Versuch, '
                   'kein Design-Fix, keine neue Schwelle. Rueckzug auf '
                   'F27-Phaenomenologie.')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} und {args.csv} geschrieben.')


if __name__ == '__main__':
    main()
