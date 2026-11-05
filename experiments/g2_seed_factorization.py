#!/usr/bin/env python
'''G2 — SEED-RAUSCHEN-FAKTORISIERUNG: WORAUSS BESTEHT DAS LOTTO?
(registriert 2026-11-05, VOR dem Lauf)

VORGESCHICHTE:
  G1 (laeuft parallel) zeigt vorlaeufig: Schon der SYMMETRISCHE Arm
  streut um +-0.15 im getriebenen ben1, obwohl g=1 (null Nicht-
  Normalitaet). Der spektrale Knopf steuert EINEN Eigenwert; die
  anderen H-1 wandern pro Seed. Also: Ist das Seed-Rauschen des
  Reservoir-Feldes (mean+-std ueber Seeds, in jedem Paper der Welt)
  Geometrie-Rauschen oder Spektrum-Rauschen?
  Mit Zufallsmatrizen UNBEANTWORTBAR (Alles korreliert mit Allem).
  Mit der Schur-Konstruktion EXAKT FAKTORISIERBAR:
    W = Q ( D + s*L ) Q^T mit
      Q ... orthonormale Basis (q_seed)
      D ... Eigenwerte (spec_seed)
      L ... streng-obres Rauschen (noise_seed)
  Die drei Generatoren sind UNABHAENGIG schaltbar.

IDEE (offensichtlich, sobald man es sieht — und unveroeffentlicht):
  Das Seed-Lotto des Feldes zerfaellt in:
    A-Basis-Rauschen:  nur Q variiert (Spektrum+Rauschen fix)
    B-Spektrum-Rauschen: nur D variiert (Q+Rauschen fix)
    Bn-Geometrie-Rauschen: nur L variiert (Q+D fix, s>0)
    C-Referenz:        alles variiert (= make_designed_model)
    R-Referenz:        Feld-Standard-ESN (Gauss, spektral skaliert)
  Wenn das Lotto dominiert durch SPEKTRUM-Streuung und die
  Basis-Rotation fast nichts beitraegt, dann kann man das Lotto
  BEENDEN: volle Spektrums-Konstruktion statt Tuning —
  "Lyapunov by construction": einmal kalibrieren, alle Seeds treffen.

REGISTRIERUNG (VOR DEM LAUF GENAGELT, KEINE NACHVERHANDLUNG):
  Zellen: H in {32,128} x s in {0.0, 0.8}. Ziel ben=-0.30.
  Kalibrierung: pro (H, s) EINMAL knob aus 10-Punkt-Raster
  (0.2..2.0) auf Seed 0 des jeweiligen Arms (festes Spektrum/
  festes Rauschen), dann Bewertung mit 10 Seeds (0..9).
  PRIMAER (Spektrum-Lotto-These), PASS genau dann wenn in ALLEN
  4 Zellen (H x s):
    (i)  std_A <= 0.5 * std_C   (Basis-Rauschen ist NICHT das
        Schicksal — Rotieren allein bringt den Exponenten kaum
        zum Schwanken)
    UND (ii) Kalibrier-Trefferquote A >= 0.6 (|ben+0.30| <= 0.05)
        UND Kalibrier-Trefferquote C < 0.3
    -> "Seed-Rauschen ist Spektrum-Rauschen; Konstruktion beendet
       das Tuning-Lotto" — Lyapunov by construction.
  KILL: Sonst — Basis-Geometrie ist irreduzibles Schicksal, auch
    Konstruktion hilft nicht. (Auch klar, auch brauchbar.)
  UNENTSCHEIDEN: >30% ungueltige Punkte in einer Vergleichsseite.
  SEKUNDAER (nur Bericht): Varianz-Anteile der Arme A, B, Bn an
  C; Trefferquote des Feld-ESN (R) nach eigener Kalibrierung.
  Instrument: getriebener Benettin exakt wie f34..g1.
'''
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import IS
from cgfn.benettin import benettin_spectrum
from cgfn.field import drive_input
from cgfn.model import ContinuousRNN

BEN_TARGET = -0.30
BEN_TOL = 0.05
KNOBS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
FIXED = 9999   # Generator-Seed fuer "fest" (konventionsgemaess)


def recurrent_matrix_factored(H, lam_max, s_nn, q_seed, spec_seed, noise_seed):
    '''Schur-Konstruktion mit UNABHAENGIGEN Generatoren fuer Q, D, L.'''
    genQ = torch.Generator().manual_seed(q_seed)
    A = torch.randn(H, H, generator=genQ)
    Q, _ = torch.linalg.qr(A)
    genS = torch.Generator().manual_seed(spec_seed)
    others = lam_max * 0.9 * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)
    eigs = torch.cat([torch.tensor([lam_max], dtype=torch.float32), others])
    T = torch.diag(eigs)
    if s_nn > 0:
        genN = torch.Generator().manual_seed(noise_seed)
        upper = torch.triu(torch.randn(H, H, generator=genN), diagonal=1)
        T = T + s_nn * lam_max * upper
    return Q @ T @ Q.T


def make_factored_model(H, lam, s_nn, q_seed, spec_seed, noise_seed):
    model = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
    W = recurrent_matrix_factored(H, lam, s_nn, q_seed, spec_seed, noise_seed)
    with torch.no_grad():
        model.W.copy_(W)
    model.W.requires_grad_(False)
    return model


def make_random_esn(H, rho, seed):
    model = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(H, H, generator=gen)
    with torch.no_grad():
        model.W.copy_(W * rho)
    model.W.requires_grad_(False)
    return model


def ben_of(m, xdr):
    spec, _ = benettin_spectrum(m, xdr)
    return float(spec[0])


def main():
    p = argparse.ArgumentParser(description='G2 seed noise factorization')
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--out', default='results/g2_seed_factorization_results.txt')
    p.add_argument('--csv', default='results/g2_seed_factorization_table.csv')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, IS)
    L = []
    L.append('=' * 118)
    L.append('G2 — SEED-RAUSCHEN-FAKTORISIERUNG (registriert: PASS nur wenn in ALLEN 4 Zellen')
    L.append(f'std_A <= 0.5*std_C UND Treffer A >= 0.6 UND Treffer C < 0.3; Ziel {BEN_TARGET}+-{BEN_TOL})')
    L.append(f'Arme: A=nur-Basis, B=nur-Spektrum, Bn=nur-Rauschen(s>0), C=alles, R=Feld-ESN; Seeds={seeds}')
    L.append('=' * 118)

    def arm_cells(H, s_nn, kind, knob, seed):
        if kind == 'A':
            return make_factored_model(H, knob, s_nn, seed, FIXED, FIXED)
        if kind == 'B':
            return make_factored_model(H, knob, s_nn, FIXED, seed, FIXED)
        if kind == 'Bn':
            return make_factored_model(H, knob, s_nn, FIXED, FIXED, seed)
        if kind == 'C':
            return make_factored_model(H, knob, s_nn, seed, seed, seed)
        raise ValueError(kind)

    rows = []
    cell_stats = {}
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append(f'########## H={H} s={s_nn} ##########')
            # --- Kalibrierung: knob je Arm auf Seed 0 ---
            knobs = {}
            for kind in ('A', 'B', 'Bn', 'C'):
                best_k, best_ben, best_gap = None, None, 1e9
                for k in KNOBS:
                    m = arm_cells(H, s_nn, kind, k, 0)
                    b = ben_of(m, xdr)
                    gap = abs(b - BEN_TARGET)
                    if b == b and gap < best_gap:
                        best_k, best_ben, best_gap = k, b, gap
                knobs[kind] = best_k
                L.append(f'Kalibrierung {kind}: knob={best_k} ben1(seed0)={best_ben:+.3f}')
            # Feld-ESN: eigene Kalibrierung
            best_r, best_rb, best_rg = None, None, 1e9
            for k in KNOBS:
                m = make_random_esn(H, k, 0)
                b = ben_of(m, xdr)
                gap = abs(b - BEN_TARGET)
                if b == b and gap < best_rg:
                    best_r, best_rb, best_rg = k, b, gap
            L.append(f'Kalibrierung R(esn): rho={best_r} ben1(seed0)={best_rb:+.3f}')

            block = {}
            for kind in ('A', 'B', 'Bn', 'C'):
                bens = []
                for seed in seeds:
                    m = arm_cells(H, s_nn, kind, knobs[kind], seed)
                    bens.append(ben_of(m, xdr))
                ok = [b for b in bens if b == b and abs(b) < 1e6]
                st = float(torch.tensor(ok, dtype=torch.float64).std()) if len(ok) > 1 else float('nan')
                hits = sum(1 for b in ok if abs(b - BEN_TARGET) <= BEN_TOL)
                block[kind] = dict(std=st, hits=hits, n=len(bens),
                                   mean=float(torch.tensor(ok).mean()) if ok else float('nan'),
                                   bad=len(bens) - len(ok))
                L.append(f"Arm {kind:>2s} (knob={knobs[kind]}): std={st:.4f} "
                         f"Treffer={hits}/{len(bens)} mean={block[kind]['mean']:+.3f} "
                         f"ungueltig={block[kind]['bad']}")
            rbens = []
            for seed in seeds:
                m = make_random_esn(H, best_r, seed)
                rbens.append(ben_of(m, xdr))
            rok = [b for b in rbens if b == b and abs(b) < 1e6]
            rst = float(torch.tensor(rok, dtype=torch.float64).std()) if len(rok) > 1 else float('nan')
            rhits = sum(1 for b in rok if abs(b - BEN_TARGET) <= BEN_TOL)
            block['R'] = dict(std=rst, hits=rhits, n=len(rbens),
                              mean=float(torch.tensor(rok).mean()) if rok else float('nan'),
                              bad=len(rbens) - len(rok))
            L.append(f"Arm R  (rho={best_r}): std={rst:.4f} Treffer={rhits}/{len(rbens)} "
                     f"mean={block['R']['mean']:+.3f} ungueltig={block['R']['bad']}")
            cell_stats[(H, s_nn)] = block
            for kind, b in block.items():
                rows.append(dict(H=H, s=s_nn, arm=kind, std=b['std'],
                                 hits=b['hits'], n=b['n'], mean=b['mean'],
                                 bad=b['bad']))
            Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(args.csv, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,std_ben1,hits,n,mean_ben1,bad\n')
        for r in rows:
            fh.write(f"{r['H']},{r['s']},{r['arm']},{r['std']:.6g},"
                     f"{r['hits']},{r['n']},{r['mean']:.6g},{r['bad']}\n")

    # --- Registrierte Auswertung ---
    L.append('')
    L.append('--- G2-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    pass_all = True
    for (H, s_nn), b in sorted(cell_stats.items()):
        stdA, stdC = b['A']['std'], b['C']['std']
        hA = b['A']['hits'] / max(1, b['A']['n'])
        hC = b['C']['hits'] / max(1, b['C']['n'])
        badA = b['A']['bad'] / max(1, b['A']['n'])
        badC = b['C']['bad'] / max(1, b['C']['n'])
        cond = (stdA == stdA and stdC == stdC and
                stdA <= 0.5 * stdC and
                hA >= 0.6 and hC < 0.3 and
                badA <= 0.3 and badC <= 0.3)
        L.append(f"Zelle H={H} s={s_nn}: std_A={stdA:.4f} vs std_C={stdC:.4f} "
                 f"(<=0.5x: {'JA' if stdA == stdA and stdA <= 0.5 * stdC else 'NEIN'}), "
                 f"Treffer A={hA:.2f} (>=0.6), Treffer C={hC:.2f} (<0.3) -> "
                 f"{'ERFUELLT' if cond else 'NICHT ERFUELLT'}")
        pass_all = pass_all and cond

    for (H, s_nn), b in sorted(cell_stats.items()):
        L.append(f"SEKUNDAER H={H} s={s_nn}: Varianz-Anteile (relativ zu C): "
                 f"A={b['A']['std']:.3f} B={b['B']['std']:.3f} "
                 f"{'Bn=' + format(b['Bn']['std'], '.3f') if s_nn > 0 else ''} "
                 f"C={b['C']['std']:.3f} R={b['R']['std']:.3f}; "
                 f"Treffer R(esn)={b['R']['hits']}/{b['R']['n']}")

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if gate_bad:
        verdict = ('UNENTSCHEIDEN — zu viele ungueltige Punkte, Design-Frage.')
    elif pass_all:
        verdict = ('PASS — Seed-Rauschen ist Spektrum-Rauschen: Die Basis-'
                   'Rotation ist NICHT das Schicksal, und volle Spektral-'
                   'Konstruktion trifft das Lyapunov-Ziel nach EINER '
                   'Kalibrierung reproduzierbar. Das Tuning-Lotto des '
                   'Feldes ist ein Spektrum-Lotto — und Konstruktion '
                   'beendet es. G3: Konstruktions-Platzierung inAufgabe-'
                   'leistung ueberfuehren.')
    else:
        verdict = ('KILL — Basis-Geometrie ist irreduzibles Schicksal: '
                   'auch bei fixem Spektrum schwankt der Exponent mit der '
                   'Rotation. Konstruktion allein beendet das Lotto nicht.')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} und {args.csv} geschrieben.')


if __name__ == '__main__':
    main()
