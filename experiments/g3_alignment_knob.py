#!/usr/bin/env python
'''G3 — AUSRICHTUNGS-KNOPF: IST DER GETRIEBENE EXPONENT EINE
AUSRICHTUNGS-GROESSE? (registriert 2026-11-06, VOR dem Lauf)

VORGESCHICHTE:
  G2 KILL: Bei fixem Spektrum schwankt der getriebene ben1 mit der
  Rotation der Eigenbasis SO STARK wie das gesamte Seed-Lotto
  (std_A ~ std_C bei H=128). Das spektrale Konstruktions-Programm
  ist damit tot. ABER: G2 hielt den ANTRIEB zufaellig. Der
  getriebene Exponent haengt an der Kopplung Antrieb <-> Basis.
  These: ben1 ist keine spektrale Groesse, sondern eine
  AUSRICHTUNGS-Groesse. Das Feld tunes die falsche Variable.

IDEE (konstruktiv, nicht nur diagnostisch):
  W = Q ( D + s*L ) Q^T, Spektrum D FIX (FIXED=9999), Q rotiert
  pro Seed (das G2-Schicksal bleibt aktiv!). Der Knopf ist die
  ANTRIEBS-AUSRICHTUNG U (Hx1, auf Einheitsnorm normiert):
    A "mitfahrend": U = Q[:,0] (dominanter Modus). Unter Rotation
      von Q rotiert U MIT — die Ausrichtung (und damit die ganze
      getriebene Dynamik in Basis-Koordinaten) ist INVARIANT.
      Vorhersage: Lotto kollabiert, Kalibrierung uebertraegt.
    B "zufaellig":  U = randn/norm (Feld-Standard). Ausrichtung
      variiert mit Q -> volles G2-Lotto.
    C "absolut":    U = e_0 in Absolutkoordinaten (fix). Ausrichtung
      variiert mit Q -> sollte wie B lottern (Mechanismus-Check:
      das B-Lotto IST ein Ausrichtungs-Lotto).

REGISTRIERUNG (VOR DEM LAUF GENAGELT, KEINE NACHVERHANDLUNG):
  Zellen: H in {32,128} x s in {0.0, 0.8}. Ziel ben=-0.30 +- 0.05.
  Kalibrierung: pro (H, s, Arm) EINMAL lam_max aus 10-Punkt-Raster
  (0.2..2.0) auf Seed 0, dann Bewertung mit 10 Seeds (0..9).
  PRIMAER (Ausrichtungs-These), PASS genau dann wenn in ALLEN
  4 Zellen:
    (i)   std_A <= 0.5 * std_B
    UND (ii)  Trefferquote A >= 0.6
    UND (iii) Trefferquote B < 0.3
  -> "Ausrichtung ist der echte, konstruierbare Knopf: ben1 laesst
     sich ueber die Antriebs-Ausrichtung steuern und uebertraegt
     seed-uebergreifend, wo spektrale Knoepfe lottern."
  KILL: Sonst — Ausrichtung allein traegt nicht; die These ist
    gefallen und G2 bleibt ein reiner Diagnose-Befund.
  UNENTSCHEIDEN: >30% ungueltige Punkte auf einer Vergleichsseite.
  SEKUNDAER: std_C >= 0.8 * std_B in >= 3/4 Zellen (unterstuetzt:
    das B-Lotto ist ein Ausrichtungs-Lotto, kein Skalen-Artefakt).
  DESKRIPTIV (nur Bericht): Ausrichtungs-Misch-Knopf w in 6 Stufen:
    U = w*q_0 + sqrt(1-w^2)*q_perp — Monotonie via Spearman(w, ben1).
  Instrument: getriebener Benettin exakt wie f34..g2 (IS aus
  f34_exploration, drive_input(4, 48, IS)).
'''
import argparse
import math
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
FIXED = 9999
W_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def factored_WQ(H, lam_max, s_nn, q_seed, spec_seed, noise_seed):
    '''Schur-Konstruktion, gibt (W, Q) zurueck — Q fuer den Antrieb.'''
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
    return Q @ T @ Q.T, Q


def make_model(H, lam, s_nn, seed, arm):
    '''Spektrum fix, Q rotiert mit seed, Antrieb je Arm.'''
    W, Q = factored_WQ(H, lam, s_nn, seed, FIXED, FIXED)
    model = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
    with torch.no_grad():
        model.W.copy_(W)
        if arm == 'A':
            u = Q[:, 0:1]
        elif arm == 'B':
            genU = torch.Generator().manual_seed(10000 + seed)
            u = torch.randn(H, 1, generator=genU)
            u = u / u.norm()
        elif arm == 'C':
            u = torch.zeros(H, 1)
            u[0, 0] = 1.0
        else:
            raise ValueError(arm)
        model.U.weight.copy_(u)
    model.W.requires_grad_(False)
    return model


def ben_of(m, xdr):
    spec, _ = benettin_spectrum(m, xdr)
    return float(spec[0])


def spearman(xs, ys):
    '''Spearman-Rangkorrelation ohne scipy.'''
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float('nan')


def main():
    p = argparse.ArgumentParser(description='G3 alignment knob')
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--out', default='results/g3_alignment_knob_results.txt')
    p.add_argument('--csv', default='results/g3_alignment_knob_table.csv')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, IS)
    L = []
    L.append('=' * 118)
    L.append('G3 — AUSRICHTUNGS-KNOPF (registriert: PASS nur wenn in ALLEN 4 Zellen')
    L.append('std_A <= 0.5*std_B UND Treffer A >= 0.6 UND Treffer B < 0.3; Ziel -0.30+-0.05)')
    L.append('Arme: A=U-mitfahrend(Q[:,0]), B=U-zufaellig(Feld-Standard), C=U-absolut(e_0); Q rotiert, Spektrum fix')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            knobs = {}
            for arm in ('A', 'B', 'C'):
                best_k, best_ben, best_gap = None, None, 1e9
                for k in KNOBS:
                    m = make_model(H, k, s_nn, 0, arm)
                    b = ben_of(m, xdr)
                    gap = abs(b - BEN_TARGET)
                    if b == b and gap < best_gap:
                        best_k, best_ben, best_gap = k, b, gap
                knobs[arm] = best_k
                L.append('Kalibrierung ' + arm + ': lam=' + str(best_k) +
                         ' ben1(seed0)=' + ('%+.3f' % best_ben if best_ben == best_ben else 'nan'))

            block = {}
            for arm in ('A', 'B', 'C'):
                bens = []
                for seed in seeds:
                    m = make_model(H, knobs[arm], s_nn, seed, arm)
                    bens.append(ben_of(m, xdr))
                ok = [b for b in bens if b == b and abs(b) < 1e6]
                st = float(torch.tensor(ok, dtype=torch.float64).std()) if len(ok) > 1 else float('nan')
                hits = sum(1 for b in ok if abs(b - BEN_TARGET) <= BEN_TOL)
                block[arm] = dict(std=st, hits=hits, n=len(bens),
                                  mean=float(torch.tensor(ok).mean()) if ok else float('nan'),
                                  bad=len(bens) - len(ok))
                L.append("Arm " + arm + " (lam=" + str(knobs[arm]) + "): std=%.4f "
                         "Treffer=%d/%d mean=%+.3f ungueltig=%d"
                         % (st, hits, len(bens), block[arm]['mean'], block[arm]['bad']))
                rows.append(dict(H=H, s=s_nn, arm=arm, lam=knobs[arm], std=st,
                                 hits=hits, n=len(bens), mean=block[arm]['mean'],
                                 bad=block[arm]['bad']))
            cell_stats[(H, s_nn)] = block

            # Deskriptiv: Ausrichtungs-Misch-Knopf auf Seed 0
            lam = knobs['A']
            W, Q = factored_WQ(H, lam, s_nn, 0, FIXED, FIXED)
            genP = torch.Generator().manual_seed(31337)
            qperp = torch.randn(H, 1, generator=genP)
            q0 = Q[:, 0:1]
            proj = (qperp.T @ q0) * q0
            qperp = qperp - proj
            qperp = qperp / qperp.norm()
            bens_w = []
            for w in W_GRID:
                u = w * q0 + math.sqrt(max(0.0, 1.0 - w * w)) * qperp
                m = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
                with torch.no_grad():
                    m.W.copy_(W)
                    m.U.weight.copy_(u)
                m.W.requires_grad_(False)
                bens_w.append(ben_of(m, xdr))
            sp = spearman(W_GRID, bens_w)
            bw = ' '.join('%.1f:%+.3f' % (w, b) for w, b in zip(W_GRID, bens_w))
            L.append('DESKRIPTIV Misch-Knob (Seed 0, lam=' + str(lam) + '): ' + bw)
            L.append('DESKRIPTIV Spearman(w, ben1) = %+.3f' % sp)
            Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(args.csv, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,lam,std_ben1,hits,n,mean_ben1,bad\n')
        for r in rows:
            fh.write('%d,%s,%s,%s,%.6g,%d,%d,%.6g,%d\n'
                     % (r['H'], r['s'], r['arm'], r['lam'], r['std'],
                        r['hits'], r['n'], r['mean'], r['bad']))

    L.append('')
    L.append('--- G3-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    pass_all = True
    sek_cells = 0
    for (H, s_nn), b in sorted(cell_stats.items()):
        stdA, stdB, stdC = b['A']['std'], b['B']['std'], b['C']['std']
        hA = b['A']['hits'] / max(1, b['A']['n'])
        hB = b['B']['hits'] / max(1, b['B']['n'])
        badA = b['A']['bad'] / max(1, b['A']['n'])
        badB = b['B']['bad'] / max(1, b['B']['n'])
        cond = (stdA == stdA and stdB == stdB and
                stdA <= 0.5 * stdB and hA >= 0.6 and hB < 0.3 and
                badA <= 0.3 and badB <= 0.3)
        sek_ok = (stdC == stdC and stdB == stdB and stdC >= 0.8 * stdB)
        sek_cells += 1 if sek_ok else 0
        L.append('Zelle H=%d s=%s: std_A=%.4f vs std_B=%.4f (<=0.5x: %s), '
                 'Treffer A=%.2f (>=0.6), Treffer B=%.2f (<0.3), std_C=%.4f '
                 '(>=0.8*stdB: %s) -> %s'
                 % (H, s_nn, stdA, stdB,
                    'JA' if stdA == stdA and stdA <= 0.5 * stdB else 'NEIN',
                    hA, hB, stdC, 'JA' if sek_ok else 'NEIN',
                    'ERFUELLT' if cond else 'NICHT ERFUELLT'))
        pass_all = pass_all and cond
    L.append('SEKUNDAER Bilanz: std_C >= 0.8*std_B in %d/4 Zellen (>=3 noetig)' % sek_cells)

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if gate_bad:
        verdict = 'UNENTSCHEIDEN — zu viele ungueltige Punkte, Design-Frage.'
    elif pass_all:
        verdict = ('PASS — Ausrichtung ist der echte Knopf: Bei fixem Spektrum '
                   'und rotierender Basis kollabiert das Seed-Lotto, sobald der '
                   'Antrieb mit der Basis mitfaehrt. Der getriebene Exponent '
                   'ist eine Ausrichtungs-Groesse, keine spektrale — das Feld '
                   'tunes die falsche Variable. G4: Ausrichtungs-Knopf gegen '
                   'Aufgabenleistung fahren.')
    else:
        verdict = ('KILL — Ausrichtung allein traegt nicht: Auch mitfahrender '
                   'Antrieb lottert oder trifft nicht. G2 bleibt Diagnose, '
                   'kein konstruierbarer Knopf.')
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + args.out + ' und ' + args.csv + ' geschrieben.')


if __name__ == '__main__':
    main()
