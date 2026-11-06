#!/usr/bin/env python
'''G4 — NICHTLINEARITAETS-ANKER: DAS SEED-LOTTO IST EIN
KOORDINATEN-ARTEFAKT DER ELEMENTWEISEN TANH. (registriert 2026-11-06)

DER SPARK (Symmetrie-Argument, am Code verifiziert):
  G2-Arme hielten Spektrum+Rauschen fix und rotierten NUR Q:
  W_s = Q_s T Q_s^T — alle Seeds ORTHOGONAL AEHNLICH. Das
  getriebene System dz/dt = -z/tau + W tanh(z) + U u(t) waere
  unter Rotation INVARIANT, WENN die Nichtlinearitaet mit-
  rotieren wuerde. Tut sie nicht: torch.tanh ist ELEMENTWEISE,
  es sitzt in ABSOLUTKOORDINATEN und definiert damit die einzigen
  festen Achsen im Raum. Die Eigenbasis rotiert pro Seed gegen
  diese festen tanh-Achsen — genau DORT lebt das G2-Schicksal.
  Das erklaert rueckwirkend G1 (Spektrum ohne Autoritaet: das
  gain-Medium ist das sech^2-Muster entlang der Trajektorie,
  nicht das Spektrum), G2 (Basis-Schicksal = Rotation gegen den
  Anker) und den G3-KILL (U-mitfahrend reicht nicht: der Anker
  blieb unberuehrt — G3 testete nie das eigentliche Objekt).

G4-FRAGE: Kollabiert das Seed-Lotto EXAKT, wenn die tanh-Achsen
  mit der Basis mitfahren? Und zwar auf Maschinengenauigkeit,
  nicht nur statistisch?

ARME (Spektrum D fix, Rauschen L fix, Q rotiert pro Seed — das
  G2-Schicksal bleibt maximal aktiv; Antrieb ueberall U=Q[:,0]):
  L  linear:          tanh -> Identitaet. Kontrollarm: das
     lineare getriebene System ist orthogonal-aehnlich-invariant,
     ben1 = max Re eig(W) - 1/tau, seed-unabhaengig. std ~ 0
     beweist das Instrument.
  N  Feld-Standard:   tanh absolut (identisch zu G3-Arm A —
     Kontinuitaet zu den G3-Daten). Vorhersage: Lotto.
  NF fester Rahmen:   tanh im festen Zufalls-Rahmen R0 (QR-Seed
     4242). Rahmen rotiert NICHT mit Q. Vorhersage: lottert
     wie N — beweist, dass der RELATIVWINKEL Rahmen-vs-Basis
     der Mechanismus ist.
  NC mitfahrend:      tanh im Q_s-System: y = Q_s^T z,
     tanh(y), zurueckrotieren. In y-Koordinaten ist das System
     byte-identisch ueber alle Seeds (T fix, e_0-Antrieb fix,
     IC y=0). Vorhersage: ben1 EXAKT gleich fuer alle Seeds,
     std auf fp32-Niveau, Kalibrierung uebertraegt trivial.

REGISTRIERUNG (VOR DEM LAUF GENAGELT, KEINE NACHVERHANDLUNG):
  Zellen: H in {32,128} x s in {0.0, 0.8}. Ziel ben=-0.30+-0.05.
  Kalibrierung: pro (H, s, Arm) EINMAL lam_max aus 10-Punkt-Raster
  (0.2..2.0) auf Seed 0, dann 10 Seeds (0..9).
  PRIMAER (Anker-These), PASS genau dann wenn in ALLEN 4 Zellen:
    (i)   std_NC <= 0.01 UND maxdev_NC = max_s |ben(s)-ben(0)| <= 0.05
    UND (ii)  std_L <= 0.01 (Instrument-Kontrollarm haelt die
        Symmetrie — sonst ist das Messgeraet blind)
    UND (iii) std_N >= 0.02 (Kontrast real: Standard lottert)
    UND (iv)  std_NF >= 0.5 * std_N (fester Rahmen lottert wie
        Standard — Relativwinkel ist der Mechanismus)
  -> "Das Seed-Lotto des getriebenen Reservoirs ist ein
     Koordinaten-Artefakt: die elementweise Nichtlinearitaet
     definiert absolute Achsen, gegen die die Reservoir-Geometrie
     pro Seed rotiert. Mitfahrende Nichtlinearitaet kollabiert es
     exakt. Konstruktions-Knopf: rotations-aquivariante
     Nichtlinearitaet."
  KILL: Sonst — auch der Anker traegt nicht exakt; die These ist
    gefallen und das Feld bleibt holistisch-diagnostisch.
  UNENTSCHEIDEN: Instrument-Check (lokaler vs. Bibliotheks-Benettin
    auf Arm N, |Delta| <= 1e-3) faellt durch ODER >30% ungueltig.
  INSTRUMENT: lokaler Benettin 1:1 nach cgfn/benettin (RK2,
    substeps=5, QR pro Substep, Batch-Mittel), NUR die dyn-Paar-
    Funktion wird je Arm ausgetauscht. Check gegen die Bibliothek
    wird pro Zelle geloggt und gegatet.
  SEKUNDAER (nur Bericht): Trefferquoten je Arm; Kalibrierungslams.
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
FIXED = 9999
FRAME_SEED = 4242


def factored_WQT(H, lam_max, s_nn, q_seed, spec_seed, noise_seed):
    '''Schur-Konstruktion, gibt (W, Q, T) zurueck.'''
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
    return Q @ T @ Q.T, Q, T


def fixed_frame(H):
    genR = torch.Generator().manual_seed(FRAME_SEED)
    A = torch.randn(H, H, generator=genR)
    R, _ = torch.linalg.qr(A)
    return R


def benettin_custom(dyn, x_seq, H, dt, substeps, tau):
    '''Lokaler Benettin 1:1 wie cgfn/benettin, dyn austauschbar.'''
    B, T, _ = x_seq.shape
    z = torch.zeros(B, H)
    dz = torch.eye(H).unsqueeze(0).expand(B, H, H).contiguous()
    log_sum = torch.zeros(H)
    t_total = 0.0
    for t in range(T):
        x_t = x_seq[:, t]
        for _ in range(substeps):
            f1, g1 = dyn(z, dz, x_t)
            z2 = z + 0.5 * dt * f1
            dz2 = dz + 0.5 * dt * g1
            f2, g2 = dyn(z2, dz2, x_t)
            z = z + dt * f2
            dz = dz + dt * g2
            t_total += dt
            Q_, R_ = torch.linalg.qr(dz)
            diag_r = torch.diagonal(R_, dim1=-2, dim2=-1).abs()
            log_sum += torch.log(diag_r.clamp_min(1e-30)).mean(dim=0)
            dz = Q_
    spectrum = (log_sum / max(t_total, 1e-9)).sort(descending=True).values
    return float(spectrum[0])


def make_dyn(H, tau, W, Q, T, Uw, R0, arm):
    '''dyn(z, dz, x_t) -> (f, g) je Arm.'''
    Uin = lambda x_t: x_t @ Uw.T  # noqa: E731

    def sech2(y):
        return 1.0 - torch.tanh(y).pow(2)

    if arm == 'L':
        def dyn(z, dz, x_t):
            f = -z / tau + z @ W.T + Uin(x_t)
            g = -dz / tau + torch.einsum('ij,bjk->bik', W, dz)
            return f, g
        return dyn
    if arm == 'N':
        def dyn(z, dz, x_t):
            f = -z / tau + torch.tanh(z) @ W.T + Uin(x_t)
            g = -dz / tau + torch.einsum('ij,bjk->bik', W,
                                         sech2(z).unsqueeze(-1) * dz)
            return f, g
        return dyn
    if arm == 'NF':
        def dyn(z, dz, x_t):
            y = z @ R0
            t = torch.tanh(y)
            f = -z / tau + (t @ R0.T) @ W.T + Uin(x_t)
            inner = sech2(y).unsqueeze(-1) * (dz @ R0)
            g = -dz / tau + (inner @ R0.T) @ W.T
            return f, g
        return dyn
    if arm == 'NC':
        def dyn(z, dz, x_t):
            y = z @ Q
            t = torch.tanh(y)
            f = -z / tau + (t @ T.T) @ Q.T + Uin(x_t)
            inner = sech2(y).unsqueeze(-1) * (dz @ Q)
            w = torch.einsum('ij,bjk->bik', T, inner)
            g = -dz / tau + w @ Q.T
            return f, g
        return dyn
    raise ValueError(arm)


def make_dyn_for(H, s_nn, seed, arm, lam):
    '''Baut (dyn, model) — model nur fuer den Bibliotheks-Check (Arm N).'''
    W, Q, T = factored_WQT(H, lam, s_nn, seed, FIXED, FIXED)
    Uw = Q[:, 0:1].contiguous()
    R0 = fixed_frame(H)
    dyn = make_dyn(H, 1.0, W, Q, T, Uw, R0, arm)
    model = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
    with torch.no_grad():
        model.W.copy_(W)
        model.U.weight.copy_(Uw)
    model.W.requires_grad_(False)
    return dyn, model


def main():
    p = argparse.ArgumentParser(description='G4 nonlinearity anchor')
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--out', default='results/g4_nonlinearity_anchor_results.txt')
    p.add_argument('--csv', default='results/g4_nonlinearity_anchor_table.csv')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, IS)
    L = []
    L.append('=' * 118)
    L.append('G4 — NICHTLINEARITAETS-ANKER (registriert: PASS nur wenn in ALLEN 4 Zellen')
    L.append('std_NC<=0.01 UND maxdev_NC<=0.05 UND std_L<=0.01 UND std_N>=0.02 UND')
    L.append('std_NF>=0.5*std_N; Ziel -0.30+-0.05; Q rotiert, Spektrum fix, U=Q[:,0])')
    L.append('Arme: L=linear, N=tanh absolut (=G3-A), NF=tanh fester Rahmen R0, NC=tanh mitfahrend (Q-System)')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    inst_ok = True
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            # Instrument-Check: lokal vs. Bibliothek auf Arm N, Seed 0, lam=1.0
            dynN, modelN = make_dyn_for(H, s_nn, 0, 'N', 1.0)
            ben_lib = float(benettin_spectrum(modelN, xdr)[0][0])
            ben_loc = benettin_custom(dynN, xdr, H, 0.1, 5, 1.0)
            d_inst = abs(ben_lib - ben_loc)
            L.append('Instrument-Check Arm N (lam=1.0): Bibliothek=%.6f lokal=%.6f Delta=%.6f'
                     % (ben_lib, ben_loc, d_inst))
            if d_inst > 1e-3:
                inst_ok = False

            knobs = {}
            for arm in ('L', 'N', 'NF', 'NC'):
                best_k, best_ben, best_gap = None, None, 1e9
                for k in KNOBS:
                    dyn, _ = make_dyn_for(H, s_nn, 0, arm, k)
                    b = benettin_custom(dyn, xdr, H, 0.1, 5, 1.0)
                    gap = abs(b - BEN_TARGET)
                    if b == b and gap < best_gap:
                        best_k, best_ben, best_gap = k, b, gap
                knobs[arm] = best_k
                L.append('Kalibrierung ' + arm + ': lam=' + str(best_k) +
                         ' ben1(seed0)=' + ('%+.4f' % best_ben if best_ben == best_ben else 'nan'))

            block = {}
            for arm in ('L', 'N', 'NF', 'NC'):
                bens = []
                for seed in seeds:
                    dyn, _ = make_dyn_for(H, s_nn, seed, arm, knobs[arm])
                    bens.append(benettin_custom(dyn, xdr, H, 0.1, 5, 1.0))
                ok = [b for b in bens if b == b and abs(b) < 1e6]
                st = float(torch.tensor(ok, dtype=torch.float64).std()) if len(ok) > 1 else float('nan')
                b0 = bens[0] if bens and bens[0] == bens[0] else float('nan')
                maxdev = max((abs(b - b0) for b in ok), default=float('nan'))
                hits = sum(1 for b in ok if abs(b - BEN_TARGET) <= BEN_TOL)
                block[arm] = dict(std=st, maxdev=maxdev, hits=hits, n=len(bens),
                                  mean=float(torch.tensor(ok).mean()) if ok else float('nan'),
                                  b0=b0, bad=len(bens) - len(ok))
                L.append('Arm %s (lam=%s): std=%.4f maxdev=%.4f Treffer=%d/%d '
                         'mean=%+.3f ungueltig=%d'
                         % (arm, knobs[arm], st, maxdev, hits, len(bens),
                            block[arm]['mean'], block[arm]['bad']))
                rows.append(dict(H=H, s=s_nn, arm=arm, lam=knobs[arm], std=st,
                                 maxdev=maxdev, hits=hits, n=len(bens),
                                 mean=block[arm]['mean'], bad=block[arm]['bad']))
            cell_stats[(H, s_nn)] = block
            Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(args.csv, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,lam,std_ben1,maxdev,hits,n,mean_ben1,bad\n')
        for r in rows:
            fh.write('%d,%s,%s,%s,%.6g,%.6g,%d,%d,%.6g,%d\n'
                     % (r['H'], r['s'], r['arm'], r['lam'], r['std'],
                        r['maxdev'], r['hits'], r['n'], r['mean'], r['bad']))

    L.append('')
    L.append('--- G4-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    pass_all = True
    for (H, s_nn), b in sorted(cell_stats.items()):
        cNC = (b['NC']['std'] == b['NC']['std'] and b['NC']['std'] <= 0.01
               and b['NC']['maxdev'] == b['NC']['maxdev'] and b['NC']['maxdev'] <= 0.05)
        cL = (b['L']['std'] == b['L']['std'] and b['L']['std'] <= 0.01)
        cN = (b['N']['std'] == b['N']['std'] and b['N']['std'] >= 0.02)
        cNF = (b['NF']['std'] == b['NF']['std'] and b['N']['std'] == b['N']['std']
               and b['NF']['std'] >= 0.5 * b['N']['std'])
        cond = cNC and cL and cN and cNF
        L.append('Zelle H=%d s=%s: std_NC=%.4f(maxdev %.4f) std_L=%.4f std_N=%.4f '
                 'std_NF=%.4f -> NC:%s L:%s N:%s NF:%s => %s'
                 % (H, s_nn, b['NC']['std'], b['NC']['maxdev'], b['L']['std'],
                    b['N']['std'], b['NF']['std'],
                    'JA' if cNC else 'NEIN', 'JA' if cL else 'NEIN',
                    'JA' if cN else 'NEIN', 'JA' if cNF else 'NEIN',
                    'ERFUELLT' if cond else 'NICHT ERFUELLT'))
        pass_all = pass_all and cond

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if not inst_ok:
        verdict = 'UNENTSCHEIDEN — Instrument-Check (lokal vs. Bibliothek) gefallen.'
    elif gate_bad:
        verdict = 'UNENTSCHEIDEN — zu viele ungueltige Punkte, Design-Frage.'
    elif pass_all:
        verdict = ('PASS — Das Seed-Lotto ist ein Koordinaten-Artefakt: Die '
                   'elementweise tanh definiert absolute Achsen; rotiert die '
                   'Reservoir-Basis pro Seed gegen sie, lottert der getriebene '
                   'Exponent (G2-Schicksal). Faehrt die Nichtlinearitaet mit, '
                   'kollabiert das Lotto AUF MASCHINENGNAUE. Der getriebene '
                   'Exponent ist eine Anker-Geometrie-Groesse: Knopf = '
                   'relativer Winkel tanh-Achsen vs. Basis. Konstruktion: '
                   'rotations-aquivariante Nichtlinearitaet beendet das '
                   'Seed-Tuning. G5: Anker-Knopf gegen Aufgabenleistung.')
    else:
        verdict = ('KILL — Mitfahrende Nichtlinearitaet kollabiert das Lotto '
                   'NICHT exakt: Auch der Anker traegt nicht allein; das Feld '
                   'bleibt holistisch.')
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + args.out + ' und ' + args.csv + ' geschrieben.')


if __name__ == '__main__':
    main()
