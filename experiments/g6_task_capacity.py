#!/usr/bin/env python
'''G6 — TRAEGT DER ANKER-KNOPF AUFGABENLEISTUNG?
Verzoegerte-Wiedergabe (Memory Capacity) unter N / NC / NF.
(registriert 2026-11-06, VOR dem Lauf)

VORGESCHICHTE:
  G5 PASS: Das Seed-Lotto des getriebenen ben1 ist ein
  Koordinaten-Artefakt der elementweisen tanh; mitfahrende
  Nichtlinearitaet (NC) kollabiert es in float64 auf
  Rundungsniveau (bis 1e12x), Standard (N) lottert.
  OFFENE FRAGE (der Nutzer hat sie gestellt): Kontrolliert die
  Symmetrie nur den EXPONENTEN oder auch die LEISTUNG? Das ist
  der Unterschied zwischen Kuriositaet und Konstruktions-Knopf.

G6-FRAGE: Kollabiert die Seed-Streuung der Aufgabenleistung
  (Memory Capacity, verzoegerte Wiedergabe) unter NC ebenfalls,
  und zu welchem Preis im Mittel?

ARME (identisch zu G5: Spektrum fix, Q rotiert pro Seed,
  Antrieb U=Q[:,0], bias-frei, float64, lam=1.0 fest — keine
  Kalibrierung, die Frage ist Varianz, nicht Treffer):
  N  tanh absolut (Feld-Standard) — Erwartung: Leistungs-Lotto.
  NC tanh mitfahrend (Q-System) — Erwartung: exakt identische
     Seeds -> Leistungs-Streuung kollabiert AUF NULL; Mittel =
     Seed-0-Leistung (kein Mittelungsgewinn, aber 100%
     Reproduzierbarkeit).
  NF tanh in festem Rahmen R0 — Erwartung: lottert wie N
     (Relativwinkel-Mechanismus auch auf Leistungsseite).

TASK (Standard-Instrument der Literatur, Jaeger 2001):
  Rausch-Eingang u(t), Readout via Ridge-Regression auf die
  Zustaende, Ziel u(t-k) fuer k in {1,2,4,8,16}; MC = Summe
  der squared correlations. T=300, Washout=50, Ridge 1e-6.
  Eingang: drive_input(1, 300, IS) — identisch fuer alle
  Arme/Seeds (nur die Reservoir-Geometrie variiert).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Zellen: H in {32,128} x s in {0.0, 0.8}. 10 Seeds (0..9).
  Primaer-Groesse: MC je Seed (Summe ueber k), std und mean
  ueber Seeds je Arm.
  PRIMAER (alle 4 Zellen):
    (i)   std_MC_NC <= 0.1 * std_MC_N  (Leistungs-Streuung
          kollabiert)
    UND (ii)  std_MC_N >= 0.02 (Baseline-Streuung ist real)
    UND (iii) mean_MC_NC >= mean_MC_N - std_MC_N (Robustheit
          ohne Preis ueber eine Seed-Sigma)
    PASS -> "Der Anker-Knopf traegt Aufgabenleistung:
       seed-robuste Reservoirs per Konstruktion — der Kollaps
       ist nicht nur eine Exponenten-Kuriositaet."
    KILL: Sonst — "Die Symmetrie kontrolliert den Exponenten,
       nicht die Leistung."
    UNENTSCHEIDEN: >30% ungueltige Messungen in einer
       Vergleichsseite.
  SEKUNDAER (Bericht, Schwelle registriert): NF lottert wie N:
    std_MC_NF >= 0.5 * std_MC_N in >= 3/4 Zellen (Relativwinkel
    auch auf Leistungsseite). Plus je-Delay-Kapazitaeten.
'''
import os
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import IS
from cgfn.field import drive_input

FIXED = 9999
FRAME_SEED = 4242
DELAYS = [1, 2, 4, 8, 16]
T_TOTAL = 300
WASHOUT = 50
RIDGE = 1e-6
DT = 0.1
SUBSTEPS = 5


def factored_WQT(H, lam_max, s_nn, q_seed, spec_seed, noise_seed):
    genQ = torch.Generator().manual_seed(q_seed)
    A = torch.randn(H, H, generator=genQ)
    Q, _ = torch.linalg.qr(A)
    genS = torch.Generator().manual_seed(spec_seed)
    others = lam_max * 0.9 * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)
    eigs = torch.cat([torch.tensor([lam_max]), others])
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


def rollout_states(H, W, Q, T, Uw, R0, arm, x_seq):
    '''Zustaende in Absolutkoordinaten, RK2 wie ContinuousRNN.'''
    B, Tn, _ = x_seq.shape
    z = torch.zeros(B, H)
    states = []
    for t in range(Tn):
        x_t = x_seq[:, t]

        def f(zz):
            if arm == 'N':
                return -zz + torch.tanh(zz) @ W.T + x_t @ Uw.T
            y = zz @ Q if arm == 'NC' else zz @ R0
            tt = torch.tanh(y)
            if arm == 'NC':
                return -zz + (tt @ T.T) @ Q.T + x_t @ Uw.T
            return -zz + (tt @ R0.T) @ W.T + x_t @ Uw.T

        for _ in range(SUBSTEPS):
            k1 = f(z)
            k2 = f(z + 0.5 * DT * k1)
            z = z + DT * k2
        states.append(z)
    return torch.stack(states, dim=1)  # (B, Tn, H)


def memory_capacity(states, x_seq, delays):
    '''MC = Summe der squared correlations u(t-k) ~ readout(z(t)).'''
    H = states.shape[-1]
    z = states[0]                     # (Tn, H)
    u = x_seq[0, :, 0]                # (Tn,)
    total = 0.0
    per_k = {}
    for k in delays:
        rows = slice(WASHOUT + k, u.shape[0])
        S = z[rows]                   # (n, H)
        y = u[WASHOUT:u.shape[0] - k]  # Ziel u(t-k)
        n, _ = S.shape
        A = S.T @ S + RIDGE * torch.eye(H)
        wout = torch.linalg.solve(A, S.T @ y)
        pred = S @ wout
        pc = pred - pred.mean()
        yc = y - y.mean()
        num = (pc * yc).sum()
        den = torch.sqrt((pc * pc).sum() * (yc * yc).sum())
        r2 = float((num / den) ** 2) if den > 0 else 0.0
        per_k[k] = r2
        total += r2
    return total, per_k


def main():
    out_path = 'results/g6_task_capacity_results.txt'
    csv_path = 'results/g6_task_capacity_table.csv'
    os.makedirs('results', exist_ok=True)
    seeds = list(range(10))
    x_seq = drive_input(1, T_TOTAL, IS)

    L = []
    L.append('=' * 118)
    L.append('G6 — TRAEGT DER ANKER-KNOPF AUFGABENLEISTUNG? (registriert: PASS nur wenn')
    L.append('in ALLEN 4 Zellen std_MC_NC<=0.1*std_MC_N UND std_MC_N>=0.02 UND')
    L.append('mean_MC_NC>=mean_MC_N-std_MC_N; NF-Sekundaer; lam=1.0, keine Kalibrierung)')
    L.append('Arme: N=tanh absolut, NC=tanh mitfahrend, NF=tanh fester Rahmen; MC-Summe k in 1,2,4,8,16')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            R0 = fixed_frame(H)
            block = {}
            for arm in ('N', 'NC', 'NF'):
                mcs, pers = [], []
                for seed in seeds:
                    W, Q, T = factored_WQT(H, 1.0, s_nn, seed, FIXED, FIXED)
                    Uw = Q[:, 0:1].contiguous()
                    states = rollout_states(H, W, Q, T, Uw, R0, arm, x_seq)
                    mc, per_k = memory_capacity(states, x_seq, DELAYS)
                    mcs.append(mc)
                    pers.append(per_k)
                ok = [v for v in mcs if v == v]
                st = float(torch.tensor(ok).std()) if len(ok) > 1 else float('nan')
                mean = float(torch.tensor(ok).mean()) if ok else float('nan')
                block[arm] = dict(std=st, mean=mean, n=len(mcs),
                                  bad=len(mcs) - len(ok))
                per0 = pers[0]
                pk = ' '.join('k=%d:%.3f' % (k, per0[k]) for k in DELAYS)
                L.append('Arm %s: MC mean=%.4f std=%.4f ungueltig=%d | Seed0 je-k: %s'
                         % (arm, mean, st, block[arm]['bad'], pk))
                rows.append(dict(H=H, s=s_nn, arm=arm, mean=mean, std=st,
                                 n=len(mcs), bad=block[arm]['bad']))
            cell_stats[(H, s_nn)] = block
            Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,mean_MC,std_MC,n,bad\n')
        for r in rows:
            fh.write('%d,%s,%s,%.6g,%.6g,%d,%d\n'
                     % (r['H'], r['s'], r['arm'], r['mean'], r['std'], r['n'], r['bad']))

    L.append('')
    L.append('--- G6-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    pass_all = True
    nf_cells = 0
    for (H, s_nn), b in sorted(cell_stats.items()):
        sN, sNC, sNF = b['N']['std'], b['NC']['std'], b['NF']['std']
        mN, mNC = b['N']['mean'], b['NC']['mean']
        c = (sNC == sNC and sN == sN and sNC <= 0.1 * sN
             and sN >= 0.02 and mNC == mNC and mNC >= mN - sN)
        nf_ok = (sNF == sNF and sN == sN and sNF >= 0.5 * sN)
        nf_cells += 1 if nf_ok else 0
        L.append('Zelle H=%d s=%s: std_NC=%.4f std_N=%.4f std_NF=%.4f mean_N=%.4f '
                 'mean_NC=%.4f -> %s'
                 % (H, s_nn, sNC, sN, sNF, mN, mNC,
                    'ERFUELLT' if c else 'NICHT ERFUELLT'))
        pass_all = pass_all and c
    L.append('SEKUNDAER NF-Bilanz: std_NF >= 0.5*std_N in %d/4 Zellen (>=3 Relativwinkel ok)'
             % nf_cells)

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if gate_bad:
        verdict = 'UNENTSCHEIDEN — zu viele ungueltige Messungen.'
    elif pass_all:
        verdict = ('PASS — Der Anker-Knopf traegt Aufgabenleistung: Unter '
                   'mitfahrender Nichtlinearitaet kollabiert die Seed-Streuung '
                   'der Memory Capacity (Summe k=1..16) auf nahe null, ohne '
                   'Preis ueber eine Seed-Sigma. Seed-robuste Reservoirs per '
                   'Konstruktion: eine Kalibrierung, alle Seeds reproduzieren. '
                   'Der G5-Befund ist kein Exponenten-Kuriosum, sondern ein '
                   'Leistungs-Griff.')
    else:
        verdict = ('KILL — Die Symmetrie kontrolliert den Exponenten, nicht '
                   'die Leistung: MC-Streuung kollabiert nicht (oder nur mit '
                   'Preis). Der Anker bleibt Diagnose, kein Engineering-Knopf.')
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + out_path + ' und ' + csv_path + ' geschrieben.')


if __name__ == '__main__':
    main()
