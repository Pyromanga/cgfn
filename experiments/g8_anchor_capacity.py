#!/usr/bin/env python
'''G8 — IST DER ZUFALLS-ANKER EIN KAPAZITAETS-SUBOPTIMUM?
Partielle Ausrichtung vs. Feld-Standard, registriert. (2026-11-06)

VORGESCHICHTE:
  G7 (KILL, registriert): Der Anker-Winkel ist KEIN kontinuier-
  licher Zuverlaessigkeits-Regler — die Varianz kollabiert als
  Kliff bei w=1 (und dort stirbt die Kapazitaet, G6). ABER als
  UNREGISTRIERTE Beobachtung: In den s=0-Zellen war
  mean_MC(w=0.75) HOEHER als der Feld-Standard (H=32: 1.08 vs
  0.95, +14%; H=128: 1.10 vs 0.80, +38%) UND std kleiner.
  Das ist eine neue, eigene These und braucht einen eigenen
  registrierten Stein mit mehr Seeds.

G8-THESE: In Normal-Zellen (s=0) ist der Zufalls-Anker des
  Feldes ein KAPAZITAETS-SUBOPTIMUM: Eine feste partielle
  Ausrichtung des Nichtlinearitaets-Rahmens an die Reservoir-
  Basis liefert MEHR Memory Capacity ALS Zufall — ohne
  Zuverlaessigkeitsverlust. Das Feld baut nicht nur zu
  streuungsreich, sondern schlicht unterdurchschnittlich.

DESIGN (Messung wie G6/G7, nur Zustands-Ebene — der G7-Tangenten-
  Bug (Chimaere aus gerahmter Trajektorie + Standard-Tangent)
  ist dokumentiert und hier irrelevant, weil nur f, nie g,
  benutzt wird):
  Schur wie gehabt: W = Q T Q^T, Spektrum fix, Q rotiert pro
  Seed, Antrieb U=Q[:,0], BIAS GENULLT (Lehre L1), float64.
  Rahmen F(w) = Q @ QR(w*I + (1-w)*P0), P0 = Q^T R0 (G7).
  Zellen: H in {32,128} x s in {0.0, 0.8}.
  w-Gitter: {0.0, 0.6, 0.75, 0.9, 1.0} (Lokalisierung des Peaks).
  Seeds: 0..19 (20 Seeds, doppelt so viele wie G7).
  Task: Memory Capacity, k in {1,2,4,8,16}, T=300, Washout=50,
  Ridge 1e-6.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  PRIMAER, PASS genau dann wenn in BEIDEN s=0-Zellen:
    (i)   mean_MC(w=0.75) >= 1.05 * mean_MC(w=0.0)
          (Kapazitaetsgewinn gegen Zufall bestaetigt)
    UND (ii)  std_MC(w=0.75) <= std_MC(w=0.0)
          (ohne Zuverlaessigkeitsverlust)
  PASS -> "Der Zufalls-Anker ist ein Kapazitaets-Suboptimum:
     partielle Ausrichtung (w*=0.75 Gegend) ist der bessere
     Default. Richtig bauen heisst: Bias nullen + Anker
     partiell ausrichten + Spektrum in Ruhe lassen."
  KILL: Sonst — die G7-Beobachtung war Rauschen; der
     Feld-Standard ist kapazitaets-neutral.
  UNENTSCHEIDEN: >30% ungueltige Punkte auf einer Seite.
  SEKUNDAER (nur Bericht, keine Behauptung): die s=0.8-Zellen
    (dort gilt die These laut G7-Daten NICHT) und die
    w-Lokalisierung des Maximums (0.6/0.9).
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
W_GRID = [0.0, 0.6, 0.75, 0.9, 1.0]
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


def frame_F(H, Q, w):
    R0 = fixed_frame(H)
    P0 = Q.T @ R0
    I = torch.eye(H)
    R_y = torch.linalg.qr(w * I + (1.0 - w) * P0)[0]
    return Q @ R_y


def rollout_states(H, W, F, Uw, x_seq):
    B, Tn, _ = x_seq.shape
    z = torch.zeros(B, H)
    states = []
    for t in range(Tn):
        x_t = x_seq[:, t]
        for _ in range(SUBSTEPS):
            def f(zz):
                return -zz + torch.tanh(zz @ F) @ F.T @ W.T + x_t @ Uw.T
            k1 = f(z)
            k2 = f(z + 0.5 * DT * k1)
            z = z + DT * k2
        states.append(z)
    return torch.stack(states, dim=1)


def memory_capacity(states, x_seq, delays):
    H = states.shape[-1]
    z = states[0]
    u = x_seq[0, :, 0]
    total = 0.0
    for k in delays:
        rows = slice(WASHOUT + k, u.shape[0])
        S = z[rows]
        y = u[WASHOUT:u.shape[0] - k]
        A = S.T @ S + RIDGE * torch.eye(H)
        wout = torch.linalg.solve(A, S.T @ y)
        pred = S @ wout
        pc = pred - pred.mean()
        yc = y - y.mean()
        den = torch.sqrt((pc * pc).sum() * (yc * yc).sum())
        if den > 0:
            total += float(((pc * yc).sum() / den) ** 2)
    return total


def mstats(vals):
    ok = [v for v in vals if v == v and abs(v) < 1e6]
    st = float(torch.tensor(ok).std()) if len(ok) > 1 else float('nan')
    mean = float(torch.tensor(ok).mean()) if ok else float('nan')
    return mean, st, len(vals) - len(ok)


def main():
    out_path = 'results/g8_anchor_capacity_results.txt'
    csv_path = 'results/g8_anchor_capacity_table.csv'
    os.makedirs('results', exist_ok=True)
    seeds = list(range(20))
    x_mc = drive_input(1, T_TOTAL, IS)

    L = []
    L.append('=' * 118)
    L.append('G8 — ZUFALLS-ANKER ALS KAPAZITAETS-SUBOPTIMUM (registriert: PASS nur wenn in')
    L.append('BEIDEN s=0-Zellen mean_MC(0.75)>=1.05*mean_MC(0) UND std_MC(0.75)<=std_MC(0);')
    L.append('s=0.8-Zellen nur Bericht; 20 Seeds, Bias genullt, float64)')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            mc_means, mc_stds, mc_bad = {}, {}, {}
            for w in W_GRID:
                mcs = []
                for seed in seeds:
                    W, Q, T = factored_WQT(H, 1.0, s_nn, seed, FIXED, FIXED)
                    Uw = Q[:, 0:1].contiguous()
                    F = frame_F(H, Q, w)
                    mcs.append(memory_capacity(rollout_states(H, W, F, Uw, x_mc),
                                               x_mc, DELAYS))
                mean, st, bad = mstats(mcs)
                mc_means[w], mc_stds[w], mc_bad[w] = mean, st, bad
                L.append('W(w=%.2f): MC mean=%.4f std=%.4f ungueltig=%d'
                         % (w, mean, st, bad))
                rows.append(dict(H=H, s=s_nn, w=w, mean=mean, std=st,
                                 n=len(seeds), bad=bad))
            cell_stats[(H, s_nn)] = dict(mc_means=mc_means, mc_stds=mc_stds,
                                         mc_bad=mc_bad)
            Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('H,s,w,mean_MC,std_MC,n,bad\n')
        for r in rows:
            fh.write('%d,%s,%.2f,%.6g,%.6g,%d,%d\n'
                     % (r['H'], r['s'], r['w'], r['mean'], r['std'], r['n'], r['bad']))

    L.append('')
    L.append('--- G8-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    pass_all = True
    for (H, s_nn), c in sorted(cell_stats.items()):
        m, sd, bad = c['mc_means'], c['mc_stds'], c['mc_bad']
        valid = all(bad[w] <= 0.3 * len(seeds) for w in W_GRID)
        if s_nn == 0.0:
            cond = (valid and m[0.75] == m[0.75] and m[0.0] == m[0.0]
                    and m[0.75] >= 1.05 * m[0.0]
                    and sd[0.75] == sd[0.75] and sd[0.0] == sd[0.0]
                    and sd[0.75] <= sd[0.0])
            L.append('PRIMAER H=%d s=0.0: mean(0.75)=%.4f vs 1.05*mean(0)=%.4f, '
                     'std(0.75)=%.4f <= std(0)=%.4f -> %s'
                     % (H, m[0.75], 1.05 * m[0.0], sd[0.75], sd[0.0],
                        'ERFUELLT' if cond else 'NICHT ERFUELLT'))
            pass_all = pass_all and cond
        else:
            L.append('SEKUNDAER H=%d s=0.8 (nur Bericht): mean(0)=%.4f mean(0.75)=%.4f '
                     'std(0)=%.4f std(0.75)=%.4f'
                     % (H, m[0.0], m[0.75], sd[0.0], sd[0.75]))
        wmax = max(W_GRID, key=lambda w: (m[w] if m[w] == m[w] else -1))
        L.append('  Peak der Kapazitaetskurve bei w=%.2f (mean=%.4f)'
                 % (wmax, m[wmax]))

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if gate_bad:
        verdict = 'UNENTSCHEIDEN — zu viele ungueltige Punkte.'
    elif pass_all:
        verdict = ('PASS — Der Zufalls-Anker ist ein Kapazitaets-Suboptimum: '
                   'Partielle Ausrichtung des Nichtlinearitaets-Rahmens an die '
                   'Reservoir-Basis liefert in Normal-Zellen mehr Memory '
                   'Capacity als der Feld-Standard, ohne Streuungsanstieg. '
                   'Richtig bauen (Stand G1-G8): (1) Bias nullen, (2) Anker '
                   'partiell ausrichten statt zufaellig, (3) Spektrum in Ruhe '
                   'lassen. Der Anker ist kein Diagnose-Objekt, sondern ein '
                   'Kapazitaets-Knopf.')
    else:
        verdict = ('KILL — Die G7-Beobachtung haelt 20 Seeds nicht stand: Der '
                   'Zufalls-Anker ist kapazitaets-neutral, keine Fehl-'
                   'Konstruktion. Lehre 4 faellt; Lehren 1-3 bleiben.')
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + out_path + ' und ' + csv_path + ' geschrieben.')


if __name__ == '__main__':
    main()
