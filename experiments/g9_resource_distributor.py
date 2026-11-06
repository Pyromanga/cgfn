#!/usr/bin/env python
'''G9 — DER ANKER ALS RESSOURCEN-VERTEILER: GLEICHVERTEILTE
FEHLWINKEL VS. ZUFALL VS. KONZENTRATION. (registriert 2026-11-06)

VORGESCHICHTE:
  G8 PASS (20 Seeds): Partielle Ausrichtung (w~0.6) schlaegt den
  Zufalls-Anker um +16% bis +41% Memory Capacity bei halbierter
  bis gedrittelter Seed-Streuung. XVII eingefroren.

G9-THESE (Mechanismus, nicht nur Parameter): Der Anker-Fehlwinkel
  VERTEILT die Nichtlinearitaets-Ressource (Spraegung/Saettigung)
  ueber die Spektral-Moden. Der Zufalls-Anker streut sie ungleich-
  maessig (manche Moden saettigen, andere laufen fast linear);
  vollstaendige Ausrichtung konzentriert sie auf EINEN Modus
  (kapazitaets-tot, G6); gleichmaessige Fehlwinkel verteilen sie
  ausgewogen (G8-Gewinn). Wenn das stimmt, ist der Anker kein
  Winkel-Parameter, sondern ein RESSOURCEN-VERTEILER — und die
  Verteilung ist designbar.

KONSTRUKTION (per-Modus-Winkel, Rahmen in Basis-Koordinaten):
  R_y = QR( D_c + D_s * P0 ),  P0 = Q^T R0 (fix, Seed 4242),
  D_c = diag(cos theta_i), D_s = diag(sin theta_i).
  theta_i = Fehlwinkel von Modus i gegen den Rahmen.
  UNI:  theta_i = theta* fuer alle i (gleichverteilt). theta*=0
        -> voll ausgerichtet (NC, tot); theta*=pi/2 -> Zufall.
  CONC: theta_0 = theta*, theta_i = 0 fuer i>0 (bewusste
        Konzentration auf EINEN Modus — Anti-Design).
  RND:  R_y = qr(P0) (Feld-Standard, Referenz).
  theta* pro (Zelle, Arm) EINMAL auf Seed 0 kalibriert aus
  Raster {0.3, 0.6, 0.9, 1.2, 1.5}.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Zellen: H in {32,128} x s in {0.0, 0.8}. Seeds 0..19.
  Bias genullt (Lehre L1), float64, lam=1.0, Spektrum fix,
  Q rotiert pro Seed, Antrieb U=Q[:,0]. Nur Zustands-Ebene
  (kein Tangent — G7-Bug ist dokumentiert und hier unmoeglich).
  PRIMAER, PASS genau dann wenn in >= 3/4 Zellen:
    (i)   mean_MC_UNI >= 1.05 * mean_MC_RND
    UND (ii)  mean_MC_UNI >= 1.05 * mean_MC_CONC
  PASS -> "Gleichverteilte Fehlwinkel sind der Mechanismus
     hinter dem G8-Gewinn: Der Anker ist ein Ressourcen-
     Verteiler, die Verteilung ist der Kapazitaets-Knopf."
  KILL: Sonst — nur der mittlere Winkel zaehlt (dann ist UNI
     == CONC == RND bei kalibriertem theta, und die Verteiler-
     erzaehlung faellt).
  UNENTSCHEIDEN: >30% ungueltige Punkte auf einer Seite.
  SEKUNDAER (nur Bericht): theta*-Kurven; CONC vs RND-Rangfolge;
    std_MC je Arm (Erwartung: UNI am stabilsten).
  Task: Memory Capacity wie G6/G8 (k in {1,2,4,8,16}, T=300,
  Washout=50, Ridge 1e-6).
'''
import math
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
THETA_GRID = [0.3, 0.6, 0.9, 1.2, 1.5]
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


def fixed_P0(H, Q):
    genR = torch.Generator().manual_seed(FRAME_SEED)
    A = torch.randn(H, H, generator=genR)
    R0, _ = torch.linalg.qr(A)
    return Q.T @ R0


def frame_F(H, Q, arm, theta):
    '''R_y per Arm: UNI gleichmaessig, CONC nur Modus 0, RND fix.'''
    P0 = fixed_P0(H, Q)
    if arm == 'RND':
        R_y = torch.linalg.qr(P0)[0]
    elif arm == 'UNI':
        thetas = torch.full((H,), theta)
        M = torch.diag(torch.cos(thetas)) + torch.diag(torch.sin(thetas)) @ P0
        R_y = torch.linalg.qr(M)[0]
    elif arm == 'CONC':
        thetas = torch.zeros(H)
        thetas[0] = theta
        M = torch.diag(torch.cos(thetas)) + torch.diag(torch.sin(thetas)) @ P0
        R_y = torch.linalg.qr(M)[0]
    else:
        raise ValueError(arm)
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
    out_path = 'results/g9_resource_distributor_results.txt'
    csv_path = 'results/g9_resource_distributor_table.csv'
    os.makedirs('results', exist_ok=True)
    seeds = list(range(20))
    x_mc = drive_input(1, T_TOTAL, IS)

    L = []
    L.append('=' * 118)
    L.append('G9 — ANKER ALS RESSOURCEN-VERTEILER (registriert: PASS nur wenn in >= 3/4 Zellen')
    L.append('mean_UNI >= 1.05*mean_RND UND mean_UNI >= 1.05*mean_CONC; 20 Seeds, Bias null, f64)')
    L.append('Arme: UNI=gleichverteilter Fehlwinkel, CONC=konzentriert (nur Modus 0), RND=Feld-Standard')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            block = {}
            for arm in ('RND', 'UNI', 'CONC'):
                if arm == 'RND':
                    best_th = 0.0
                else:
                    best_th, best_mc, best_gap = None, None, -1e9
                    for th in THETA_GRID:
                        W, Q, T = factored_WQT(H, 1.0, s_nn, 0, FIXED, FIXED)
                        Uw = Q[:, 0:1].contiguous()
                        F = frame_F(H, Q, arm, th)
                        mc = memory_capacity(rollout_states(H, W, F, Uw, x_mc),
                                             x_mc, DELAYS)
                        if mc == mc and mc > best_gap:
                            best_th, best_gap = th, mc
                    L.append('Kalibrierung %s: theta*=%.2f MC(seed0)=%.4f'
                             % (arm, best_th, best_gap))
                mcs = []
                for seed in seeds:
                    W, Q, T = factored_WQT(H, 1.0, s_nn, seed, FIXED, FIXED)
                    Uw = Q[:, 0:1].contiguous()
                    F = frame_F(H, Q, arm, best_th)
                    mcs.append(memory_capacity(rollout_states(H, W, F, Uw, x_mc),
                                               x_mc, DELAYS))
                mean, st, bad = mstats(mcs)
                block[arm] = dict(mean=mean, std=st, bad=bad, theta=best_th)
                L.append('Arm %s (theta*=%.2f): MC mean=%.4f std=%.4f ungueltig=%d'
                         % (arm, best_th, mean, st, bad))
                rows.append(dict(H=H, s=s_nn, arm=arm, theta=best_th, mean=mean,
                                 std=st, n=len(seeds), bad=bad))
            cell_stats[(H, s_nn)] = block
            Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,theta,mean_MC,std_MC,n,bad\n')
        for r in rows:
            fh.write('%d,%s,%s,%.2f,%.6g,%.6g,%d,%d\n'
                     % (r['H'], r['s'], r['arm'], r['theta'], r['mean'],
                        r['std'], r['n'], r['bad']))

    L.append('')
    L.append('--- G9-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    cells_ok = 0
    for (H, s_nn), b in sorted(cell_stats.items()):
        mU, mR, mC = b['UNI']['mean'], b['RND']['mean'], b['CONC']['mean']
        ok = (mU == mU and mR == mR and mC == mC
              and mU >= 1.05 * mR and mU >= 1.05 * mC)
        cells_ok += 1 if ok else 0
        L.append('Zelle H=%d s=%s: mean_UNI=%.4f mean_RND=%.4f mean_CONC=%.4f '
                 '(UNI>=1.05*RND: %s, UNI>=1.05*CONC: %s) -> %s'
                 % (H, s_nn, mU, mR, mC,
                    'JA' if mU >= 1.05 * mR else 'NEIN',
                    'JA' if mU >= 1.05 * mC else 'NEIN',
                    'ERFUELLT' if ok else 'NICHT ERFUELLT'))
        L.append('  SEKUNDAER: std_UNI=%.4f std_RND=%.4f std_CONC=%.4f | '
                 'theta*_UNI=%.2f theta*_CONC=%.2f | CONC vs RND: %s'
                 % (b['UNI']['std'], b['RND']['std'], b['CONC']['std'],
                    b['UNI']['theta'], b['CONC']['theta'],
                    'CONC<RND' if b['CONC']['mean'] < b['RND']['mean'] else 'CONC>=RND'))

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if gate_bad:
        verdict = 'UNENTSCHEIDEN — zu viele ungueltige Punkte.'
    elif cells_ok >= 3:
        verdict = ('PASS — Gleichverteilte Fehlwinkel sind der Mechanismus: UNI '
                   'schlaegt sowohl Zufall als auch bewusste Konzentration. Der '
                   'Anker ist ein RESSOURCEN-VERTEILER: Die Verteilung der '
                   'Nichtlinearitaets-Ressource ueber die Spektral-Moden ist '
                   'der Kapazitaets-Knopf — designbar, nicht nur tunbar. '
                   'G10: Verteilungs-Design gegen Kapazitaets-Optimum (freie '
                   'Form), dann Transfer-Frage in echte Nichtlinearitaets-'
                   'Frames.')
    else:
        verdict = ('KILL — Nur der mittlere Winkel zaehlt: UNI, CONC, RND '
                   'unterscheiden sich bei kalibriertem theta* nicht sauber. '
                   'Die Verteiler-Erzaehlung faellt; der G8-Gewinn bleibt ein '
                   'Winkel-Parameter, kein Mechanismus.')
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + out_path + ' und ' + csv_path + ' geschrieben.')


if __name__ == '__main__':
    main()
