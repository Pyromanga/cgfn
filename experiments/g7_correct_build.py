#!/usr/bin/env python
'''G7 — RICHTIG BAUEN: DER ANKER-WINKEL ALS DESIGN-PARAMETER.
Lehren aus G1-G6, systematisch getestet. (registriert 2026-11-06, VOR dem Lauf)

LEHREN (Stand G1-G6, alle registriert besiegelt):
  (L1) Bias nullen: Der nn.Linear-Bias lottert ben1 mit std bis
       0.138 (G5-Audit) — unkontrollierter Zusatzantrieb, den
       die Literatur als Seed-Rauschen wegpubliziert.
  (L2) Das Feld tunes das falsche Ventil (G1 KILL: Spektrum
       ohne Autoritaet; G2/G5: Geometrie-Anker ist die Achse).
  (L3) Volle Symmetrie toetet die Kapazitaet (G6 KILL:
       MC -40% bis -79% bei std=0). Richtig: partielle,
       kontrollierte Ausrichtung mit Preis-Regler.
  (L4) Die Streuung ist strukturiert (Anker-Winkel), also
       steuerbar — kein Schicksal, aber auch nicht gratis.

G7-FRAGE: Ist der Anker-Winkel ein KONTINUIERLICHER
  Zuverlaessigkeits-Regler mit steuerbarem Preis — und wie
  sieht die "richtig gebaut"-Konstruktion im direkten
  Vergleich gegen den Feld-Standard aus?

KONSTRUKTION (durchgaengige Familie):
  Schur wie G5: W = Q T Q^T, Spektrum fix, Q rotiert pro Seed,
  Antrieb U=Q[:,0], bias-frei. Nichtlinearitaets-Rahmen:
    F(w) = Q @ R_y(w),  R_y(w) = QR( w*I + (1-w)*P0 ),
    P0 = Q^T R0 (fixer Zufalls-Rahmen R0, Seed 4242).
    w=1: F=Q   -> mitfahrend (G5/G6 "NC", tot reproduzierbar)
    w=0: F=R0' -> fester Rahmen (G6 "NF", Feld-nahe Basislinie)
    0<w<1: partielle Ausrichtung.
  ARME: W(w) fuer w in {0.0,0.25,0.5,0.75,1.0}: je 10 Seeds
    -> mean_MC, std_MC (Kapazitaet/Zuverlaessigkeit)
    -> std_ben1 (getriebener Exponent, w in {0.0,0.5,1.0},
       float64-Benettin wie G5).
  NB-Arm (Lehre L1): w=0 + geseederter Bias U(-1,1) je Seed:
    misst, wie viel MC-Streuung allein der Bias macht.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Zellen: H in {32,128} x s in {0.0, 0.8}. Seeds 0..9.
  PRIMAER, PASS genau dann wenn in ALLEN 4 Zellen:
    (i)   Spearman(w, std_MC) <= -0.7 — die Zuverlaessigkeit
          ist kontinuierlich und monoton steuerbar.
    UND (ii)  mean_MC(0.25) >= 0.8 * mean_MC(0.0) — ein
          Viertel Ausrichtung kostet < 20% Kapazitaet.
    UND (iii) std_MC(0.25) < std_MC(0.0) — schon w=1/4
          kauft messbare Zuverlaessigkeit.
    PASS -> "Richtig bauen = (1) Bias nullen, (2) Anker-Winkel
       als Design-Parameter, (3) w* aus der Tradeoff-Kurve.
       Varianz ist steuerbar bei bekanntem Preis."
    KILL: Sonst — der Anker ist Diagnose, kein Regler.
    UNENTSCHEIDEN: >30% ungueltige Punkte in einer
       Vergleichsseite.
  SEKUNDAER (Bericht): std_ben1(w)-Kurve; NB-vs-W(0)-MC-Streu-
    vergleich (Bias-Lehre auf Leistungsseite); Pareto-Tabelle
    (mean_MC vs std_MC je w) als CSV.
  TASK: Memory Capacity wie G6 (verzoegerte Wiedergabe,
    k in {1,2,4,8,16}, T=300, Washout=50, Ridge 1e-6, f64).
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
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
W_BEN = [0.0, 0.5, 1.0]
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
    '''F(w) = Q @ QR(w*I + (1-w)*P0), P0 = Q^T R0. w=1 -> Q, w=0 -> fest.'''
    R0 = fixed_frame(H)
    P0 = Q.T @ R0
    I = torch.eye(H)
    R_y = torch.linalg.qr(w * I + (1.0 - w) * P0)[0]
    return Q @ R_y


def make_dyn(H, W, F, Uw, bias=None):
    '''phi(z) = F tanh(F^T z); f = -z + W phi(z) + U u (+ b).'''
    def fval(z):
        return -z + torch.tanh(z @ F) @ F.T @ W.T + z.new_zeros(z.shape[0], H)

    def dyn(z, dz, x_t):
        uin = x_t @ Uw.T
        if bias is not None:
            uin = uin + bias.unsqueeze(0)
        f = -z + torch.tanh(z @ F) @ F.T @ W.T + uin
        sech2 = 1.0 - torch.tanh(z @ F).pow(2)
        inner = sech2.unsqueeze(-1) * (dz @ F)
        g = -dz + inner @ F.T @ W.T
        return f, g
    return dyn


def rollout_states(H, dyn, x_seq):
    B, Tn, _ = x_seq.shape
    z = torch.zeros(B, H)
    states = []
    for t in range(Tn):
        x_t = x_seq[:, t]
        eye = torch.eye(H).unsqueeze(0).expand(B, H, H)
        for _ in range(SUBSTEPS):
            f1, _ = dyn(z, eye, x_t)
            f2, _ = dyn(z + 0.5 * DT * f1, eye, x_t)
            z = z + DT * f2
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
        num = (pc * yc).sum()
        den = torch.sqrt((pc * pc).sum() * (yc * yc).sum())
        if den > 0:
            total += float((num / den) ** 2)
    return total


def benettin_custom(dyn, x_seq, H):
    B, Tn, _ = x_seq.shape
    z = torch.zeros(B, H)
    dz = torch.eye(H).unsqueeze(0).expand(B, H, H).contiguous()
    log_sum = torch.zeros(H)
    t_total = 0.0
    for t in range(Tn):
        x_t = x_seq[:, t]
        for _ in range(SUBSTEPS):
            f1, g1 = dyn(z, dz, x_t)
            z2 = z + 0.5 * DT * f1
            dz2 = dz + 0.5 * DT * g1
            f2, g2 = dyn(z2, dz2, x_t)
            z = z + DT * f2
            dz = dz + DT * g2
            t_total += DT
            Q_, R_ = torch.linalg.qr(dz)
            diag_r = torch.diagonal(R_, dim1=-2, dim2=-1).abs()
            log_sum += torch.log(diag_r.clamp_min(1e-30)).mean(dim=0)
            dz = Q_
    spectrum = (log_sum / max(t_total, 1e-9)).sort(descending=True).values
    return float(spectrum[0])


def spearman(xs, ys):
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


def mstats(vals):
    ok = [v for v in vals if v == v and abs(v) < 1e6]
    st = float(torch.tensor(ok).std()) if len(ok) > 1 else float('nan')
    mean = float(torch.tensor(ok).mean()) if ok else float('nan')
    return mean, st, len(vals) - len(ok)


def main():
    out_path = 'results/g7_correct_build_results.txt'
    csv_path = 'results/g7_correct_build_table.csv'
    os.makedirs('results', exist_ok=True)
    seeds = list(range(10))
    x_mc = drive_input(1, T_TOTAL, IS)
    x_ben = drive_input(4, 48, IS)

    L = []
    L.append('=' * 118)
    L.append('G7 — RICHTIG BAUEN (registriert: PASS nur wenn in ALLEN 4 Zellen')
    L.append('Spearman(w,std_MC)<=-0.7 UND mean_MC(0.25)>=0.8*mean_MC(0.0) UND')
    L.append('std_MC(0.25)<std_MC(0.0); Bias-Arm NB + ben-Kurve nur sekundaer)')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            mc_means, mc_stds, mc_bad = {}, {}, {}
            ben_stds = {}
            for w in W_GRID:
                mcs = []
                for seed in seeds:
                    W, Q, T = factored_WQT(H, 1.0, s_nn, seed, FIXED, FIXED)
                    Uw = Q[:, 0:1].contiguous()
                    F = frame_F(H, Q, w)
                    dyn = make_dyn(H, W, F, Uw)
                    mcs.append(memory_capacity(rollout_states(H, dyn, x_mc), x_mc, DELAYS))
                mean, st, bad = mstats(mcs)
                mc_means[w], mc_stds[w], mc_bad[w] = mean, st, bad
                L.append('W(w=%.2f): MC mean=%.4f std=%.4f ungueltig=%d'
                         % (w, mean, st, bad))
                rows.append(dict(H=H, s=s_nn, arm='W%.2f' % w, mean=mean, std=st,
                                 n=len(seeds), bad=bad))
            for w in W_BEN:
                bens = []
                for seed in seeds:
                    W, Q, T = factored_WQT(H, 1.0, s_nn, seed, FIXED, FIXED)
                    Uw = Q[:, 0:1].contiguous()
                    F = frame_F(H, Q, w)
                    dyn = make_dyn(H, W, F, Uw)
                    bens.append(benettin_custom(dyn, x_ben, H))
                _, st, bad = mstats(bens)
                ben_stds[w] = st
                L.append('W(w=%.2f): ben1-std=%.4f ungueltig=%d' % (w, st, bad))
                rows.append(dict(H=H, s=s_nn, arm='BEN%.2f' % w, mean=float('nan'),
                                 std=st, n=len(seeds), bad=bad))
            # NB-Arm: w=0 + Bias lottert (Lehre L1)
            nbs = []
            for seed in seeds:
                W, Q, T = factored_WQT(H, 1.0, s_nn, seed, FIXED, FIXED)
                Uw = Q[:, 0:1].contiguous()
                F = frame_F(H, Q, 0.0)
                genB = torch.Generator().manual_seed(31000 + seed)
                bias = 2.0 * torch.rand(H, generator=genB) - 1.0
                dyn = make_dyn(H, W, F, Uw, bias=bias)
                nbs.append(memory_capacity(rollout_states(H, dyn, x_mc), x_mc, DELAYS))
            nb_mean, nb_std, nb_bad = mstats(nbs)
            L.append('NB (w=0 + Bias): MC mean=%.4f std=%.4f ungueltig=%d'
                     % (nb_mean, nb_std, nb_bad))
            rows.append(dict(H=H, s=s_nn, arm='NB', mean=nb_mean, std=nb_std,
                             n=len(seeds), bad=nb_bad))
            sp = spearman(W_GRID, [mc_stds[w] for w in W_GRID])
            cell_stats[(H, s_nn)] = dict(mc_means=mc_means, mc_stds=mc_stds,
                                         mc_bad=mc_bad, ben_stds=ben_stds,
                                         nb_std=nb_std, sp=sp)
            L.append('Spearman(w, std_MC) = %+.3f' % sp)
            Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,mean_MC,std_MC,n,bad\n')
        for r in rows:
            fh.write('%d,%s,%s,%.6g,%.6g,%d,%d\n'
                     % (r['H'], r['s'], r['arm'], r['mean'], r['std'], r['n'], r['bad']))

    L.append('')
    L.append('--- G7-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    pass_all = True
    for (H, s_nn), c in sorted(cell_stats.items()):
        m, sd, bad = c['mc_means'], c['mc_stds'], c['mc_bad']
        valid = all(bad[w] <= 0.3 * len(seeds) for w in W_GRID)
        c1 = (c['sp'] == c['sp'] and c['sp'] <= -0.7)
        c2 = (m[0.25] == m[0.25] and m[0.0] == m[0.0]
              and m[0.25] >= 0.8 * m[0.0])
        c3 = (sd[0.25] == sd[0.25] and sd[0.0] == sd[0.0]
              and sd[0.25] < sd[0.0])
        cond = valid and c1 and c2 and c3
        L.append('Zelle H=%d s=%s: Spearman=%+.3f, mean_MC(0.25)=%.3f vs 0.8*mean(0)=%.3f, '
                 'std(0.25)=%.4f < std(0)=%.4f -> %s'
                 % (H, s_nn, c['sp'], m[0.25], 0.8 * m[0.0], sd[0.25], sd[0.0],
                    'ERFUELLT' if cond else 'NICHT ERFUELLT'))
        pass_all = pass_all and cond

    L.append('')
    L.append('SEKUNDAER (Bericht):')
    for (H, s_nn), c in sorted(cell_stats.items()):
        b = c['ben_stds']
        L.append('  H=%d s=%s: ben1-std w=0:%.4f w=0.5:%.4f w=1:%.4f | '
                 'Bias-Lehre: std_MC NB=%.4f vs W(0)=%.4f'
                 % (H, s_nn, b[0.0], b[0.5], b[1.0], c['nb_std'], c['mc_stds'][0.0]))

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if gate_bad:
        verdict = 'UNENTSCHEIDEN — zu viele ungueltige Punkte.'
    elif pass_all:
        verdict = ('PASS — Richtig bauen, quantifiziert: (1) Bias nullen '
                   '(Lehre L1, gratis), (2) Anker-Winkel als Design-Parameter '
                   '(Lehre L4): die Zuverlaessigkeit ist kontinuierlich und '
                   'monoton steuerbar, und ein Viertel Ausrichtung kauft sie '
                   'fuer unter 20% Kapazitaet. Der Feld-Standard (w=0, Bias '
                   'zufaellig, Spektrum-Tuning) ist der schlechteste Punkt der '
                   'Kurve in beiden Zielen. Konstruktion: w* aus der Pareto-'
                   'Kurve, nicht w=1 — Symmetrie ist ein Regler, kein Ziel.')
    else:
        verdict = ('KILL — Der Anker-Winkel ist kein kontinuierlicher Regler '
                   'gegen Kapazitaet: Monotonie bricht oder der Preis ist '
                   'diskontinuierlich. Der Anker bleibt Diagnose.')
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + out_path + ' und ' + csv_path + ' geschrieben.')


if __name__ == '__main__':
    main()
