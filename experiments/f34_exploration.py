'''
F34 — ZWEIPHASEN-SUCHE, PHASE 1: EXPLORATION (registriert 2026-11-04).

Vorgeschichte: F30.3 PASS -> F30.4 KILL (Konfund: Chaos-Split);
F33 UNENTSCHEIDEN (eigene Design-Fehler) mit explorativem Signal:
MC korreliert unter Stabilitaets-Matching auf beiden Aufgaben-
familien (+0.90/+0.86); top25 (FTLE-Form) tot (-0.10/-0.28).

PHASE-1-PRINZIP (vorab registriert, keine Nachverhandlung):
  Serielle Ein-These-pro-Lauf-Suche hat dreimal versagt (F30.3,
  E3b, FTLE). Phase 1 ist SKRUPELLOSE BREITE: viele Observablen,
  viele Aufgaben, stability-gematchte Gitter, alles geloggt,
  KEINE Schwelle, KEIN Verdikt, KEIN Anspruch.
  Phase 2 (separater Lauf, spaeter): die top-K Observablen werden
  VOR dem Lauf registriert und auf FRISCHEN Seeds/Gitterpunkten
  mit nagelbarer Schwelle bestaetigt. Erst was Phase 2 ueberlebt,
  ist echt. Alles hier ist Explorationsmaterial.

DESIGN:
  - Gitter: H in {32, 128} x s in {0.0,0.4,0.8,1.2,1.6} x seeds
    {0,1,2,3} = 40 Modelle. lam-Bisektion (10 Iter.) auf
    ben-Ziel -0.30 (Konfund kontrolliert, F24/F33-Muster).
  - Jede Konfiguration EINMAL gebaut, ALLE Observablen und ALLE
    Aufgaben darauf gemessen (Bisektionskosten geteilt).
  - Observablen (ca. 25, alle geloggt):
      Spektrum: ben1, top10, top25, mean, std, spread, q75,
                frac_edge(> -0.05), frac_m02(> -0.2)
      Zoo auf W: kappa_T, kappa_cond, delta_H, kreiss_lb
      Feld (J aus d_emp): transient g, t*, Exzess E, mc_opt, h2
      Empirie: Tr(Sigma_emp), sv_max(S), top-eig(S), part_ratio(S)
      Rollout: MC (Jaeger), NLM (nichtlineare Memory)
      Struktur: lam* (Bisektionsprodukt)
  - Aufgaben (5 Familien, geschlossene Ridge-Readouts):
      sinus_memory (delay 3), NARMA-10, white_delay8 (reine
      Memory-Aufgabe), prod_task (u_t * u_{t-3}, nichtlinear),
      trend_task (gleitender Mittelwert, Integration).
  - Zellen: 2 H x 5 Aufgaben = 10 Zellen, n=20 je Zelle.

OUTPUT: results/f34_exploration_table.csv (breite Tabelle) und
results/f34_exploration_results.txt (Korrelations-Ranking pro
Zelle und gemittelt, NUR deskriptiv).
'''
import argparse
import math
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_spectrum
from cgfn.design import make_designed_model
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity
from cgfn.resource import (empirical_sigma, kappa_zoo,
                           mc_linear_prediction, normal_excess,
                           transient_growth)

BEN_TARGET = -0.30
IS = 0.5


def ridge_r2(Ftr, Ytr, Fte, Yte, ridge=1e-2):
    A = Ftr.T @ Ftr + ridge * torch.eye(Ftr.shape[1])
    W = torch.linalg.solve(A, Ftr.T @ Ytr.unsqueeze(1))
    pred = (Fte @ W).squeeze(1)
    denom = ((Yte - Yte.mean()) ** 2).mean().clamp_min(1e-9)
    return (1.0 - ((Yte - pred) ** 2).mean() / denom).item()


@torch.no_grad()
def task_sinus(m, seed):
    def collect(n, sd, seq_len=48, delay=3):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            t = torch.arange(seq_len, dtype=torch.float32)
            freq = 0.25 + 1.75 * torch.rand(1, 1, generator=gen)
            phase = 2 * math.pi * torch.rand(1, 1, generator=gen)
            x = torch.sin(freq * t.unsqueeze(0) + phase)
            y = x.clone()
            y[:, delay:] = x[:, :-delay]
            _, st = m(x.unsqueeze(-1), None)
            feats.append(st[0, delay:]); targs.append(y[0, delay:])
        return torch.cat(feats, 0), torch.cat(targs, 0)
    return ridge_r2(*collect(64, seed + 555), *collect(32, seed + 777))


@torch.no_grad()
def task_white_delay8(m, seed, seq_len=64, delay=8):
    def collect(n, sd):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            u = torch.randn(seq_len, 1, generator=gen) * IS
            _, st = m(u.unsqueeze(0), None)
            feats.append(st[0, delay:])
            targs.append(u[:-delay, 0])
        return torch.cat(feats, 0), torch.cat(targs, 0)
    return ridge_r2(*collect(64, seed + 555), *collect(32, seed + 777))


@torch.no_grad()
def task_prod(m, seed, seq_len=64, lag=3):
    def collect(n, sd):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            u = torch.randn(seq_len, 1, generator=gen) * IS
            _, st = m(u.unsqueeze(0), None)
            feats.append(st[0, 8:])
            u0 = u[:, 0]
            targs.append(u0[8:] * u0[8 - lag:-lag])
        return torch.cat(feats, 0), torch.cat(targs, 0)
    return ridge_r2(*collect(64, seed + 555), *collect(32, seed + 777))


@torch.no_grad()
def task_trend(m, seed, seq_len=64, win=5):
    def collect(n, sd):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            u = torch.randn(seq_len, 1, generator=gen) * IS
            _, st = m(u.unsqueeze(0), None)
            feats.append(st[0, 8:])
            u0 = u[:, 0]
            y = torch.stack([u0[t - win + 1:t + 1].mean()
                             for t in range(8, seq_len)])
            targs.append(y)
        return torch.cat(feats, 0), torch.cat(targs, 0)
    return ridge_r2(*collect(64, seed + 555), *collect(32, seed + 777))


@torch.no_grad()
def task_narma10(m, seed, seq_len=160, washout=40):
    def collect(n, sd):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            u = torch.rand(seq_len, generator=gen) * 0.5
            y = torch.zeros(seq_len)
            yh = torch.zeros(seq_len + 10)
            for t in range(seq_len - 1):
                prod = 1.0
                for i in range(10):
                    prod = prod * yh[10 + t - i]
                un = u[t - 9] if t >= 9 else u[0]
                y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * prod + 1.5 * un * u[t] + 0.1
                yh[10 + t + 1] = y[t + 1]
            _, st = m(u.unsqueeze(0).unsqueeze(-1), None)
            feats.append(st[0, washout:seq_len - 1])
            targs.append(y[washout + 1:])
        return torch.cat(feats, 0), torch.cat(targs, 0)
    return ridge_r2(*collect(50, seed + 555), *collect(25, seed + 777))


@torch.no_grad()
def nlm(m, seed, seq_len=96, n_seq=40):
    gen = torch.Generator().manual_seed(seed + 4242)
    feats, targets = [], []
    for _ in range(n_seq):
        u = torch.randn(seq_len, 1, generator=gen) * IS
        _, st = m(u.unsqueeze(0), None)
        feats.append(st[0])
        u0 = u[:, 0]
        targets.append(torch.stack([u0 * torch.roll(u0, 1, 0),
                                    u0 * torch.roll(u0, 2, 0),
                                    u0 * torch.roll(u0, 4, 0)], dim=1))
    F = torch.cat(feats, 0); T = torch.cat(targets, 0)
    n = F.shape[0] // 2
    A = F[:n].T @ F[:n] + 1e-2 * torch.eye(F.shape[1])
    W = torch.linalg.solve(A, F[:n].T @ T[:n])
    pred = F[n:] @ W
    denom = ((T[n:] - T[n:].mean(0)) ** 2).mean(0).clamp_min(1e-9)
    return (1.0 - ((T[n:] - pred) ** 2).mean(0) / denom).mean().item()


def observables(m, xdr):
    spec, _ = benettin_spectrum(m, xdr)
    sp = spec.sort(descending=True).values
    H = sp.numel()
    S_emp, d_emp = empirical_sigma(m, T=1500, input_scale=IS, burn=200, seed=1234)
    W = m.W.detach().to(torch.float64)
    J = -torch.eye(H, dtype=torch.float64) / float(m.tau) + W * d_emp.to(torch.float64).unsqueeze(0)
    kz = kappa_zoo(W)
    obs = {}
    obs['ben1'] = float(sp[0])
    obs['top10'] = float(sp[:max(1, H // 10)].mean())
    obs['top25'] = float(sp[:max(1, H // 4)].mean())
    obs['mean_spec'] = float(sp.mean())
    obs['std_spec'] = float(sp.std())
    obs['spread'] = float(sp[0] - sp[-1])
    obs['q75'] = float(sp[:max(1, int(0.75 * H))][-1])
    obs['frac_edge'] = float((sp > -0.05).float().mean())
    obs['frac_m02'] = float((sp > -0.2).float().mean())
    obs['kappa_T'] = kz['kappa_T']
    obs['kappa_cond'] = kz['kappa_cond']
    obs['delta_H'] = kz['delta_H']
    obs['kreiss_lb'] = kz['kreiss_lb']
    g, t_star = transient_growth(J)
    obs['g'] = g; obs['t_star'] = t_star
    B = m.U.weight.detach().to(torch.float64)
    Delta = float(m.substeps) * float(m.dt)
    Q = (IS ** 2) * Delta * (B @ B.T)
    try:
        E, _S, _Sn = normal_excess(J, Q)
        obs['E'] = float(E)
    except Exception:
        obs['E'] = float('nan')
    mcl = mc_linear_prediction(J, B, input_scale=IS, Delta=Delta)
    obs['mc_opt'] = float(mcl['mc_opt']) if mcl else float('nan')
    obs['h2'] = float(mcl['h2']) if mcl else float('nan')
    eig_S = torch.linalg.eigvalsh(S_emp)
    obs['tr_S'] = float(eig_S.sum())
    obs['sv_max_S'] = float(eig_S[-1])
    obs['part_ratio'] = float((eig_S.clamp_min(0).sum() ** 2) /
                              (eig_S.clamp_min(0) ** 2).sum().clamp_min(1e-12))
    return obs


def tune_lam(hidden, s, seed, xdr, lo=0.1, hi=2.5, iters=10):
    lam = 0.5 * (lo + hi)
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        torch.manual_seed(seed)
        m = make_designed_model(hidden, 1.0, lam, s, seed, task_dim=0)
        spec, _ = benettin_spectrum(m, xdr)
        ben = float(spec[0])
        if ben < BEN_TARGET:
            lo = lam
        else:
            hi = lam
    torch.manual_seed(seed)
    m = make_designed_model(hidden, 1.0, lam, s, seed, task_dim=0)
    return lam, m


def corr2(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    p = argparse.ArgumentParser(description='F34 phase 1 exploration')
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--s-values', default='0.0,0.4,0.8,1.2,1.6')
    p.add_argument('--out', default='results/f34_exploration_results.txt')
    p.add_argument('--csv', default='results/f34_exploration_table.csv')
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
    L.append('F34 PHASE 1 — EXPLORATION (kein Verdikt, keine Schwelle; Phase 2 bestaetigt separat)')
    L.append(f'Gitter: H in {{32,128}} x s={s_vals} x seeds={seeds}, Bisektion auf ben={BEN_TARGET}')
    L.append('25 Observablen x 5 Aufgaben x 2 H. ALLES hier ist Explorationsmaterial.')
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
                t2s = {}
                for tname, tfn in tasks.items():
                    t2s[tname] = tfn(m, seed)
                rows.append(dict(H=H, s=s_nn, seed=seed, obs=obs, t2=t2s))
                L.append(f"H={H} s={s_nn:.1f} seed={seed} lam*={lam:.3f} ben1={obs['ben1']:+.3f} "
                         f"MC={mc:.2f} E={obs['E']:.2f} "
                         f"sinus={t2s['sinus']:+.3f} narma={t2s['narma']:+.3f} "
                         f"wdelay8={t2s['wdelay8']:+.3f} prod={t2s['prod']:+.3f} "
                         f"trend={t2s['trend']:+.3f}")
        Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    # --- Breite CSV ---
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

    # --- Korrelations-Ranking (rein deskriptiv) ---
    L.append('')
    L.append('--- Korrelations-Ranking pro Zelle (deskriptiv, KEIN Verdikt in Phase 1) ---')
    agg = {k: [] for k in obs_keys}
    for H in (32, 128):
        for tname in tasks:
            sub = [r for r in rows if r['H'] == H]
            tvals = [r['t2'][tname] for r in sub]
            cs = []
            for k in obs_keys:
                ovals = [r['obs'][k] for r in sub]
                c = corr2(ovals, tvals)
                if c == c:
                    cs.append((c, k))
                    agg[k].append(c)
            cs.sort(key=lambda x: -abs(x[0]))
            top = ' | '.join(f"{k}: {c:+.3f}" for c, k in cs[:5])
            L.append(f'Zelle H={H} {tname}: TOP5 {top}')
    L.append('')
    L.append('--- Gemittelt ueber 10 Zellen (|corr|-Ranking, deskriptiv) ---')
    means = sorted(((sum(v) / len(v), k, min(v), max(v)) for k, v in agg.items() if v),
                   key=lambda x: -abs(x[0]))
    for mcorr, k, lo, hi in means:
        L.append(f'{k:>12s}: mean corr {mcorr:+.3f}  (min {lo:+.3f}, max {hi:+.3f}, n_cells={len(agg[k])})')
    L.append('')
    L.append('PHASE 1 FERTIG. Phase 2: top-K Observablen VOR dem Lauf registrieren,')
    L.append('Bestaetigung auf FRISCHEN Seeds/Gittern mit nagelbarer Schwelle.')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} und {args.csv} geschrieben.')


if __name__ == '__main__':
    main()
