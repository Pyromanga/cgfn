#!/usr/bin/env python
'''T21 — DELLE-STATISTIK (registriert 2026-11-10 VOR dem Lauf, XVII
Nachtrag 33). Frage: Ist die s50-Delle bei H=48 (1.83 gegen 2.00 bei
H=24, 8.5 Prozent) REAL oder Seed-Rauschen? Frischer Seed-Block
210..309 (100, disjunkt). H in {24,48} x s in {1.55,1.83,2.2}.
PRIMAER: d183 = P_lernt(48,1.83) - P_lernt(24,1.83); Schwelle
±0.15 (2 Sigma). SEKUNDAER: s50rep je H (3-Punkt-Kurve, -1 =
ZENSIERT, kein Crash). Crash nur bei Sanity, ndiv, colnorm.'''
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import torch

torch.set_default_dtype(torch.float64)

import lab_kernel
from contracts.t21 import contract_t21

C = contract_t21()
P = C['params']
FIXED_P0 = P['fixed_p0']
TEST_SEED = P['test_seed']
T_SEQ = P['t_seq']
TAU = P['tau']
DT = P['dt']
SUBSTEPS = P['substeps']
STEPS = P['steps']
BATCH = P['batch']
LR = P['lr']
LAM_MAX = P['lam_max']
SEEDS = P['seeds']
A_LO, A_HI = P['a_lo'], P['a_hi']
B_LO, B_HI = P['b_lo'], P['b_hi']
HOLD = P['hold']


def make_task(batch, gen):
    x = torch.zeros(batch, T_SEQ, 2)
    y = torch.zeros(batch, dtype=torch.long)
    for i in range(batch):
        pa = int(torch.randint(A_LO, A_HI, (1,), generator=gen).item())
        pb = int(torch.randint(B_LO, B_HI, (1,), generator=gen).item())
        a = float(2.0 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
        b = float(2.0 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
        x[i, pa:pa + HOLD, 0] = a
        x[i, pb:pb + HOLD, 1] = b
        y[i] = 0 if (a * b > 0) else 1
    return x, y


def eigenbasis(H, seed, wb):
    gen = torch.Generator().manual_seed(seed)
    genP = torch.Generator().manual_seed(FIXED_P0)
    P0, _ = torch.linalg.qr(torch.randn(H, H, generator=genP))
    th = wb * torch.rand(H, generator=gen)
    M = torch.diag(torch.cos(th)) + torch.diag(torch.sin(th)) @ P0
    R, _ = torch.linalg.qr(M)
    return R


def init_W(H, seed, wb):
    genS = torch.Generator().manual_seed(9999 + H)
    eigs = torch.cat([torch.tensor([LAM_MAX]),
                      0.9 * LAM_MAX * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)])
    R = eigenbasis(H, seed, wb)
    W = R @ torch.diag(eigs) @ R.T
    return W


def make_Uw(arm, H, gen):
    s = float(arm['s'])
    v0 = torch.randn(H, generator=gen)
    v1 = torch.randn(H, generator=gen)
    v0 = v0 / v0.norm() * s
    v1 = v1 / v1.norm() * s
    return torch.stack([v0, v1], dim=1)


def dyn(z, W, b, uin):
    return -z / TAU + torch.tanh(z + b) @ W.T + uin


def rollout(W, b, Uw, x):
    B = x.shape[0]
    H = W.shape[0]
    z = torch.zeros(B, H)
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T)
            z = z + DT * k2
    return z


def train_parity(seed, arm, H):
    gen = torch.Generator().manual_seed(seed)
    W0 = init_W(H, seed, float(arm['wb']))
    Uw = make_Uw(arm, H, gen)
    colnorm = float(Uw.norm(dim=0).mean())
    b = 0.1 * torch.randn(H, generator=gen)
    W = W0.clone().requires_grad_(True)
    head_w = torch.zeros(2, H, requires_grad=True)
    head_b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([W, head_w, head_b], lr=LR)
    train_gen = torch.Generator().manual_seed(1000 + seed)
    for _ in range(STEPS):
        x, y = make_task(BATCH, train_gen)
        z = rollout(W, b, Uw, x)
        logits = z @ head_w.T + head_b
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            return None, colnorm
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(1024, test_gen)
    with torch.no_grad():
        z = rollout(W, b, Uw, xT)
        acc = ((z @ head_w.T + head_b).argmax(dim=1) == yT).double().mean().item()
    return acc, colnorm


def median(v):
    if not v:
        return float('nan')
    return float(torch.tensor(v).median())


def s50_from_curve(grid):
    grid = [(s, 0.0 if p != p else p) for s, p in grid]
    if not grid or grid[0][1] >= 0.5 or max(p for _, p in grid) < 0.5:
        return None
    for (s0, p0), (s1, p1) in zip(grid, grid[1:]):
        if p0 < 0.5 <= p1:
            return s0 + (0.5 - p0) * (s1 - s0) / (p1 - p0)
    return None


def fmt_rep(v):
    return 'ZENSIERT' if v == -1.0 else '%.4f' % v


def main():
    out_path = 'results/t21_delle_statistik_results.txt'
    csv_path = 'results/t21_delle_statistik_table.csv'
    ms_path = 'results/t21_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T21 — DELLE-STATISTIK: REAL ODER SEED-RAUSCHEN?')
    L.append('These P-DELLE-REAL (Nachtrag 33): s50(48) < s50(24) ist systematisch.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t21.py')
    L.append('Seeds 210..309 (100, FRISCH, disjunkt zu 190..209), f64 | Verdikt-Mass d183 (±0.15)')
    L.append('=' * 118)

    errs = lab_kernel.preflight_gate(C)
    if errs:
        L.append('K-GATE VERWEIGERT VOR DEM LAUF — keine Runner-Minuten verbrannt:')
        for e in errs:
            L.append('  ALARM: %s' % e)
        out = '\n'.join(L) + '\n'
        Path(out_path).write_text(out, encoding='utf-8')
        print(out)
        sys.exit(1)
    L.append('K-PRE-FLIGHT: OK — Contract strukturell feuerbar.')
    L.append('')

    rows = []
    measurements = {}
    for arm in C['arms']:
        name, H = arm['name'], arm['H']
        pvals = []
        cns = []
        ndiv = 0
        for seed in SEEDS:
            acc, cn = train_parity(seed, arm, H)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            cns.append(cn)
            rows.append(dict(arm=name, H=H, seed=seed, acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            d183=None, s50rep=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, s=%s): n_ok=%d/%d | P_lernt=%.3f | '
                 'median acc=%.4f | colnorm=%.4f | div=%d'
                 % (name, H, arm['role'], arm['s'], len(ok), len(pvals),
                    pf, median(ok), measurements[name]['colnorm'], ndiv))
        L.append('  Parity-Seeds (erste 10): ' + ', '.join(
            '%d:%s' % (s, '%.4f' % v if v == v else 'DIV')
            for s, v in zip(SEEDS[:10], pvals[:10])))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%d,%.6g\n' % (r['arm'], r['H'], r['seed'], r['acc']))

    p48 = measurements['H48_s1.83']['pf']
    p24 = measurements['H24_s1.83']['pf']
    if p48 == p48 and p24 == p24:
        d183 = p48 - p24
    else:
        d183 = None
        L.append('ALARM: pf nicht endlich — d183 nicht messbar (Crash-Lage).')

    s50rep = {}
    for h in [24, 48]:
        grid = sorted((float(a['s']), measurements[a['name']]['pf'])
                      for a in C['arms'] if a['role'] == 'sgrid' and a['H'] == h)
        s50 = s50_from_curve(grid)
        s50rep[h] = -1.0 if s50 is None else s50
        for a in C['arms']:
            if a['role'] == 'sgrid' and a['H'] == h:
                measurements[a['name']]['s50rep'] = s50rep[h]

    for a in C['arms']:
        measurements[a['name']]['d183'] = d183

    L.append('')
    L.append('--- T21-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    for h in [24, 48]:
        grid = sorted((float(a['s']), measurements[a['name']]['pf'])
                      for a in C['arms'] if a['role'] == 'sgrid' and a['H'] == h)
        L.append('  H=%d: Kurve %s | s50rep=%s'
                 % (h, ' '.join('%.2f:%.3f' % g for g in grid), fmt_rep(s50rep[h])))
    L.append('GEMESSEN: d183 = P_lernt(48,1.83) - P_lernt(24,1.83) = %s '
             '(Schwelle ±0.15, 2 Sigma bei 100 Seeds)'
             % ('NICHT GEMESSEN' if d183 is None else '%.4f' % d183))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '33', '2026-11-10')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, d183=d183, s50rep=s50rep),
        indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
