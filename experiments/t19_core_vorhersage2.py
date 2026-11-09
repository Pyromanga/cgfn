successfully downloaded text file (SHA: e2947c71c89a47bd3ee69f89f6ca12ececd53b7a)#!/usr/bin/env python
'''T19-R2 — REPARATURLAUF: s50(48) MESSEN (registriert 2026-11-09
VOR dem Lauf, XVII Nachtrag 28). Lauf 1 war UNENTSCHEIDEN: Das
Raster {1.9..3.3} lag um die Vorhersage 2.55 (Prediction-Lock-in),
aber P_lernt(1.9)=0.55 — s50(48) sitzt UNTERHALB. R2 erweitert das
Raster nach UNTEN ({0.7..2.2}, 7 Punkte). Schwellen UNVERAENDERT:
PASS [2.05, 3.05), KILL-A < 2.05, KILL-B >= 3.05. Nach einem
weiteren Crash ist die Linie TOT. Contract contracts/t19.py,
Verdikt generiert. Rollen: REF48 = Konventions-Bericht, HIGH48 =
Sanity-Anker, R48_s = s50-Kurve.'''
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
from contracts.t19 import contract_t19

C = contract_t19()
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
WB = P['wb']
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


def eigenbasis(H, seed):
    gen = torch.Generator().manual_seed(seed)
    genP = torch.Generator().manual_seed(FIXED_P0)
    P0, _ = torch.linalg.qr(torch.randn(H, H, generator=genP))
    th = WB * torch.rand(H, generator=gen)
    M = torch.diag(torch.cos(th)) + torch.diag(torch.sin(th)) @ P0
    R, _ = torch.linalg.qr(M)
    return R


def init_W(H, seed):
    genS = torch.Generator().manual_seed(9999 + H)
    eigs = torch.cat([torch.tensor([LAM_MAX]),
                      0.9 * LAM_MAX * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)])
    R = eigenbasis(H, seed)
    W = R @ torch.diag(eigs) @ R.T
    dm = float(W.diag().abs().mean() / W.abs().mean())
    return W, R, dm


def make_Uw(arm, H, gen):
    if arm['kind'] == 'ref':
        return 0.5 * torch.randn(H, 2, generator=gen)
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
    W0, R, dm = init_W(H, seed)
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
            return None, dm, colnorm
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(1024, test_gen)
    with torch.no_grad():
        z = rollout(W, b, Uw, xT)
        acc = ((z @ head_w.T + head_b).argmax(dim=1) == yT).double().mean().item()
    return acc, dm, colnorm


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


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t19_core_vorhersage2_results.txt'
    csv_path = 'results/t19_core_vorhersage2_table.csv'
    ms_path = 'results/t19_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T19-R2 — REPARATURLAUF: RASTER UNTEN, s50(48) MESSEN')
    L.append('Vorhersage: s50(48) ~ 2.55 (+-20 Prozent, Nachtrag 27)')
    L.append('R2-Registrierung: XVII Nachtrag 28 | Lauf 1: UNENTSCHEIDEN (Raster verfehlt)')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t19.py')
    L.append('Seeds 190..209 (20), f64, H=48 | NACH UNENTSCHEIDEN R2: LINIE TOT')
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
        dms = []
        cns = []
        ndiv = 0
        for seed in SEEDS:
            acc, dm, cn = train_parity(seed, arm, H)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            dms.append(dm)
            cns.append(cn)
            rows.append(dict(arm=name, H=H, seed=seed, acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()),
            dm=float(torch.tensor(dms).mean()),
            dmdev=None, ndiv=ndiv, s50=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s): n_ok=%d/%d | P_lernt=%.2f | median '
                 'acc=%.4f | colnorm=%.4f | DM=%.4f | div=%d'
                 % (name, H, arm['role'], len(ok), len(pvals), pf, median(ok),
                    measurements[name]['colnorm'], measurements[name]['dm'], ndiv))
        L.append('  Parity-Seeds (erste 10): ' + ', '.join(
            '%d:%s' % (s, '%.4f' % v if v == v else 'DIV')
            for s, v in zip(SEEDS[:10], pvals[:10])))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%d,%.6g\n' % (r['arm'], r['H'], r['seed'], r['acc']))

    names_h = [a['name'] for a in C['arms']]
    gm = sum(measurements[n]['dm'] for n in names_h) / len(names_h)
    for n in names_h:
        measurements[n]['dmdev'] = abs(measurements[n]['dm'] - gm) / gm

    grid = sorted((float(a['s']), measurements[a['name']]['pf'])
                  for a in C['arms'] if a['role'] == 'sgrid')
    s50 = s50_from_curve(grid)
    for a in C['arms']:
        measurements[a['name']]['s50'] = s50

    L.append('')
    L.append('--- T19-R2-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('VORHERSAGE: s50(48) ~ 2.55 (reiner Potenz-Fit 2.55 | affin 2.58)')
    L.append('R2-BAND (Nachtrag 28, Bericht): s50 in [1.2, 1.9] vermutet')
    L.append('GEMESSEN:   s50(48) = %s' % fmt(s50))
    L.append('sgrid-Kurve: %s' % ', '.join('s=%s:%.2f' % (s, measurements['R48_%s' % s]['pf'])
                                            for s in sorted(float(a['s'])
                                                           for a in C['arms']
                                                           if a['role'] == 'sgrid')))
    if s50 is not None:
        import math
        e_lo = math.log(s50 / 2.0) / math.log(48.0 / 24.0)
        e_hi = math.log(3.5 / s50) / math.log(96.0 / 48.0)
        L.append('Implizierte Schritt-Exponenten (Bericht): 24->48: %.4f | '
                 '48->96: %.4f' % (e_lo, e_hi))
        L.append('MONOTONIE-CHECK (Bericht): s50(48) %s s50(24)=2.00'
                 % ('<' if s50 < 2.0 else '>='))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '28', '2026-11-09')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(
        json.dumps(dict(measurements=measurements, s50=s50), indent=1),
        encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s (D-Diff-Snapshot)' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
