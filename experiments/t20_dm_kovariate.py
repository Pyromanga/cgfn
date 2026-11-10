#!/usr/bin/env python
'''T20 — DM-KOVARIATE (registriert 2026-11-09 VOR dem Lauf, XVII
Nachtrag 30). Kausal-Test: DM bei festem H=48 variieren (Basis-
Konjugation, Spektrum identisch), s50 je Stufe messen. Verdikt-Mass
s50spread = max - min der Stufen-s50 (registriert an jedem Arm).
PASS DM-KAUSAL: >= 0.2 | KILL DM-NEBENSPUR: < 0.2 | Crash (DM-
Bereich < 10 Prozent, Stufen-Raster verfehlt, Sanity, ndiv,
colnorm): EINE Reparatur, dann Linie tot. Rollen: REF48 = Bericht,
HIGH48 = Sanity-Anker, Qa_s = Stufen-sgrid.'''
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
from contracts.t20 import contract_t20

C = contract_t20()
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
Q_SEED = P['q_seed']
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
    return W


def conjugate(W, a, H):
    if a == 0:
        return W
    genQ = torch.Generator().manual_seed(Q_SEED)
    G = torch.randn(H, H, generator=genQ)
    M = torch.eye(H) + a * G
    Q, _ = torch.linalg.qr(M)
    return Q @ W @ Q.T


def dm_of(W):
    return float(W.diag().abs().mean() / W.abs().mean())


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
    W0 = conjugate(init_W(H, seed), int(arm.get('conj_a', 0)), H)
    dm = dm_of(W0)
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
    out_path = 'results/t20_dm_kovariate_results.txt'
    csv_path = 'results/t20_dm_kovariate_table.csv'
    ms_path = 'results/t20_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T20 — DM-KOVARIATE: KAUSAL-TEST BEI FESTEM H=48')
    L.append('These P-DM-KAUSAL (Nachtrag 30): s50 haengt an DM, nicht an H.')
    L.append('Konjugation W = Q_a W Q_a^T, a in {0, 1, 4, 12}, Spektrum fix.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t20.py')
    L.append('Seeds 190..209 (20), f64, H=48 | Verdikt-Mass s50spread (0.2-Schwelle)')
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
            dm=float(torch.tensor(dms).mean()), ndiv=ndiv,
            s50=None, s50spread=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, a=%d): n_ok=%d/%d | P_lernt=%.2f | median '
                 'acc=%.4f | colnorm=%.4f | DM=%.4f | div=%d'
                 % (name, H, arm['role'], arm.get('conj_a', 0), len(ok), len(pvals),
                    pf, median(ok), measurements[name]['colnorm'],
                    measurements[name]['dm'], ndiv))
        L.append('  Parity-Seeds (erste 10): ' + ', '.join(
            '%d:%s' % (s, '%.4f' % v if v == v else 'DIV')
            for s, v in zip(SEEDS[:10], pvals[:10])))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%d,%.6g\n' % (r['arm'], r['H'], r['seed'], r['acc']))

    # Stufen-s50: je Konjugationsstufe aus ihren sgrid-Armen
    stages = sorted(set(int(a['conj_a']) for a in C['arms'] if a['role'] == 'sgrid'))
    stage_s50 = {}
    stage_dm = {}
    for st in stages:
        grid = sorted((float(a['s']), measurements[a['name']]['pf'])
                      for a in C['arms'] if a['role'] == 'sgrid'
                      and int(a['conj_a']) == st)
        stage_s50[st] = s50_from_curve(grid)
        stage_dm[st] = float(torch.tensor(
            [measurements[a['name']]['dm'] for a in C['arms']
             if a['role'] == 'sgrid' and int(a['conj_a']) == st]).mean())

    # DM-Bereich-Guard (registriert: < 10 Prozent -> UNENTSCHEIDEN)
    dm_vals = [stage_dm[st] for st in stages]
    dm_spread = (max(dm_vals) - min(dm_vals)) / (sum(dm_vals) / len(dm_vals))
    if dm_spread < 0.10:
        L.append('ALARM: DM-BEREICH ZU SCHMAL: %.4f < 0.10 — '
                 'Konjugation bewegt DM nicht genug (registrierter Crash).' % dm_spread)
        spread = None
    elif any(stage_s50[st] is None for st in stages):
        L.append('ALARM: STUFEN-RASTER VERFEHLT (mindestens eine Stufen-s50 '
                 'nicht interpolierbar — registrierter Crash).')
        spread = None
    else:
        spread = max(stage_s50[st] for st in stages) - \
                 min(stage_s50[st] for st in stages)

    for a in C['arms']:
        st = int(a.get('conj_a', 0))
        key = stage_s50[st] if st in stage_s50 else stage_s50[stages[0]]
        measurements[a['name']]['s50'] = key
        measurements[a['name']]['s50spread'] = spread

    L.append('')
    L.append('--- T20-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('Stufen-Bericht:')
    for st in stages:
        L.append('  a=%d: DM=%.4f | s50=%s' % (st, stage_dm[st], fmt(stage_s50[st])))
    L.append('DM-Bereich (rel): %.4f (Guard-Schwelle 0.10)' % dm_spread)
    L.append('GEMESSEN: s50spread = %s (Schwelle 0.2)' % fmt(spread))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '30', '2026-11-09')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, stage_s50=stage_s50,
             stage_dm=stage_dm, spread=spread), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s (D-Diff-Snapshot)' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
