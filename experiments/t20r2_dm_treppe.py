#!/usr/bin/env python
'''T20-R2 — DM-TREPPE (registriert 2026-11-10 VOR dem Lauf, XVII
Nachtrag 31). Glatter DM-Regler statt Konjugation: WB steuert die
Diagonallast der Eigenbasis (WB->0 macht R zur Identitaet, DM->1).
H=48 fix, Spektrum fix. PRIMAER: pfspread220 (max-min von
P_lernt(s=2.2) ueber WB-Stufen); PASS >= 0.4, KILL < 0.4.
SEKUNDAER: s50rep je Stufe, -1.0 = ZENSIERT (kein Crash — Lektion
aus T20). Crash nur bei DM-Bereich < 10 Prozent, Sanity, ndiv,
colnorm. Danach Linie tot.'''
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
from contracts.t20r2 import contract_t20r2

C = contract_t20r2()
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


def dm_of(W):
    return float(W.diag().abs().mean() / W.abs().mean())


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


def fmt_rep(v):
    return 'ZENSIERT' if v == -1.0 else '%.4f' % v


def main():
    out_path = 'results/t20r2_dm_treppe_results.txt'
    csv_path = 'results/t20r2_dm_treppe_table.csv'
    ms_path = 'results/t20r2_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T20-R2 — DM-TREPPE: GLATTER REGLER (WB) STATT KONJUGATION')
    L.append('These P-DM-TREPPE-KAUSAL (Nachtrag 31): P_lernt(2.2) hoert auf WB.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t20r2.py')
    L.append('Seeds 190..209 (20), f64, H=48 | Verdikt-Mass pfspread220 (0.4-Schwelle)')
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
            s50rep=None, pfspread220=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, wb=%s): n_ok=%d/%d | P_lernt=%.2f | median '
                 'acc=%.4f | colnorm=%.4f | DM=%.4f | div=%d'
                 % (name, H, arm['role'], arm['wb'], len(ok), len(pvals),
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

    stages = sorted(set(float(a['wb']) for a in C['arms'] if a['role'] == 'sgrid'))
    stage_s50rep = {}
    stage_dm = {}
    stage_pf220 = {}
    for st in stages:
        grid = sorted((float(a['s']), measurements[a['name']]['pf'])
                      for a in C['arms'] if a['role'] == 'sgrid'
                      and float(a['wb']) == st)
        s50 = s50_from_curve(grid)
        stage_s50rep[st] = -1.0 if s50 is None else s50
        stage_dm[st] = float(torch.tensor(
            [measurements[a['name']]['dm'] for a in C['arms']
             if a['role'] == 'sgrid' and float(a['wb']) == st]).mean())
        stage_pf220[st] = [p for s, p in grid if abs(s - 2.2) < 1e-9][0]

    dm_vals = [stage_dm[st] for st in stages]
    dm_spread = (max(dm_vals) - min(dm_vals)) / (sum(dm_vals) / len(dm_vals))
    if dm_spread < 0.10:
        L.append('ALARM: DM-BEREICH ZU SCHMAL: %.4f < 0.10 — WB-Treppe '
                 'bewegt DM nicht genug (registrierter Crash).' % dm_spread)
        spread = None
    else:
        spread = max(stage_pf220[st] for st in stages) - \
                  min(stage_pf220[st] for st in stages)

    for a in C['arms']:
        st = float(a['wb'])
        measurements[a['name']]['s50rep'] = stage_s50rep[st]
        measurements[a['name']]['pfspread220'] = spread

    L.append('')
    L.append('--- T20-R2-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('Stufen-Bericht:')
    for st in stages:
        L.append('  wb=%s: DM=%.4f | pf(2.2)=%.2f | s50rep=%s'
                 % (st, stage_dm[st], stage_pf220[st], fmt_rep(stage_s50rep[st])))
    L.append('DM-Bereich (rel): %.4f (Guard-Schwelle 0.10)' % dm_spread)
    L.append('GEMESSEN: pfspread220 = %s (Schwelle 0.4)'
             % ('NICHT GEMESSEN' if spread is None else '%.4f' % spread))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '31', '2026-11-10')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, stage_s50rep=stage_s50rep,
             stage_dm=stage_dm, stage_pf220=stage_pf220,
             dm_spread=dm_spread, spread=spread), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s (D-Diff-Snapshot)' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
