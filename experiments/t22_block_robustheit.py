#!/usr/bin/env python
'''T22 — BLOCK-ROBUSTHEIT ALT vs NEU (registriert 2026-11-11 VOR dem
Lauf, XVII Nachtrag 38; L5-Durchlauf 1: erster LLM-Vorschlag).
EINE Zelle (H=48, s=2.2, wb=0.6), 5 neue disjunkte 20er-Bloecke
310..409, Anchor HIGH48 (Seeds 410..429). PRIMAER: zalt = (0.75 -
mean(P_neu5)) / std(P_neu5); PASS >= 2.5 (ALT-AUSREISSER), KILL <
2.5 (KONTINUITAET). Crash: std < 0.02, Sanity, ndiv, colnorm.
SEKUNDAER: Block-P je Block, 10-Block-Verteilung (5 Reanalyse-
Konstanten + 5 neu), zalt-10 als Bericht.'''
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
from contracts.t22 import contract_t22

C = contract_t22()
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
A_LO, A_HI = P['a_lo'], P['a_hi']
B_LO, B_HI = P['b_lo'], P['b_hi']
HOLD = P['hold']
P_ALT = P['p_alt_220']
BLOCKS_REAN = P['blocks_reanalyse']


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


def main():
    out_path = 'results/t22_block_robustheit_results.txt'
    csv_path = 'results/t22_block_robustheit_table.csv'
    ms_path = 'results/t22_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T22 — BLOCK-ROBUSTHEIT ALT vs NEU (L5-DURCHLAUF 1, erster LLM-Vorschlag)')
    L.append('These (Nachtrag 38): Der alte Block 190..209 (P48(2.2)=0.75) war ein')
    L.append('AUSREISSER der Block-Verteilung. Verdikt-Mass zalt (Schwelle 2.5).')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t22.py')
    L.append('5 neue disjunkte 20er-Bloecke 310..409, H=48, s=2.2, f64 | L5-Band gem. Nachtrag 38')
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
        seeds = arm['seeds']
        pvals = []
        cns = []
        ndiv = 0
        for seed in seeds:
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
            zalt=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, s=%s, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | median acc=%.4f | colnorm=%.4f | div=%d'
                 % (name, H, arm['role'], arm['s'], seeds[0], seeds[-1],
                    len(ok), len(pvals), pf, median(ok),
                    measurements[name]['colnorm'], ndiv))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%d,%.6g\n' % (r['arm'], r['H'], r['seed'], r['acc']))

    block_ps = [measurements['B%d' % i]['pf'] for i in range(5)]
    mean5 = float(torch.tensor(block_ps).mean())
    std5 = float(torch.tensor(block_ps).std(unbiased=False))
    if std5 < 0.02:
        L.append('ALARM: std(P_neu5) = %.4f < 0.02 — degenerierte Streuung, '
                 'zalt nicht messbar (registrierter Crash).' % std5)
        zalt = None
    else:
        zalt = (P_ALT - mean5) / std5

    all10 = BLOCKS_REAN + block_ps
    mean10 = float(torch.tensor(all10).mean())
    std10 = float(torch.tensor(all10).std(unbiased=False))
    zalt10 = (P_ALT - mean10) / std10 if std10 >= 0.02 else None

    for a in C['arms']:
        measurements[a['name']]['zalt'] = zalt

    L.append('')
    L.append('--- T22-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('Neue Block-P (310..409): %s' % ' '.join('%.2f' % p for p in block_ps))
    L.append('mean(P_neu5) = %.4f | std(P_neu5) = %.4f' % (mean5, std5))
    L.append('10-Block-Verteilung (5 Reanalyse + 5 neu): mean = %.4f | std = %.4f'
             % (mean10, std10))
    L.append('zalt (gegen 5 neue)   = %s' % ('NICHT GEMESSEN' if zalt is None else '%.4f' % zalt))
    L.append('zalt-10 (Bericht)     = %s' % ('NICHT GEMESSEN' if zalt10 is None else '%.4f' % zalt10))
    L.append('GEMESSEN: zalt = (0.75 - mean5) / std5, Schwelle 2.5 (Nachtrag 38)')
    L.append('L5-BAND (registriert VOR dem Lauf, Nachtrag 38): mean in [0.30,0.56] | std in [0.05,0.13] | zalt in [2.0,6.0]')
    L.append('L5-BAND-TREFFER: mean %s | std %s | zalt %s' % (
        'JA' if 0.30 <= mean5 <= 0.56 else 'NEIN',
        'JA' if 0.05 <= std5 <= 0.13 else 'NEIN',
        'JA' if zalt is not None and 2.0 <= zalt <= 6.0 else 'NEIN'))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '39', '2026-11-11')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, block_ps=block_ps,
             mean5=mean5, std5=std5, zalt=zalt, zalt10=zalt10,
             all10=all10, mean10=mean10, std10=std10), indent=1),
        encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
