#!/usr/bin/env python
'''T23 — SCHALTER-DOSIS v1.1 (registriert 2026-11-11 VOR dem Lauf,
XVII Nachtrag 40; L5-Durchlauf 2: erster MASCHINEN-Vorschlag,
P4-Generator v1.1). Repliziert der Lern-Kollaps bei
Konjugationstiefe a=1 (T20, alter Ausreisser-Block) auf frischen
Seed-Bloecken? EINE Zelle (H=48, s=2.2, wb=0.6), Stufen a in
{0, 1, 4, 24}, je Stufe 2 disjunkte 20er-Bloecke (430..589),
Anchor HIGH48 (590..609). PRIMAER: drop1 = P(0)-P(1);
PASS >= 0.30 (>= 3.1 SE) | KILL < 0.30. Crash: Sanity, ndiv > 6
je Block, std(P_Block) < 0.02 je Stufe, colnorm. SEKUNDAER:
Schalter-FORM drop124 = P(1)-P(24) (Plateau vs Dosis).'''
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
from contracts.t23 import contract_t23, STAGES

C = contract_t23()
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
Q_SEED = P['q_seed']
DROP1_TH = P['drop1_threshold']
STD_MIN = P['block_std_min']
A_LO, A_HI = P['a_lo'], P['a_hi']
B_LO, B_HI = P['b_lo'], P['b_hi']
HOLD = P['hold']

# Band (registriert VOR dem Lauf, Nachtrag 40 / P4-Proposal v1.1)
BAND = [('P(0)', 0.31, 0.58), ('P(1)', 0.00, 0.25),
        ('P(4)', 0.00, 0.30), ('P(24)', 0.00, 0.25),
        ('drop1', 0.05, 0.55), ('drop124', -0.10, 0.20)]


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


def init_W(H, seed):
    genS = torch.Generator().manual_seed(9999 + H)
    eigs = torch.cat([torch.tensor([LAM_MAX]),
                      0.9 * LAM_MAX * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)])
    R = eigenbasis(H, seed, WB)
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
    W0 = conjugate(init_W(H, seed), int(arm.get('conj_a', 0)), H)
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


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t23_schalter_dosis_results.txt'
    csv_path = 'results/t23_schalter_dosis_table.csv'
    ms_path = 'results/t23_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T23 — SCHALTER-DOSIS v1.1 (L5-DURCHLAUF 2, erster MASCHINEN-Vorschlag: P4-Generator v1.1)')
    L.append('These (Nachtrag 40 / P4-Proposal v1.1): Der a=1-Kollaps aus T20 (alter')
    L.append('Ausreisser-Block 190..209) repliziert auf frischen Bloecken — Schalter real.')
    L.append('EINE Zelle (H=48, s=2.2, wb=0.6), Stufen a in {0, 1, 4, 24}, Konjugation Q_a W Q_a^T.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t23.py')
    L.append('Je Stufe 2 disjunkte 20er-Bloecke (430..589) + Anchor HIGH48 (590..609), f64')
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
    Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    rows = []
    measurements = {}
    stage_blocks = {}
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
        if arm['role'] == 'block':
            stage_blocks.setdefault(int(arm.get('conj_a', 0)), {})[name] = pf
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            pstage=None, drop1=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, a=%d, s=%s, seeds %d..%d): '
                 'n_ok=%d/%d | P_lernt=%.2f | median acc=%.4f | '
                 'colnorm=%.4f | div=%d'
                 % (name, H, arm['role'], int(arm.get('conj_a', 0)),
                    arm['s'], seeds[0], seeds[-1], len(ok), len(pvals),
                    pf, median(ok), measurements[name]['colnorm'], ndiv))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,conj_a,seed,acc\n')
        for r in rows:
            ca = 0 if r['arm'] == 'HIGH48' else int(r['arm'].split('_')[0][1:])
            fh.write('%s,%d,%d,%d,%.6g\n' % (r['arm'], r['H'], ca, r['seed'], r['acc']))

    # Stufen-Aggregation (Block-Gesetz R1; nur role='block')
    stage_p = {}
    stage_std = {}
    crash_std = False
    for st in STAGES:
        blocks = stage_blocks.get(st, {})
        vals = list(blocks.values())
        stage_p[st] = float(torch.tensor(vals).mean()) if vals else None
        stage_std[st] = float(torch.tensor(vals).std(unbiased=False)) if vals else None
        if vals and stage_std[st] < STD_MIN:
            crash_std = True
    drop1 = stage_p.get(0) - stage_p.get(1) if (stage_p.get(0) is not None and stage_p.get(1) is not None) else None
    drop124 = stage_p.get(1) - stage_p.get(24) if (stage_p.get(1) is not None and stage_p.get(24) is not None) else None

    for a in C['arms']:
        st = int(a.get('conj_a', 0)) if a['role'] == 'block' else None
        measurements[a['name']]['pstage'] = stage_p.get(st) if st is not None else None
        measurements[a['name']]['drop1'] = drop1

    L.append('')
    L.append('--- T23-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('Stufen-Bericht (Block-Mittel je Stufe):')
    for st in STAGES:
        L.append('  a=%d: P_Stufe=%s | std(P_Block)=%s'
                 % (st, fmt(stage_p.get(st)), fmt(stage_std.get(st))))
    if crash_std:
        L.append('ALARM: std(P_Block) < %.2f in mindestens einer Stufe — '
                 'degenerierte Streuung (registrierter Crash).' % STD_MIN)
    L.append('SE(Stufen-Mittel) = 0.068 | SE(drop1) = 0.096 (Nachtrag 40)')
    L.append('drop1 = P(0) - P(1) = %s (Schwelle %.2f = 3.1 SE)'
             % (fmt(drop1), DROP1_TH))
    L.append('drop124 = P(1) - P(24) = %s (Form: < 0.15 = Plateau nach Kipp,'
             ' SEKUNDAER)' % fmt(drop124))
    L.append('GEMESSEN: drop1 Schwelle %.2f (Nachtrag 40 / P4-Proposal v1.1)'
             % DROP1_TH)
    L.append('L5-BAND (registriert VOR dem Lauf, Nachtrag 40):')
    hits = 0
    for bname, lo, hi in BAND:
        val = {'P(0)': stage_p.get(0), 'P(1)': stage_p.get(1),
               'P(4)': stage_p.get(4), 'P(24)': stage_p.get(24),
               'drop1': drop1, 'drop124': drop124}[bname]
        hit = val is not None and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('L5-BAND-TREFFER: %d/%d (Maschinen-Band, P4-Generator v1.1)'
             % (hits, len(BAND)))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '41', '2026-11-11')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, stage_p=stage_p,
             stage_std=stage_std, drop1=drop1, drop124=drop124,
             band_hits=hits), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
