#!/usr/bin/env python
'''T31 — VOLL-AEQUIVARIANTE ROTATION v1 (registriert 2026-11-17
VOR dem Lauf, Nachtrag 55; Antwort auf den Readout-Einwand).
T23-T25 rotieren nur W — b und Uw bleiben in der alten Basis.
FRAGE: Haelt der Schalter unter VOLL aequivanter Rotation
(W, b, Uw alle mit demselben Q, trainierter Readout)?
Arme: rot=part (nur W, exakt T23) und rot=full; Stufen a {0,1},
Bloecke 1270..1349, 80 Runs, f64, ~30 min.
PASS — SCHALTER-IMMUN: P_PART(1) <= 0.15 UND P_FULL(1) <= 0.15.
KILL — ROTATIONS-ARTEFAKT: P_PART(1) <= 0.15 UND P_FULL(1) >= 0.5.
Sanity PART_A0 >= 0.25, FULL_A0 >= 0.25.
R2: jedes Mass an JEDEM Arm.'''
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
from contracts.t31 import contract_t31

C = contract_t31()
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
A_LO, A_HI = P['a_lo'], P['a_hi']
B_LO, B_HI = P['b_lo'], P['b_hi']
HOLD = P['hold']

BAND = [('P_PART(1)', 0.00, 0.22)]


def make_inputs(batch, gen):
    x = torch.zeros(batch, T_SEQ, 2)
    for i in range(batch):
        pa = int(torch.randint(A_LO, A_HI, (1,), generator=gen).item())
        pb = int(torch.randint(B_LO, B_HI, (1,), generator=gen).item())
        a = float(2.0 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
        b = float(2.0 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
        x[i, pa:pa + HOLD, 0] = a
        x[i, pb:pb + HOLD, 1] = b
    return x


def make_task(batch, gen):
    x = make_inputs(batch, gen)
    y = torch.zeros(batch, dtype=torch.long)
    for i in range(batch):
        a = float(x[i, :, 0][x[i, :, 0] != 0][0])
        b = float(x[i, :, 1][x[i, :, 1] != 0][0])
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
    return R @ torch.diag(eigs) @ R.T


def conj_Q(a, H):
    if a == 0:
        return torch.eye(H)
    genQ = torch.Generator().manual_seed(Q_SEED)
    G = torch.randn(H, H, generator=genQ)
    Q, _ = torch.linalg.qr(torch.eye(H) + a * G)
    return Q


def make_Uw(H, gen):
    s = 2.2
    v0 = torch.randn(H, generator=gen)
    v1 = torch.randn(H, generator=gen)
    v0 = v0 / v0.norm() * s
    v1 = v1 / v1.norm() * s
    return torch.stack([v0, v1], dim=1)


def dyn(z, W, b, uin):
    return -z / TAU + torch.tanh(z + b) @ W.T + uin


def rollout(W, b, Uw, x):
    B = x.shape[0]
    z = torch.zeros(B, W.shape[0])
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T)
            z = z + DT * k2
    return z


def train_one(seed, arm, H):
    a = int(arm.get('conj_a', 0))
    rot = arm.get('rot', 'part')
    gen = torch.Generator().manual_seed(seed)
    W0 = init_W(H, seed)
    Q = conj_Q(a, H)
    if a != 0:
        W0 = Q @ W0 @ Q.T
    Uw = make_Uw(H, gen)
    b = 0.1 * torch.randn(H, generator=gen)
    if rot == 'full' and a != 0:
        b = Q @ b
        Uw = Q @ Uw
    colnorm = float(Uw.norm(dim=0).mean())
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


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t31_aequivariant_rotation_results.txt'
    csv_path = 'results/t31_aequivariant_rotation_table.csv'
    ms_path = 'results/t31_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T31 — VOLL-AEQUIVARIANTE ROTATION (Antwort auf den Readout-Einwand: Teil-Rotation vs volle Rotation)')
    L.append('PART: nur W gedreht (exakt T23) | FULL: W, b, Uw alle mit demselben Q, trainierter Readout.')
    L.append('Stufen a {0,1}, Bloecke 1270..1349, 80 Runs, f64.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t31.py')
    L.append('=' * 118)

    errs = lab_kernel.preflight_gate(C)
    if errs:
        L.append('K-GATE VERWEIGERT VOR DEM LAUF — keine Runner-Minuten verbrannt:')
        for e in errs:
            L.append('  ALARM: %s' % e)
        out = chr(10).join(L) + chr(10)
        Path(out_path).write_text(out, encoding='utf-8')
        print(out)
        sys.exit(1)
    L.append('K-PRE-FLIGHT: OK — Contract strukturell feuerbar.')
    Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    rows = []
    measurements = {}
    for arm in C['arms']:
        name, H = arm['name'], arm['H']
        seeds = arm['seeds']
        pvals = []
        cns = []
        ndiv = 0
        for seed in seeds:
            acc, cn = train_one(seed, arm, H)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            cns.append(cn)
            rows.append(dict(arm=name, rot=arm.get('rot', 'part'),
                             conj_a=int(arm.get('conj_a', 0)), seed=seed,
                             acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            immun=None, artefakt=None)
        L.append('')
        L.append('Arm %s (rot=%s, a=%d, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | median acc=%s | div=%d'
                 % (name, arm.get('rot', 'part'), int(arm.get('conj_a', 0)),
                    seeds[0], seeds[-1], len(ok), len(pvals), pf,
                    fmt(float(torch.tensor(ok).median())) if ok else 'nan',
                    ndiv))
        Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    pp0 = measurements.get('PART_A0', {}).get('pf', float('nan'))
    pp1 = measurements.get('PART_A1', {}).get('pf', float('nan'))
    pf0 = measurements.get('FULL_A0', {}).get('pf', float('nan'))
    pf1 = measurements.get('FULL_A1', {}).get('pf', float('nan'))
    immun = 1.0 if (pp1 <= 0.15 and pf1 <= 0.15) else 0.0
    artefakt = 1.0 if (pp1 <= 0.15 and pf1 >= 0.5) else 0.0
    for a in C['arms']:
        measurements[a['name']]['immun'] = immun
        measurements[a['name']]['artefakt'] = artefakt

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,rot,conj_a,seed,acc' + chr(10))
        for r in rows:
            fh.write('%s,%s,%d,%d,%.6g' % (r['arm'], r['rot'],
                                           r['conj_a'], r['seed'],
                                           r['acc']) + chr(10))

    L.append('')
    L.append('--- T31-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('P-Stufen: P_PART(0)=%.4f | P_PART(1)=%.4f | P_FULL(0)=%.4f | P_FULL(1)=%.4f'
             % (pp0, pp1, pf0, pf1))
    L.append('IMMUN-Klausel: P_PART(1) <= 0.15: %s | P_FULL(1) <= 0.15: %s -> immun = %.1f'
             % (pp1 <= 0.15, pf1 <= 0.15, immun))
    L.append('ARTEFAKT-Klausel: P_PART(1) <= 0.15: %s | P_FULL(1) >= 0.5: %s -> artefakt = %.1f'
             % (pp1 <= 0.15, pf1 >= 0.5, artefakt))
    L.append('L5-BAND (registriert VOR dem Lauf, N55):')
    hits = 0
    for bname, lo, hi in BAND:
        val = {'P_PART(1)': pp1}[bname]
        hit = val == val and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('  FULL-Arme: ERSTFAHRT ohne Band (registriert).')
    L.append('L5-BAND-TREFFER: %d/%d (Hand-Band, N55)' % (hits, len(BAND)))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '55', '2026-11-17')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = chr(10).join(L) + chr(10)
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, immun=immun, artefakt=artefakt,
             band_hits=hits), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
