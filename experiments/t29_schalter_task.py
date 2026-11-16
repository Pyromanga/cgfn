#!/usr/bin/env python
'''T29 — SCHALTER-TASK v1 (registriert 2026-11-13 VOR dem Lauf,
XVII Nachtrag 53; Konsequenz aus T26+T28). Ist der Reservoir-
Schalter eine GEDAECHTNIS-Eigenschaft oder allgemeine Geometrie?
Exakt die T20/T23-Maschine (H=48, s=2.2, wb=0.6, delayed-recall-
Eingabeprotokoll, Gradienten-Readout, 600 Steps), zwei Aufgaben:
MEM (delayed recall, Referenz) und RND (Zufalls-Lehrer je Seed:
binaeres Label aus frischer tanh-Funktion der Gesamtsequenz).
Stufen a {0,1}, Bloecke 1030..1109, 80 Runs, f64.
dm29-Doppelklausel wie registriert; Sanity RND_A0 >= 0.5,
MEM_A0 >= 0.25. R2: jedes Mass an JEDEM Arm.'''
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
from contracts.t29 import contract_t29

C = contract_t29()
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
TEACHER0 = P['teacher_seed0']
T_HIDDEN = P['t_hidden']
DM29_ZERO = P['dm29_zero']

BAND = [('P_MEM(0)', 0.25, 0.64), ('P_MEM(1)', 0.00, 0.22)]


def make_inputs(batch, gen):
    # Exakt das T20/T23-Eingabeprotokoll (delayed recall)
    x = torch.zeros(batch, T_SEQ, 2)
    for i in range(batch):
        pa = int(torch.randint(A_LO, A_HI, (1,), generator=gen).item())
        pb = int(torch.randint(B_LO, B_HI, (1,), generator=gen).item())
        a = float(2.0 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
        b = float(2.0 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
        x[i, pa:pa + HOLD, 0] = a
        x[i, pb:pb + HOLD, 1] = b
    return x


def make_task(seed, batch, gen, task):
    x = make_inputs(batch, gen)
    if task == 'mem':
        y = torch.zeros(batch, dtype=torch.long)
        for i in range(batch):
            a = float(x[i, :, 0][x[i, :, 0] != 0][0])
            b = float(x[i, :, 1][x[i, :, 1] != 0][0])
            y[i] = 0 if (a * b > 0) else 1
        return x, y
    # RND: frischer Zufalls-Lehrer je Seed (Lehre T28)
    tgen = torch.Generator().manual_seed(TEACHER0 + seed)
    A1 = torch.randn(T_SEQ * 2, T_HIDDEN, generator=tgen)
    a2 = torch.randn(T_HIDDEN, generator=tgen)
    a2 = a2 / a2.norm()
    with torch.no_grad():
        score = torch.tanh(x.reshape(batch, T_SEQ * 2) @ A1) @ a2
    y = (score > 0).long()
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


def conjugate(W, a, H):
    if a == 0:
        return W
    genQ = torch.Generator().manual_seed(Q_SEED)
    G = torch.randn(H, H, generator=genQ)
    Q, _ = torch.linalg.qr(torch.eye(H) + a * G)
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
    z = torch.zeros(B, W.shape[0])
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T)
            z = z + DT * k2
    return z


def train_one(seed, arm, H):
    task = arm.get('task', 'mem')
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
        x, y = make_task(seed, BATCH, train_gen, task)
        z = rollout(W, b, Uw, x)
        logits = z @ head_w.T + head_b
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            return None, colnorm
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(seed, 1024, test_gen, task)
    with torch.no_grad():
        z = rollout(W, b, Uw, xT)
        acc = ((z @ head_w.T + head_b).argmax(dim=1) == yT).double().mean().item()
    return acc, colnorm


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t29_schalter_task_results.txt'
    csv_path = 'results/t29_schalter_task_table.csv'
    ms_path = 'results/t29_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T29 — SCHALTER-TASK v1 (Konsequenz aus T26+T28: Gedaechtnis oder Geometrie?)')
    L.append('These (Nachtrag 53): Kollabiert P bei a >= 1 nur fuer GEDAECHTNIS-Aufgaben')
    L.append('(MEM, Referenz T23/T24/T25) oder auch fuer Zufalls-Lehrer (RND, Lehre T28)?')
    L.append('Exakt die T20/T23-Maschine, zwei Aufgaben auf demselben Eingabeprotokoll.')
    L.append('Stufen a {0,1}, Bloecke 1030..1109, 80 Runs, f64.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t29.py')
    L.append('PRIMAER: dm29 = P_RND(1) - P_MEM(1) mit Doppelklausel, sonst -1.')
    L.append('R2: jedes Mass an JEDEM Arm.')
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
            rows.append(dict(arm=name, task=arm.get('task', 'mem'),
                             conj_a=int(arm.get('conj_a', 0)), seed=seed,
                             acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            dm29=None)
        L.append('')
        L.append('Arm %s (task=%s, a=%d, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | median acc=%s | div=%d'
                 % (name, arm.get('task', 'mem'), int(arm.get('conj_a', 0)),
                    seeds[0], seeds[-1], len(ok), len(pvals), pf,
                    fmt(float(torch.tensor(ok).median())) if ok else 'nan',
                    ndiv))
        Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    pm0 = measurements.get('MEM_A0', {}).get('pf', float('nan'))
    pm1 = measurements.get('MEM_A1', {}).get('pf', float('nan'))
    pr0 = measurements.get('RND_A0', {}).get('pf', float('nan'))
    pr1 = measurements.get('RND_A1', {}).get('pf', float('nan'))
    n = 20
    se = (max(pr1 * (1 - pr1), 1e-9) / n + max(pm1 * (1 - pm1), 1e-9) / n) ** 0.5
    diff = pr1 - pm1
    clause = (pr1 >= 0.5 and pm1 <= 0.15 and pr0 >= 0.5 and pm0 >= 0.25 and diff >= 3 * se)
    dm29 = diff if clause else -1.0
    for a in C['arms']:
        measurements[a['name']]['dm29'] = dm29

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,task,conj_a,seed,acc' + chr(10))
        for r in rows:
            fh.write('%s,%s,%d,%d,%.6g' % (r['arm'], r['task'],
                                           r['conj_a'], r['seed'],
                                           r['acc']) + chr(10))

    L.append('')
    L.append('--- T29-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('P-Stufen: P_MEM(0)=%.4f | P_MEM(1)=%.4f | P_RND(0)=%.4f | P_RND(1)=%.4f'
             % (pm0, pm1, pr0, pr1))
    L.append('SE(P_RND1 - P_MEM1) = %.4f (gemessen) | 3*SE = %.4f' % (se, 3 * se))
    L.append('DOPPELKLAUSEL: P_RND(1) >= 0.5: %s | P_MEM(1) <= 0.15: %s | '
             'P_RND(0) >= 0.5: %s | P_MEM(0) >= 0.25: %s | Diff >= 3*SE: %s '
             '-> dm29 = %.4f'
             % (pr1 >= 0.5, pm1 <= 0.15, pr0 >= 0.5, pm0 >= 0.25,
                diff >= 3 * se, dm29))
    L.append('L5-BAND (registriert VOR dem Lauf, Nachtrag 53, R8):')
    hits = 0
    for bname, lo, hi in BAND:
        val = {'P_MEM(0)': pm0, 'P_MEM(1)': pm1}[bname]
        hit = val == val and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('  RND-Arme: ERSTFAHRT ohne Band (registriert).')
    L.append('L5-BAND-TREFFER: %d/%d (Hand-Band, Nachtrag 53, R8)' % (hits, len(BAND)))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '53', '2026-11-13')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = chr(10).join(L) + chr(10)
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, dm29=dm29, se=se,
             band_hits=hits), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
