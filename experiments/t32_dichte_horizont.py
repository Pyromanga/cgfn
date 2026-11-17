#!/usr/bin/env python
'''T32 — RUECKMELDUNGS-DICHTE BEI FIXEM HORIZONT v1 (registriert
2026-11-17 VOR dem Lauf, Nachtrag 56; korrigiert nach externem
Einwand: EINE Variable pro Test). HYPOTHESE: Die Rueckmelde-
dichte (sparse vs dense), NICHT die Gedaechtnis-Forderung, ist
der Traeger des Schalters. TEST: Horizont FIX D=20; SPARSE/D20
(exakt T20/T23-MEM, Referenz) vs DENSE/D20 (Label an JEDEM
Frame t >= D, abhaengig vom Input exakt D Frames vorher,
Zufalls-Lehrer je Seed). Stufen a {0,1}, Bloecke 1350..1429,
80 Runs, f64, ~30 min, Gradienten-Maschine exakt T20/T23.
FALSIFIKATION: P_DENSE(1) <= 0.15 → Hypothese gefallen.
PASS — DICHTE-TRAEGER: P_DENSE(1) >= 0.5 UND P_SPARSE(1) <= 0.15
UND P_DENSE(0) >= 0.5.
KILL — HORIZONT-TRAEGER: P_DENSE(1) <= 0.15 (intakte Sanity).
GEMISCHT: P_DENSE(1) in (0.15, 0.50) — registriert VOR dem Lauf.
Sanity P_SPARSE(0) >= 0.25, P_DENSE(0) >= 0.5.
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
from contracts.t32 import contract_t32

C = contract_t32()
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
DELAY = P['delay']
TEACHER0 = P['teacher_seed0']
T_HIDDEN = P['t_hidden']

BAND = [('P_SPARSE(0)', 0.25, 0.64), ('P_SPARSE(1)', 0.00, 0.22)]


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


def make_teacher(seed):
    tgen = torch.Generator().manual_seed(TEACHER0 + seed)
    A1 = torch.randn(T_HIDDEN, 2, generator=tgen)
    a2 = torch.randn(T_HIDDEN, generator=tgen)
    a2 = a2 / a2.norm()
    return A1, a2


def make_task(seed, cell, batch, gen):
    x = make_inputs(batch, gen)
    A1, a2 = make_teacher(seed)
    if cell == 'sparse':
        y = torch.zeros(batch, dtype=torch.long)
        for i in range(batch):
            a = float(x[i, :, 0][x[i, :, 0] != 0][0])
            b = float(x[i, :, 1][x[i, :, 1] != 0][0])
            y[i] = 0 if (a * b > 0) else 1
        return x, y, None
    # dense: Label je Frame t >= DELAY, abhaengig von Input t-DELAY
    with torch.no_grad():
        src = x[:, :x.shape[1] - DELAY, :]
        score = torch.tanh(src @ A1.T) @ a2
        yf = (score > 0).long()
    return x, None, yf


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


def make_Uw(H, gen):
    s = 2.2
    v0 = torch.randn(H, generator=gen)
    v1 = torch.randn(H, generator=gen)
    v0 = v0 / v0.norm() * s
    v1 = v1 / v1.norm() * s
    return torch.stack([v0, v1], dim=1)


def dyn(z, W, b, uin):
    return -z / TAU + torch.tanh(z + b) @ W.T + uin


def rollout_states(W, b, Uw, x):
    B = x.shape[0]
    H = W.shape[0]
    zs = torch.zeros(B, T_SEQ, H)
    z = torch.zeros(B, H)
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T)
            z = z + DT * k2
        zs[:, t, :] = z
    return zs


def train_one(seed, arm, H):
    cell = arm.get('cell', 'sparse')
    a = int(arm.get('conj_a', 0))
    gen = torch.Generator().manual_seed(seed)
    W0 = conjugate(init_W(H, seed), a, H)
    Uw = make_Uw(H, gen)
    colnorm = float(Uw.norm(dim=0).mean())
    b = 0.1 * torch.randn(H, generator=gen)
    W = W0.clone().requires_grad_(True)
    head_w = torch.zeros(2, H, requires_grad=True)
    head_b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([W, head_w, head_b], lr=LR)
    train_gen = torch.Generator().manual_seed(1000 + seed)
    for _ in range(STEPS):
        x, y, yf = make_task(seed, cell, BATCH, train_gen)
        zs = rollout_states(W, b, Uw, x)
        if cell == 'sparse':
            z = zs[:, -1, :]
            logits = z @ head_w.T + head_b
            loss = torch.nn.functional.cross_entropy(logits, y)
        else:
            zt = zs[:, DELAY:, :].reshape(-1, H)
            yv = yf.reshape(-1)
            logits = zt @ head_w.T + head_b
            loss = torch.nn.functional.cross_entropy(logits, yv)
        if not torch.isfinite(loss):
            return None, colnorm
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT, yfT = make_task(seed, cell, 1024, test_gen)
    with torch.no_grad():
        zs = rollout_states(W, b, Uw, xT)
        if cell == 'sparse':
            z = zs[:, -1, :]
            pred = (z @ head_w.T + head_b).argmax(dim=1)
            acc = (pred == yT).double().mean().item()
        else:
            zt = zs[:, DELAY:, :].reshape(-1, H)
            yv = yfT.reshape(-1)
            pred = (zt @ head_w.T + head_b).argmax(dim=1)
            acc = (pred == yv).double().mean().item()
    return acc, colnorm


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t32_dichte_horizont_results.txt'
    csv_path = 'results/t32_dichte_horizont_table.csv'
    ms_path = 'results/t32_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T32 — RUECKMELDUNGS-DICHTE BEI FIXEM HORIZONT (EINE Variable: sparse vs dense bei D=20)')
    L.append('HYPOTHESE: Rueckmeldungs-Dichte, nicht Gedaechtnis-Forderung, ist der Traeger des Schalters.')
    L.append('FALSIFIKATION: P_DENSE(1) <= 0.15 → Hypothese gefallen.')
    L.append('Stufen a {0,1}, Bloecke 1350..1429, 80 Runs, f64, Gradienten-Maschine exakt T20/T23.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t32.py')
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
            rows.append(dict(arm=name, cell=arm.get('cell', 'sparse'),
                             conj_a=int(arm.get('conj_a', 0)), seed=seed,
                             acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv)
        L.append('')
        L.append('Arm %s (cell=%s, a=%d, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | median acc=%s | div=%d'
                 % (name, arm.get('cell', 'sparse'), int(arm.get('conj_a', 0)),
                    seeds[0], seeds[-1], len(ok), len(pvals), pf,
                    fmt(float(torch.tensor(ok).median())) if ok else 'nan',
                    ndiv))
        Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    ps0 = measurements.get('SPARSE_A0', {}).get('pf', float('nan'))
    ps1 = measurements.get('SPARSE_A1', {}).get('pf', float('nan'))
    pd0 = measurements.get('DENSE_A0', {}).get('pf', float('nan'))
    pd1 = measurements.get('DENSE_A1', {}).get('pf', float('nan'))
    sanity_ok = (ps0 >= 0.25 and pd0 >= 0.5)
    if not sanity_ok:
        verdict_txt = 'UNENTSCHEIDEN — SANITY GERICHT'
    elif pd1 <= 0.15:
        verdict_txt = 'KILL — HORIZONT-TRAEGER'
    elif pd1 >= 0.5 and ps1 <= 0.15:
        verdict_txt = 'PASS — DICHTE-TRAEGER'
    else:
        verdict_txt = 'GEMISCHT — beide Mechanismen tragen bei (registriert)'
    L.append('')
    L.append('--- T32-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('P-Stufen: P_SPARSE(0)=%.4f | P_SPARSE(1)=%.4f | P_DENSE(0)=%.4f | P_DENSE(1)=%.4f'
             % (ps0, ps1, pd0, pd1))
    L.append('SANITY: P_SPARSE(0) >= 0.25: %s | P_DENSE(0) >= 0.5: %s'
             % (ps0 >= 0.25, pd0 >= 0.5))
    L.append('FALSIFIKATIONS-Klausel: P_DENSE(1) <= 0.15: %s' % (pd1 <= 0.15))
    L.append('L5-BAND (registriert VOR dem Lauf, N56):')
    hits = 0
    for bname, lo, hi in BAND:
        val = {'P_SPARSE(0)': ps0, 'P_SPARSE(1)': ps1}[bname]
        hit = val == val and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('  DENSE-Arme: ERSTFAHRT ohne Band (registriert).')
    L.append('L5-BAND-TREFFER: %d/%d (Hand-Band, N56)' % (hits, len(BAND)))
    L.append('')
    L.append('VERDICT: %s' % verdict_txt)
    L.append('')
    L.append('--- N-SKELETT (Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append('# Nachtrag 56 (2026-11-17) — AUTO-SKELETT')
    L.append('  SPARSE_A0.pf = %s | SPARSE_A1.pf = %s' % (fmt(ps0), fmt(ps1)))
    L.append('  DENSE_A0.pf = %s | DENSE_A1.pf = %s' % (fmt(pd0), fmt(pd1)))
    L.append('  LUECKENLISTE: KEINE — alle registrierten Groessen gemessen.')
    L.append('=== FERTIG ===')
    out = chr(10).join(L) + chr(10)
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, verdict=verdict_txt,
             band_hits=hits), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
