#!/usr/bin/env python
'''T24 — SCHALTER-RICHTUNG v1 (registriert 2026-11-11 VOR dem
Lauf, XVII Nachtrag 43; L5-Durchlauf 3, Quelle MASCHINE:
P4-Generator v1.2). Kippt JEDE 1-dim-Konjugation das Lernen
(RICHTUNGS-AGNOSTISCH, Symmetriebruch) oder nur bestimmte
Rotationsachsen (RICHTUNGS-SPEZIFISCH, Achsen-Lotto)?
EINE Zelle (H=48, s=2.2, wb=0.6), a=1, 5 Achsen q_seed in
{2020, 5150, 8080, 1111, 4242} je 20 frische Seeds 610..709,
Control a=0 (710..729), Anchor HIGH48 (730..749).
PRIMAER: pqmax = max(P_Q); PASS <= 0.15 | KILL >= 0.30.
KEINE std-Klausel (Praezedenz Nachtrag 42: pqmax hat keinen
Nenner). Masse pq/pqmax an JEDEM Arm inkl. Anchor (R2).'''
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
from contracts.t24 import contract_t24, Q_SEEDS

C = contract_t24()
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
PQ_PASS = P['pq_pass']
PQ_KILL = P['pq_kill']
A_LO, A_HI = P['a_lo'], P['a_hi']
B_LO, B_HI = P['b_lo'], P['b_hi']
HOLD = P['hold']

# Band (registriert VOR dem Lauf, Nachtrag 43 / P4-Proposal v1.2,
# R8-Haertung: n=1-Quellen +50% Weite)
BAND = [('P_Q(2020)', 0.00, 0.22), ('P_Q(5150)', 0.00, 0.22),
        ('P_Q(8080)', 0.00, 0.22), ('P_Q(1111)', 0.00, 0.22),
        ('P_Q(4242)', 0.00, 0.22), ('pqmax', 0.05, 0.25),
        ('P(0)', 0.25, 0.64)]


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


def conjugate(W, a, H, q_seed):
    if a == 0:
        return W
    genQ = torch.Generator().manual_seed(q_seed)
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
    W0 = conjugate(init_W(H, seed), int(arm.get('conj_a', 0)), H,
                   int(arm.get('q_seed', 2020)))
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


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t24_schalter_richtung_results.txt'
    csv_path = 'results/t24_schalter_richtung_table.csv'
    ms_path = 'results/t24_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T24 — SCHALTER-RICHTUNG v1 (L5-DURCHLAUF 3, Quelle MASCHINE: P4-Generator v1.2)')
    L.append('These (Nachtrag 43 / P4-Proposal v1.2): Der a=1-Schalter (T23) ist')
    L.append('RICHTUNGS-AGNOSTISCH (jede Achse kippt, Symmetriebruch) oder')
    L.append('RICHTUNGS-SPEZIFISCH (Achsen-Lotto). EINE Zelle (H=48, s=2.2, wb=0.6),')
    L.append('a=1, 5 Achsen q_seed {2020, 5150, 8080, 1111, 4242}, Konjugation Q W Q^T.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t24.py')
    L.append('Je Achse 20 frische Seeds (610..709) + Control a=0 (710..729)')
    L.append('+ Anchor HIGH48 (730..749), f64 | pqmax an JEDEM Arm (R2)')
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
    arm_pf = {}
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
            rows.append(dict(arm=name, H=H, conj_a=int(arm.get('conj_a', 0)),
                             q_seed=int(arm.get('q_seed', 2020)),
                             seed=seed, acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        arm_pf[name] = pf
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            pq=None, pqmax=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, a=%d, q_seed=%d, s=%s, seeds %d..%d): '
                 'n_ok=%d/%d | P_lernt=%.2f | colnorm=%.4f | div=%d'
                 % (name, H, arm['role'], int(arm.get('conj_a', 0)),
                    int(arm.get('q_seed', 2020)), arm['s'], seeds[0],
                    seeds[-1], len(ok), len(pvals), pf,
                    measurements[name]['colnorm'], ndiv))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    # Achsen-Aggregation (ein Block je Achse; Block-Gesetz R1)
    axis_p = {}
    for qs in Q_SEEDS:
        axis_p[qs] = arm_pf.get('Q%d' % qs)
    p_ctrl = arm_pf.get('A0_CTRL')
    pqmax = None
    vals = [axis_p[qs] for qs in Q_SEEDS if axis_p[qs] == axis_p[qs]]
    if vals:
        pqmax = max(vals)

    # R2-READOUT: pq und pqmax an JEDEM Arm inkl. Anchor (T21/T23-Lektion,
    # maschinell, keine Ausnahme)
    for a in C['arms']:
        measurements[a['name']]['pq'] = arm_pf.get(a['name'])
        measurements[a['name']]['pqmax'] = pqmax

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,conj_a,q_seed,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%d,%d,%d,%.6g\n'
                     % (r['arm'], r['H'], r['conj_a'], r['q_seed'],
                        r['seed'], r['acc']))

    L.append('')
    L.append('--- T24-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('Achsen-Bericht (Block-P je Achse, a=1):')
    for qs in Q_SEEDS:
        L.append('  q_seed=%d: P_Q=%s' % (qs, fmt(axis_p[qs])))
    L.append('Control a=0: P(0)=%s | Anchor HIGH48: pf=%s'
             % (fmt(p_ctrl), fmt(arm_pf.get('HIGH48'))))
    L.append('SE(P_Block) = 0.068 (T22 10-Block-Statistik)')
    L.append('pqmax = max(P_Q) = %s (PASS <= %.2f | KILL >= %.2f,'
             ' Luecke = UNENTSCHEIDEN)' % (fmt(pqmax), PQ_PASS, PQ_KILL))
    L.append('L5-BAND (registriert VOR dem Lauf, Nachtrag 43, R8-Haertung):')
    hits = 0
    for bname, lo, hi in BAND:
        val = {'P_Q(2020)': axis_p.get(2020), 'P_Q(5150)': axis_p.get(5150),
               'P_Q(8080)': axis_p.get(8080), 'P_Q(1111)': axis_p.get(1111),
               'P_Q(4242)': axis_p.get(4242), 'pqmax': pqmax,
               'P(0)': p_ctrl}[bname]
        hit = val is not None and val == val and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('L5-BAND-TREFFER: %d/%d (Maschinen-Band, P4-Generator v1.2, R8)'
             % (hits, len(BAND)))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '44', '2026-11-11')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, axis_p={str(k): axis_p[k] for k in axis_p},
             p_ctrl=p_ctrl, pqmax=pqmax, band_hits=hits), indent=1),
        encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
