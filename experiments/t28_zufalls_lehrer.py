#!/usr/bin/env python
'''T28 — ZUFALLS-LEHRER v1 (registriert 2026-11-12 VOR dem
Lauf, XVII Nachtrag 51; Linie TRAIN, zweiter Stein). Kippt die
Konjugation Q_a W_init Q_a^T die Lernbarkeit eines
gradiententrainierten MLP, wenn die AUFGABE hochentropisch ist?
Je Seed ein frischer zufaelliger 2-Schichten-tanh-Lehrer
(8 verborgene Einheiten); Lernkriterium kalibrierungsfrei:
R2 >= 0.9 am Testset (4096 frische Samples, gleicher Lehrer).
MLP H=48, tanh, kein Bias, W1 = Q_a W_init Q_a^T (T20/T25/T27),
Adam lr=5e-3, 800 Steps, batch=64. Stufen a {0,1,24}, Bloecke
970..1029. peff-Doppelklausel wie T27, Sanity P(0) >= 0.5.
MASSE AN JEDEM ARM (R2, maschinelle Schleife).'''
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
from contracts.t28 import contract_t28, STAGES

C = contract_t28()
P = C['params']
FIXED_P0 = P['fixed_p0']
DIN = P['din']
T_HIDDEN = P['t_hidden']
STEPS = P['steps']
BATCH = P['batch']
LR = P['lr']
LAM_MAX = P['lam_max']
WB = P['wb']
Q_SEED = P['q_seed']
R2_PASS = P['r2_pass']
PF_PASS = P['pf_pass']
PEFF_ZERO = P['peff_zero']

BAND = [('DM_init(0)', 8.80, 19.80)]


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
    # T20-Definition, UNVERAENDERT uebernommen (t20_dm_kovariate.py)
    return float(W.diag().abs().mean() / W.abs().mean())


def make_teacher(seed, H):
    # Frischer zufaelliger Lehrer je Seed (HIGH-ENTROPIE-Kern)
    gen = torch.Generator().manual_seed(50000 + seed)
    A1 = torch.randn(H, T_HIDDEN, generator=gen) / (DIN ** 0.5)
    a2 = torch.randn(T_HIDDEN, generator=gen)
    a2 = a2 / a2.norm()
    return A1, a2


def make_batch(n, gen, A1, a2):
    x = torch.randn(n, DIN, generator=gen)
    with torch.no_grad():
        y = torch.tanh(x @ A1) @ a2
    return x, y


def train_one(seed, arm, H):
    gen = torch.Generator().manual_seed(seed)
    W0 = conjugate(init_W(H, seed), int(arm.get('conj_a', 0)), H)
    dm_init = dm_of(W0)
    colnorm = float(W0.abs().mean(dim=0).norm())
    A1, a2 = make_teacher(seed, H)
    W = W0.clone().requires_grad_(True)
    v = torch.zeros(H, requires_grad=True)
    opt = torch.optim.Adam([W, v], lr=LR)
    train_gen = torch.Generator().manual_seed(1000 + seed)
    diverged = False
    for _ in range(STEPS):
        x, y = make_batch(BATCH, train_gen, A1, a2)
        pred = torch.tanh(x @ W.T) @ v
        loss = torch.mean((pred - y) ** 2)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(777)
    xT, yT = make_batch(4096, test_gen, A1, a2)
    with torch.no_grad():
        predT = torch.tanh(xT @ W.T) @ v
        mse = float(torch.mean((predT - yT) ** 2).item())
        var = float(torch.var(yT).item())
    r2 = 1.0 - mse / var if var > 0 else float('nan')
    dm_final = dm_of(W.detach())
    learned = (not diverged) and r2 == r2 and r2 >= R2_PASS
    return dict(learned=learned, r2=r2, diverged=diverged,
                dm_init=dm_init, dm_final=dm_final,
                colnorm=colnorm)


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t28_zufalls_lehrer_results.txt'
    csv_path = 'results/t28_zufalls_lehrer_table.csv'
    ms_path = 'results/t28_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T28 — ZUFALLS-LEHRER v1 (LINIE TRAIN, zweiter Stein; HIGH-ENTROPIE-Aufgabe)')
    L.append('These (Nachtrag 51): Kippt die Konjugation die Lernbarkeit, wenn die')
    L.append('Aufgabe selbst hochentropisch ist? Je Seed ein frischer zufaelliger')
    L.append('tanh-Lehrer (8 hidden); Lernkriterium kalibrierungsfrei: R2 >= 0.9.')
    L.append('MLP H=48, tanh, kein Bias, W1 = Q_a W_init Q_a^T (Q_SEED=2020).')
    L.append('Stufen a {0,1,24}, Bloecke 970..1029, 800 Steps, batch=64, f64.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t28.py')
    L.append('PRIMAER: peff = P(0)-P(1) falls (P(1) <= 0.15 UND P(24) <= 0.15 UND')
    L.append('>= 3*SE gemessen UND P(0) >= 0.5), sonst -1. Grauzone komplementaer.')
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
    arm_pf = {}
    for arm in C['arms']:
        name, H = arm['name'], arm['H']
        seeds = arm['seeds']
        learned_n = 0
        ndiv = 0
        dm_inits = []
        dm_finals = []
        cns = []
        for seed in seeds:
            m = train_one(seed, arm, H)
            if m['diverged'] or m['r2'] != m['r2']:
                ndiv += 1
            if m['learned']:
                learned_n += 1
            dm_inits.append(m['dm_init'])
            if m['dm_final'] == m['dm_final']:
                dm_finals.append(m['dm_final'])
            cns.append(m['colnorm'])
            rows.append(dict(arm=name, conj_a=int(arm.get('conj_a', 0)),
                             seed=seed, learned=int(m['learned']),
                             r2=m['r2'], dm_init=m['dm_init'],
                             dm_final=m['dm_final']))
        pf = learned_n / len(seeds)
        arm_pf[name] = pf
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            dm_init=float(torch.tensor(dm_inits).mean()),
            dm_final=(float(torch.tensor(dm_finals).mean())
                      if dm_finals else float('nan')),
            peff=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, a=%d, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | div=%d | DM_init=%.4f | DM_final=%s'
                 % (name, H, arm['role'], int(arm.get('conj_a', 0)),
                    seeds[0], seeds[-1], len(seeds) - ndiv, len(seeds),
                    pf, ndiv, measurements[name]['dm_init'],
                    fmt(measurements[name]['dm_final'])))
        Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    p0 = arm_pf.get('A0', float('nan'))
    p1 = arm_pf.get('A1', float('nan'))
    p24 = arm_pf.get('A24', float('nan'))
    n0 = len(C['arms'][0]['seeds'])
    n1 = len(C['arms'][1]['seeds'])
    se = (max(p0 * (1 - p0), 1e-9) / n0 + max(p1 * (1 - p1), 1e-9) / n1) ** 0.5
    diff = p0 - p1
    clause = (p1 <= PF_PASS and p24 <= PF_PASS and diff >= 3 * se and p0 >= 0.5)
    peff = diff if clause else -1.0
    for a in C['arms']:
        measurements[a['name']]['peff'] = peff

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,conj_a,seed,learned,r2,dm_init,dm_final' + chr(10))
        for r in rows:
            fh.write('%s,%d,%d,%d,%.6g,%.6g,%.6g' % (
                r['arm'], r['conj_a'], r['seed'], r['learned'],
                r['r2'], r['dm_init'], r['dm_final']) + chr(10))

    L.append('')
    L.append('--- T28-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('P-Stufen: P(0)=%.4f | P(1)=%.4f | P(24)=%.4f' % (p0, p1, p24))
    L.append('SE(P0-P1) = %.4f (gemessen) | 3*SE = %.4f' % (se, 3 * se))
    L.append('DOPPELKLAUSEL: P(1) <= 0.15: %s | P(24) <= 0.15: %s | '
             'Diff >= 3*SE: %s | P(0) >= 0.5: %s -> peff = %.4f'
             % (p1 <= PF_PASS, p24 <= PF_PASS, diff >= 3 * se,
                p0 >= 0.5, peff))
    dm0 = measurements['A0']['dm_final']
    dm1 = measurements['A1']['dm_final']
    if dm0 == dm0 and dm1 == dm1:
        se_dm = max(float(torch.tensor(
            [r['dm_final'] for r in rows if r['arm'] == 'A0']).std().item()), 1e-9)
        L.append('DM_FINAL-BERICHT: DM_final(0) - DM_final(1) = %.4f | '
                 'SE = %.4f | trennt (>= 3*SE): %s'
                 % (dm0 - dm1, se_dm, abs(dm0 - dm1) >= 3 * se_dm))
    L.append('L5-BAND (registriert VOR dem Lauf, Nachtrag 51, R8):')
    hits = 0
    for bname, lo, hi in BAND:
        val = measurements['A0']['dm_init']
        hit = val == val and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('  Aufgabenbezogene Groessen: ERSTFAHRT ohne Band (registriert).')
    L.append('L5-BAND-TREFFER: %d/%d (Hand-Band, Nachtrag 51, R8)' % (hits, len(BAND)))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '51', '2026-11-12')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = chr(10).join(L) + chr(10)
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, peff=peff, se=se,
             band_hits=hits), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
