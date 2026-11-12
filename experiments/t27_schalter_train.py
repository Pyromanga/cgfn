#!/usr/bin/env python
'''T27 — SCHALTER-TRAIN v1 (registriert 2026-11-12 VOR dem
Lauf, XVII Nachtrag 48; Linie TRAIN, erster Gradienten-Stein).
Kippt ein GRADIENTEN-TRAINIERTES Netz denselben Schalter wie
das eingefrorene Reservoir? MLP H=48, tanh, kein Bias,
Head 2x48. Init: W1 = Q_a W_init Q_a^T (T20/T25-Konstruktion,
Q_SEED=2020). Adam lr=5e-3, 600 Steps, batch=32, delayed
recall (t_seq=24), Test 1024. Stufen a {0,1,24}, Bloecke
910..969. PRIMAER: peff = P(0)-P(1) falls (P(1) <= 0.15 UND
P(24) <= 0.15 UND >= 3 * SE gemessen UND P(0) im Band),
sonst -1. PASS peff >= 0.0 | KILL < 0.0 (Grauzone komplementaer,
registriert). DM_final als Ordnungsparameter-Bericht (TEIL-
VERDICT SCHALTER-OHNE-ORDNUNGSPARAMETER).
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
from contracts.t27 import contract_t27, STAGES

C = contract_t27()
P = C['params']
FIXED_P0 = P['fixed_p0']
TEST_SEED = P['test_seed']
T_SEQ = P['t_seq']
STEPS = P['steps']
BATCH = P['batch']
LR = P['lr']
LAM_MAX = P['lam_max']
WB = P['wb']
Q_SEED = P['q_seed']
A_LO, A_HI = P['a_lo'], P['a_hi']
B_LO, B_HI = P['b_lo'], P['b_hi']
HOLD = P['hold']
PF_PASS = P['pf_pass']
DIN = T_SEQ * 2  # Input-Dim 48 = H (bewusst: gleiche Konstruktion)

BAND = [('P(0)', 0.25, 0.64), ('P(1)', 0.00, 0.30),
        ('P(24)', 0.00, 0.30), ('DM_init(0)', 8.80, 19.80)]


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
    # T20-Definition, UNVERAENDERT (am W1 des MLP)
    return float(W.diag().abs().mean() / W.abs().mean())


def train_mlp(seed, a, H):
    gen = torch.Generator().manual_seed(seed)
    W1_0 = conjugate(init_W(H, seed), a, H)
    dm_init = dm_of(W1_0)
    cn = float(W1_0.norm(dim=0).mean())
    # MLP: x (B, DIN) -> tanh(x @ W1.T) -> logits @ head.T, KEIN Bias
    W1 = W1_0.clone().requires_grad_(True)
    head = torch.zeros(2, H, requires_grad=True)
    opt = torch.optim.Adam([W1, head], lr=LR)
    train_gen = torch.Generator().manual_seed(1000 + seed)
    diverged = False
    for _ in range(STEPS):
        x2, y = make_task(BATCH, train_gen)
        xf = x2.reshape(x2.shape[0], DIN)
        logits = torch.tanh(xf @ W1.T) @ head.T
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    if diverged:
        return None, dm_init, float('nan'), cn
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(1024, test_gen)
    with torch.no_grad():
        xf = xT.reshape(xT.shape[0], DIN)
        acc = ((torch.tanh(xf @ W1.T) @ head.T).argmax(dim=1) == yT).double().mean().item()
        dm_final = dm_of(W1.detach())
    return acc, dm_init, dm_final, cn


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t27_schalter_train_results.txt'
    csv_path = 'results/t27_schalter_train_table.csv'
    ms_path = 'results/t27_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T27 — SCHALTER-TRAIN v1 (LINIE TRAIN, erster Gradienten-Stein)')
    L.append('These (Nachtrag 48): Kippt ein GRADIENTEN-TRAINIERTES Netz denselben')
    L.append('Schalter wie das eingefrorene Reservoir? MLP H=48, tanh, kein Bias,')
    L.append('Init-Konjugation exakt T20/T25 (Q_SEED=2020). Stufen a {0,1,24},')
    L.append('je 1 frischer 20er-Block 910..969, f64.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t27.py')
    L.append('PRIMAER: peff = P(0)-P(1) falls (P(1) <= 0.15 UND P(24) <= 0.15 UND')
    L.append('>= 3*SE gemessen UND P(0) im Band), sonst -1. Grauzone komplementaer.')
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
    arm_dm_init = {}
    arm_dm_final = {}
    arm_dm_final_se = {}
    for arm in C['arms']:
        name, H = arm['name'], arm['H']
        a = int(arm.get('conj_a', 0))
        seeds = arm['seeds']
        pvals = []
        dmi = []
        dmf = []
        cns = []
        ndiv = 0
        for seed in seeds:
            acc, dm_i, dm_f, cn = train_mlp(seed, a, H)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
                dm_f = float('nan')
            pvals.append(acc)
            dmi.append(dm_i)
            dmf.append(dm_f)
            cns.append(cn)
            rows.append(dict(arm=name, H=H, conj_a=a, seed=seed, acc=acc,
                             dm_init=dm_i, dm_final=dm_f))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        arm_pf[name] = pf
        arm_dm_init[name] = float(torch.tensor(dmi).mean())
        dmf_ok = [v for v in dmf if v == v]
        arm_dm_final[name] = float(torch.tensor(dmf_ok).mean()) if dmf_ok else float('nan')
        arm_dm_final_se[name] = float(torch.tensor(dmf_ok).std() / (len(dmf_ok) ** 0.5)) if len(dmf_ok) > 1 else float('nan')
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            dm_init=arm_dm_init[name], dm_final=arm_dm_final[name],
            peff=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, a=%d, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | DM_init=%.4f | DM_final=%.4f (SE %.4f) | div=%d'
                 % (name, H, arm['role'], a, seeds[0], seeds[-1],
                    len(ok), len(pvals), pf, arm_dm_init[name],
                    arm_dm_final[name], arm_dm_final_se[name], ndiv))
        Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    p0, p1, p24 = arm_pf.get('A0'), arm_pf.get('A1'), arm_pf.get('A24')
    a0acc = [r['acc'] for r in rows if r['arm'] == 'A0' and r['acc'] == r['acc']]
    a1acc = [r['acc'] for r in rows if r['arm'] == 'A1' and r['acc'] == r['acc']]
    se0 = float(torch.tensor(a0acc).std() / max(1, len(a0acc)) ** 0.5) if len(a0acc) > 1 else 0.0
    se1 = float(torch.tensor(a1acc).std() / max(1, len(a1acc)) ** 0.5) if len(a1acc) > 1 else 0.0
    se_p = (se0 ** 2 + se1 ** 2) ** 0.5
    band_ok = p0 is not None and p0 == p0 and 0.25 <= p0 <= 0.64
    cond = (p1 is not None and p1 == p1 and p24 is not None and p24 == p24
            and p1 <= PF_PASS and p24 <= PF_PASS
            and (p0 - p1) >= 3 * se_p and band_ok)
    peff = (p0 - p1) if cond else -1.0
    for a in C['arms']:
        measurements[a['name']]['peff'] = peff

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,conj_a,seed,acc,dm_init,dm_final' + chr(10))
        for r in rows:
            fh.write('%s,%d,%d,%d,%.6g,%.6g,%.6g' % (
                r['arm'], r['H'], r['conj_a'], r['seed'], r['acc'],
                r['dm_init'], r['dm_final']) + chr(10))

    L.append('')
    L.append('--- T27-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('P-Stufen: P(0)=%s | P(1)=%s | P(24)=%s'
             % (fmt(p0), fmt(p1), fmt(p24)))
    L.append('SE(P0-P1) = %.4f (gemessen) | 3*SE = %.4f' % (se_p, 3 * se_p))
    L.append('DOPPELKLAUSEL: P(1) <= 0.15: %s | P(24) <= 0.15: %s | '
             'Diff >= 3*SE: %s | P(0) im Band: %s -> peff = %.4f'
             % (p1 is not None and p1 <= PF_PASS,
                p24 is not None and p24 <= PF_PASS,
                (p0 - p1) >= 3 * se_p, band_ok, peff))
    dmf01 = arm_dm_final.get('A0', float('nan')) - arm_dm_final.get('A1', float('nan'))
    se_f01 = (arm_dm_final_se.get('A0', float('nan')) ** 2
              + arm_dm_final_se.get('A1', float('nan')) ** 2) ** 0.5
    dm_trennt = dmf01 == dmf01 and se_f01 == se_f01 and abs(dmf01) >= 3 * se_f01
    L.append('DM_FINAL-BERICHT: DM_final(0) - DM_final(1) = %.4f | '
             'SE = %.4f | trennt (>= 3*SE): %s'
             % (dmf01, se_f01, dm_trennt))
    if peff >= 0.0 and not dm_trennt:
        L.append('TEIL-VERDICT (Bericht): SCHALTER-OHNE-ORDNUNGSPARAMETER — '
                 'der Schalter transferiert, der Marker nicht.')
    bandvals = {'P(0)': p0, 'P(1)': p1, 'P(24)': p24,
                'DM_init(0)': arm_dm_init.get('A0')}
    L.append('L5-BAND (registriert VOR dem Lauf, Nachtrag 48, R8-Haertung):')
    hits = 0
    for bname, lo, hi in BAND:
        val = bandvals[bname]
        hit = val is not None and val == val and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('L5-BAND-TREFFER: %d/%d (Band Nachtrag 48, R8)' % (hits, len(BAND)))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '50', '2026-11-12')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = chr(10).join(L) + chr(10)
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, peff=peff, se_p=se_p,
             dm_final_diff=dmf01, dm_trennt=bool(dm_trennt),
             band_hits=hits), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
