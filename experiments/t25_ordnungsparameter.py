#!/usr/bin/env python
'''T25 — ORDNUNGSPARAMETER v1 (registriert 2026-11-12 VOR dem
Lauf, XVII Nachtrag 45; L5-Durchlauf 4, Quelle MASCHINE:
P4-Generator v1.3). Ist der DM-Kollaps (T20-Beifang 13.21 ->
1.40 parallel zum Schalter) ein ORDNUNGSPARAMETER des
Kipppunkts oder ein Begleitprodukt?
Stufen a in {0,1,4,24}, je 1 frischer 20er-Block 750..829 +
Anchor HIGH48 830..849. DM je Seed (T20-Definition, unveraendert:
dm_of(W) = W.diag().abs().mean() / W.abs().mean()).
PRIMAER: dmeff = dmdrop falls (dmdrop >= 5.0 UND dmdrop >=
3 * SE(dmdrop)), sonst -1. PASS dmeff >= 5.0 | KILL < 5.0.
MASSE AN JEDEM ARM inkl. Anchor (R2, maschinelle Schleife).'''
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
from contracts.t25 import contract_t25, STAGES

C = contract_t25()
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
DM_PASS = P['dm_pass']
DM_SE_FACTOR = P['dm_se_factor']

# Band (registriert VOR dem Lauf, Nachtrag 45 / P4-Proposal v1.3,
# R8-Haertung: T20 n=1-Quellen +50% Weite)
BAND = [('DM(0)', 8.80, 19.80), ('DM(1)', 0.90, 2.10),
        ('dmdrop', 5.90, 17.70), ('P(0)', 0.25, 0.64),
        ('P(1)', 0.00, 0.22), ('P(4)', 0.00, 0.30),
        ('P(24)', 0.00, 0.25)]


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
    # T20-Definition, UNVERAENDERT uebernommen (t20_dm_kovariate.py)
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


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def main():
    out_path = 'results/t25_ordnungsparameter_results.txt'
    csv_path = 'results/t25_ordnungsparameter_table.csv'
    ms_path = 'results/t25_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T25 — ORDNUNGSPARAMETER v1 (L5-DURCHLAUF 4, Quelle MASCHINE: P4-Generator v1.3)')
    L.append('These (Nachtrag 45 / P4-Proposal v1.3): Ist der DM-Kollaps (T20:')
    L.append('13.21 -> 1.40 parallel zum Schalter) ein ORDNUNGSPARAMETER des')
    L.append('Kipppunkts (trennt gekippte von intakten Netzen VOR dem Verdikt)')
    L.append('oder ein Begleitprodukt (konsistent mit T20-R2 KILL DM-NEBENSPUR)?')
    L.append('EINE Zelle (H=48, s=2.2, wb=0.6), Stufen a {0,1,4,24},')
    L.append('je 1 frischer 20er-Block 750..829 + Anchor HIGH48 830..849, f64.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t25.py')
    L.append('PRIMAER: dmeff = dmdrop falls (dmdrop >= 5.0 UND >= 3*SE), sonst -1')
    L.append('DM-Definition unveraendert aus T20 (dm_of). R2: jedes Mass an JEDEM Arm.')
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
    arm_dm = {}
    arm_dm_se = {}
    for arm in C['arms']:
        name, H = arm['name'], arm['H']
        seeds = arm['seeds']
        pvals = []
        dms = []
        cns = []
        ndiv = 0
        for seed in seeds:
            acc, dm, cn = train_parity(seed, arm, H)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            dms.append(dm)
            cns.append(cn)
            rows.append(dict(arm=name, H=H, conj_a=int(arm.get('conj_a', 0)),
                             seed=seed, acc=acc, dm=dm))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        dm_m = float(torch.tensor(dms).mean())
        dm_se = float(torch.tensor(dms).std() / (len(dms) ** 0.5))
        arm_pf[name] = pf
        arm_dm[name] = dm_m
        arm_dm_se[name] = dm_se
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            dm=dm_m, dmdrop=None, dmeff=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, a=%d, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | DM=%.4f (SE %.4f) | colnorm=%.4f | div=%d'
                 % (name, H, arm['role'], int(arm.get('conj_a', 0)),
                    seeds[0], seeds[-1], len(ok), len(pvals), pf, dm_m,
                    dm_se, measurements[name]['colnorm'], ndiv))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    # Aggregation: DM-Mittel je Stufe, dmdrop, SE, Doppelklausel -> dmeff
    stage_dm = {a: arm_dm['A%d' % a] for a in STAGES}
    stage_dm_se = {a: arm_dm_se['A%d' % a] for a in STAGES}
    dmdrop = stage_dm[0] - stage_dm[1]
    se_dmdrop = (stage_dm_se[0] ** 2 + stage_dm_se[1] ** 2) ** 0.5
    klausel_ok = (dmdrop >= DM_PASS and dmdrop >= DM_SE_FACTOR * se_dmdrop)
    dmeff = dmdrop if klausel_ok else -1.0

    # R2-READOUT: dmdrop und dmeff an JEDEM Arm inkl. Anchor (maschinell)
    for a in C['arms']:
        measurements[a['name']]['dmdrop'] = dmdrop
        measurements[a['name']]['dmeff'] = dmeff

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,conj_a,seed,acc,dm\n')
        for r in rows:
            fh.write('%s,%d,%d,%d,%.6g,%.6g\n'
                     % (r['arm'], r['H'], r['conj_a'], r['seed'],
                        r['acc'], r['dm']))

    L.append('')
    L.append('--- T25-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('Stufen-Bericht (DM-Mittel je Stufe, 20 Seeds):')
    for a in STAGES:
        L.append('  a=%d: DM=%.4f (SE %.4f) | P=%.2f'
                 % (a, stage_dm[a], stage_dm_se[a], arm_pf.get('A%d' % a)))
    L.append('Anchor HIGH48: pf=%.2f | DM=%.4f'
             % (arm_pf.get('HIGH48'), arm_dm.get('HIGH48')))
    L.append('dmdrop = DM(0) - DM(1) = %.4f | SE(dmdrop) = %.4f | '
             '3*SE = %.4f' % (dmdrop, se_dmdrop, DM_SE_FACTOR * se_dmdrop))
    L.append('DOPPELKLAUSEL: dmdrop >= 5.0: %s | dmdrop >= 3*SE: %s -> '
             'dmeff = %.4f (PASS >= 5.0 | KILL < 5.0)'
             % (dmdrop >= DM_PASS, dmdrop >= DM_SE_FACTOR * se_dmdrop, dmeff))
    # Per-Seed-Bericht (Diagnostik, KEIN Verdikt-Mass): DM getrennt nach
    # gelernt/gescheitert im a=0-Block (Kipp-Zugehoerigkeit)
    a0 = [r for r in rows if r['arm'] == 'A0' and r['acc'] == r['acc']]
    learned = [r['dm'] for r in a0 if r['acc'] >= 0.99]
    failed = [r['dm'] for r in a0 if r['acc'] < 0.99]
    if learned and failed:
        L.append('PER-SEED (a=0, Bericht): DM gelernt %.4f (n=%d) vs '
                 'DM gescheitert %.4f (n=%d)'
                 % (sum(learned) / len(learned), len(learned),
                    sum(failed) / len(failed), len(failed)))
    else:
        L.append('PER-SEED (a=0, Bericht): keine Trennung moeglich '
                 '(gelernt n=%d, gescheitert n=%d)' % (len(learned), len(failed)))
    L.append('L5-BAND (registriert VOR dem Lauf, Nachtrag 45, R8-Haertung):')
    hits = 0
    for bname, lo, hi in BAND:
        val = {'DM(0)': stage_dm[0], 'DM(1)': stage_dm[1],
               'dmdrop': dmdrop, 'P(0)': arm_pf.get('A0'),
               'P(1)': arm_pf.get('A1'), 'P(4)': arm_pf.get('A4'),
               'P(24)': arm_pf.get('A24')}[bname]
        hit = val is not None and val == val and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('L5-BAND-TREFFER: %d/%d (Maschinen-Band, P4-Generator v1.3, R8)'
             % (hits, len(BAND)))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '46', '2026-11-12')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, stage_dm=stage_dm,
             stage_dm_se=stage_dm_se, dmdrop=dmdrop,
             se_dmdrop=se_dmdrop, klausel_ok=klausel_ok, dmeff=dmeff,
             band_hits=hits), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
