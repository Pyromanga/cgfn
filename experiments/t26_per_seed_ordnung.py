#!/usr/bin/env python
'''T26 — PER-SEED-ORDNUNG v1 (registriert 2026-11-12 VOR dem
Lauf, XVII Nachtrag 47; L5-Durchlauf 5, Quelle MASCHINE:
P4-Generator v1.4). Trennt UEBERHAUPT eine messbare Groesse
gelernte von gescheiterten Seeds INNERHALB eines Blocks?
T25-Beyfang: DM gelernt 13.133 (n=11) vs gescheitert 12.974
(n=9), Differenz 0.159 < SE 0.21 — kein Trenner.
Stufen a in {0,1}, je 1 frischer 20er-Block 850..889 + Anchor
HIGH48 890..909. Kandidaten (fixiert VOR dem Readout): dm,
initloss, gainmean, gainstd, wnorm. sepmax = max t_c
(zweiseitig, >= 3 * SE gemessen, Richtung konstant a=0/a=1),
sonst -1. PASS sepmax >= 3.0 | KILL < 3.0.
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
from contracts.t26 import contract_t26, STAGES

C = contract_t26()
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
SEP_PASS = P['sep_pass']
CANDIDATES = P['candidates']

BAND = [('DM_gelernt', 8.80, 19.70), ('DM_gescheitert', 8.60, 19.50),
        ('Differenz_DM', 0.11, 0.24), ('P(0)', 0.25, 0.64),
        ('P(1)', 0.00, 0.22)]


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
    z = torch.zeros(B, W.shape[0])
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
    wnorm = float(W0.norm())
    Uw = make_Uw(arm, H, gen)
    colnorm = float(Uw.norm(dim=0).mean())
    b = 0.1 * torch.randn(H, generator=gen)
    # Kandidaten-Masse am INITIALISIERTEN Netz (fixiert, R2)
    head_w0 = torch.zeros(2, H)
    train_gen0 = torch.Generator().manual_seed(1000 + seed)
    x0, y0 = make_task(BATCH, train_gen0)
    with torch.no_grad():
        z0 = rollout(W0, b, Uw, x0)
        initloss = float(torch.nn.functional.cross_entropy(
            z0 @ head_w0.T, y0).item())
        g = torch.tanh(z0).abs()
        gainmean = float(g.mean().item())
        gainstd = float(g.std().item())
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
            return None, dict(dm=dm, initloss=initloss,
                              gainmean=gainmean, gainstd=gainstd,
                              wnorm=wnorm), colnorm
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(1024, test_gen)
    with torch.no_grad():
        z = rollout(W, b, Uw, xT)
        acc = ((z @ head_w.T + head_b).argmax(dim=1) == yT).double().mean().item()
    return acc, dict(dm=dm, initloss=initloss, gainmean=gainmean,
                     gainstd=gainstd, wnorm=wnorm), colnorm


def fmt(v):
    return 'NICHT GEMESSEN' if v is None else '%.4f' % v


def sep_t(vals_learned, vals_failed):
    # zweiseitig: t = |m_l - m_f| / SE_diff; -1 wenn nicht berechenbar
    if len(vals_learned) < 2 or len(vals_failed) < 2:
        return -1.0, float('nan')
    ml = sum(vals_learned) / len(vals_learned)
    mf = sum(vals_failed) / len(vals_failed)
    sl = torch.tensor(vals_learned).std().item() if len(vals_learned) > 1 else 0.0
    sf = torch.tensor(vals_failed).std().item() if len(vals_failed) > 1 else 0.0
    se = (sl ** 2 / len(vals_learned) + sf ** 2 / len(vals_failed)) ** 0.5
    se = max(se, 1e-9)
    return abs(ml - mf) / se, (ml - mf)


def main():
    out_path = 'results/t26_per_seed_ordnung_results.txt'
    csv_path = 'results/t26_per_seed_ordnung_table.csv'
    ms_path = 'results/t26_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T26 — PER-SEED-ORDNUNG v1 (L5-DURCHLAUF 5, Quelle MASCHINE: P4-Generator v1.4)')
    L.append('These (Nachtrag 47 / P4-Proposal v1.4): Trennt UEBERHAUPT eine messbare')
    L.append('Groesse gelernte von gescheiterten Seeds INNERHALB eines Blocks?')
    L.append('T25-Beyfang: DM gelernt 13.133 (n=11) vs gescheitert 12.974 (n=9),')
    L.append('Differenz 0.159 < SE — kein Trenner. Kandidaten (fixiert): %s' % ', '.join(CANDIDATES))
    L.append('EINE Zelle (H=48, s=2.2, wb=0.6), Stufen a {0,1}, je 1 frischer')
    L.append('20er-Block 850..889 + Anchor HIGH48 890..909, f64.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t26.py')
    L.append('PRIMAER: sepmax = max t_c (zweiseitig, >= 3*SE gemessen, Richtung')
    L.append('konstant a=0/a=1), sonst -1. PASS sepmax >= 3.0 | KILL < 3.0.')
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
    arm_sep = {}
    for arm in C['arms']:
        name, H = arm['name'], arm['H']
        seeds = arm['seeds']
        pvals = []
        cns = []
        ndiv = 0
        cand_rows = []
        for seed in seeds:
            acc, cand, cn = train_parity(seed, arm, H)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            cns.append(cn)
            cand['acc'] = acc
            cand_rows.append(cand)
            row = dict(arm=name, H=H, conj_a=int(arm.get('conj_a', 0)), seed=seed, acc=acc)
            row.update(cand)
            rows.append(row)
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        arm_pf[name] = pf
        learned_rows = [r for r in cand_rows if r['acc'] == r['acc'] and r['acc'] >= 0.99]
        failed_rows = [r for r in cand_rows if r['acc'] == r['acc'] and r['acc'] < 0.99]
        tvals = {}
        dvals = {}
        for c in CANDIDATES:
            t, d = sep_t([r[c] for r in learned_rows], [r[c] for r in failed_rows])
            tvals[c] = t
            dvals[c] = d
        arm_sep[name] = dict(tvals=tvals, dvals=dvals)
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv,
            sepmax=None)
        L.append('')
        L.append('Arm %s (H=%d, role=%s, a=%d, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | gelernt n=%d, gescheitert n=%d | div=%d'
                 % (name, H, arm['role'], int(arm.get('conj_a', 0)),
                    seeds[0], seeds[-1], len(ok), len(pvals), pf,
                    len(learned_rows), len(failed_rows), ndiv))
        for c in CANDIDATES:
            L.append('  %s: t=%.2f (Diff %s)' % (c, tvals[c], fmt(dvals[c])))
        Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    def sign(x):
        return 0 if x != x else (1 if x > 0 else (-1 if x < 0 else 0))
    sepmax = -1.0
    winner = None
    for c in CANDIDATES:
        t0c = arm_sep['A0']['tvals'][c]
        d0 = arm_sep['A0']['dvals'][c]
        d1 = arm_sep['A1']['dvals'][c]
        if t0c >= SEP_PASS and sign(d0) != 0 and sign(d0) == sign(d1):
            if t0c > sepmax:
                sepmax = t0c
                winner = c
    for a in C['arms']:
        measurements[a['name']]['sepmax'] = sepmax

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,conj_a,seed,acc,dm,initloss,gainmean,gainstd,wnorm' + chr(10))
        for r in rows:
            fh.write('%s,%d,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g' % (
                r['arm'], r['H'], r['conj_a'], r['seed'], r['acc'],
                r['dm'], r['initloss'], r['gainmean'], r['gainstd'],
                r['wnorm']) + chr(10))

    L.append('')
    L.append('--- T26-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('SEP-Bericht je Kandidat (zweiseitig, SE im Lauf gemessen):')
    for c in CANDIDATES:
        L.append('  %s: t(a=0)=%.2f | t(a=1)=%.2f | Diff a=0 %s, a=1 %s'
                 % (c, arm_sep['A0']['tvals'][c], arm_sep['A1']['tvals'][c],
                    fmt(arm_sep['A0']['dvals'][c]), fmt(arm_sep['A1']['dvals'][c])))
    L.append('sepmax = %.2f (Gewinner: %s) | PASS-Schwelle %.1f'
             % (sepmax, winner, SEP_PASS))
    a0l = [r['dm'] for r in rows if r['arm'] == 'A0' and r['acc'] == r['acc'] and r['acc'] >= 0.99]
    a0f = [r['dm'] for r in rows if r['arm'] == 'A0' and r['acc'] == r['acc'] and r['acc'] < 0.99]
    dm_l = sum(a0l) / len(a0l) if a0l else float('nan')
    dm_f = sum(a0f) / len(a0f) if a0f else float('nan')
    d_dm = dm_l - dm_f if (dm_l == dm_l and dm_f == dm_f) else float('nan')
    bandvals = {'DM_gelernt': dm_l, 'DM_gescheitert': dm_f,
                'Differenz_DM': d_dm, 'P(0)': arm_pf.get('A0'),
                'P(1)': arm_pf.get('A1')}
    L.append('L5-BAND (registriert VOR dem Lauf, Nachtrag 47, R8-Haertung):')
    hits = 0
    for bname, lo, hi in BAND:
        val = bandvals[bname]
        hit = val is not None and val == val and lo <= val <= hi
        hits += int(hit)
        L.append('  %s in [%.2f, %.2f]: gemessen %s — %s'
                 % (bname, lo, hi, fmt(val), 'TREFFER' if hit else 'NEIN'))
    L.append('L5-BAND-TREFFER: %d/%d (Maschinen-Band, P4-Generator v1.4, R8)'
             % (hits, len(BAND)))

    res = lab_kernel.evaluate(C, measurements)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    skelett = lab_kernel.auto_nachtrag(C, res, '49', '2026-11-12')
    L.append('')
    L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = chr(10).join(L) + chr(10)
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements,
             sep_tvals={k: v['tvals'] for k, v in arm_sep.items()},
             sep_dvals={k: v['dvals'] for k, v in arm_sep.items()},
             sepmax=sepmax, winner=winner, band_hits=hits), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s | %s' % (out_path, csv_path, ms_path))


if __name__ == '__main__':
    main()
