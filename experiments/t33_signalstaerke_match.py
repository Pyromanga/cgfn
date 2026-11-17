#!/usr/bin/env python
'''T33 — SIGNALSTRAEKE-MATCH (T32-Konfund) v1 (registriert
2026-11-17 VOR dem Lauf, Nachtrag 57). HYPOTHESE (T32): Die
Rueckmeldungs-Dichte, nicht die Gedaechtnis-Forderung, ist der
Traeger des Schalters. OFFENER KONFUND aus T32: dense variierte
auch die Optimierungs-Dynamik (4 Label-Frames statt 1; VARIANZ
des Gradienten sinkt; Korrektur: 4x, nicht ~20x; Magnitude je
Step bei Mean-Reduction vergleichbar).
TEST: Vier Arme, a {0,1}:
  MB (DENSE_MASK, 600 Steps): dichte Aufgabe, je Step nur EINES
    der 4 Label-Frames im Loss (zyklisch k mod 4) — matcht
    Terme/Step, Varianz und Total-Signal bei erhaltener dichter
    Aufgabenstruktur.
  S4X (SPARSE, 2400 Steps): sparse mit 4x Training ("zu wenig
    Training?" als Alternative).
Bloecke 1430..1509, 80 Runs, f64, ~75-90 min. Referenz aus T32.
PASS — DICHTE-TRAEGER (geschlossen): P_MB(1) >= 0.5 UND
  P_S4X(1) <= 0.15.
GEMISCHT — REDUNDANZ: P_MB(1) >= 0.5 UND P_S4X(1) >= 0.5 —
  dichte Aufgabe UND dichtes Total-Signal koennen den Kollaps
  je verhindern; keins ist notwendig (Befund, registriert).
GEMISCHT — PER-FRAME-EXPOSITIONS-KONFUND: P_MB(1) <= 0.15 UND
  P_S4X(1) >= 0.5 — KILL der Dichte-These erst mit MB-2400
  (nicht gefahren); MB-600 hat die geringste Per-Frame-
  Exposition (150 vs 600/600/2400), registriert N57-Erg.
GEMISCHT — ZWISCHENWELT: sonst bei intakter Sanity.
Sanity P_MB(0) >= 0.5, P_S4X(0) >= 0.5.
Baender: ALLE Arme ERSTFAHRT ohne Band (neue Optimierungs-Regime).
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
from contracts.t33 import contract_t33

C = contract_t33()
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

BAND = []  # ALLE Arme ERSTFAHRT ohne Band (registriert, N57)


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


def make_task(seed, cell_task, batch, gen):
    x = make_inputs(batch, gen)
    A1, a2 = make_teacher(seed)
    if cell_task == 'sparse':
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
    cell_task = arm.get('cell_task', 'dense' if cell == 'dense_mask' else 'sparse')
    a = int(arm.get('conj_a', 0))
    n_steps = int(arm.get('steps', STEPS))
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
    for k in range(n_steps):
        x, y, yf = make_task(seed, cell_task, BATCH, train_gen)
        zs = rollout_states(W, b, Uw, x)
        if cell_task == 'sparse':
            z = zs[:, -1, :]
            logits = z @ head_w.T + head_b
            loss = torch.nn.functional.cross_entropy(logits, y)
        elif cell == 'dense_mask':
            # MB: nur EINES der 4 Label-Frames je Step, zyklisch k mod 4
            zt = zs[:, DELAY:, :]
            j = k % zt.shape[1]
            zsel = zt[:, j, :]
            ysel = yf[:, j]
            logits = zsel @ head_w.T + head_b
            loss = torch.nn.functional.cross_entropy(logits, ysel)
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
    xT, yT, yfT = make_task(seed, cell_task, 1024, test_gen)
    with torch.no_grad():
        zs = rollout_states(W, b, Uw, xT)
        if cell_task == 'sparse':
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
    out_path = 'results/t33_signalstaerke_match_results.txt'
    ms_path = 'results/t33_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T33 — SIGNALSTRAEKE-MATCH (T32-Konfund: MB matcht Terme/Step + Varianz + Total; S4X testet zu wenig Training)')
    L.append('HYPOTHESE (T32): Rueckmeldungs-Dichte, nicht Gedaechtnis-Forderung, ist der Traeger des Schalters.')
    L.append('Korrektur ggue. erster Lesart: 4x Loss-Terme (nicht ~20x); Konfund ist die VARIANZ, nicht die Magnitude.')
    L.append('PASS: P_MB(1) >= 0.5 UND P_S4X(1) <= 0.15 | GEMISCHT-Redundanz / GEMISCHT-Expositions-Konfund (N57-Erg).')
    L.append('Stufen a {0,1}, Bloecke 1430..1509, 80 Runs, f64, MB 600 Steps / S4X 2400 Steps.')
    L.append('Verdikt generiert von tools/lab_kernel.py | Contract: contracts/t33.py')
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
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        measurements[name] = dict(
            pf=pf, colnorm=float(torch.tensor(cns).mean()), ndiv=ndiv)
        L.append('')
        L.append('Arm %s (cell=%s, task=%s, a=%d, steps=%d, seeds %d..%d): n_ok=%d/%d | '
                 'P_lernt=%.2f | median acc=%s | div=%d'
                 % (name, arm.get('cell'), arm.get('cell_task'),
                    int(arm.get('conj_a', 0)), int(arm.get('steps', STEPS)),
                    seeds[0], seeds[-1], len(ok), len(pvals), pf,
                    fmt(float(torch.tensor(ok).median())) if ok else 'nan',
                    ndiv))
        Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    pmb0 = measurements.get('DENSE_MASK_A0', {}).get('pf', float('nan'))
    pmb1 = measurements.get('DENSE_MASK_A1', {}).get('pf', float('nan'))
    ps40 = measurements.get('SPARSE4X_A0', {}).get('pf', float('nan'))
    ps41 = measurements.get('SPARSE4X_A1', {}).get('pf', float('nan'))
    sanity_ok = (pmb0 >= 0.5 and ps40 >= 0.5)
    if not sanity_ok:
        verdict_txt = 'UNENTSCHEIDEN — SANITY GERICHT'
    elif pmb1 >= 0.5 and ps41 <= 0.15:
        verdict_txt = 'PASS — DICHTE-TRAEGER (KONFUND GESCHLOSSEN)'
    elif pmb1 >= 0.5 and ps41 >= 0.5:
        verdict_txt = 'GEMISCHT — REDUNDANZ (beide Faktoren tragen; keins notwendig; registriert N57-Erg)'
    elif pmb1 <= 0.15 and ps41 >= 0.5:
        verdict_txt = 'GEMISCHT — PER-FRAME-EXPOSITIONS-KONFUND (KILL nur mit MB-2400; registriert N57-Erg)'
    else:
        verdict_txt = 'GEMISCHT — ZWISCHENWELT (registriert, N57-Erg)'
    L.append('')
    L.append('--- T33-KERNEL-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('P-Stufen: P_MB(0)=%.4f | P_MB(1)=%.4f | P_S4X(0)=%.4f | P_S4X(1)=%.4f'
             % (pmb0, pmb1, ps40, ps41))
    L.append('SANITY: P_MB(0) >= 0.5: %s | P_S4X(0) >= 0.5: %s'
             % (pmb0 >= 0.5, ps40 >= 0.5))
    L.append('Referenz (T32, nicht neu gefahren): P_SPARSE(1)=0.05 | P_DENSE(1)=1.00')
    L.append('L5-BAND: ALLE Arme ERSTFAHRT ohne Band (registriert, N57 — neue Optimierungs-Regime).')
    L.append('')
    L.append('VERDICT: %s' % verdict_txt)
    L.append('')
    L.append('--- N-SKELETT (Zahlen maschinell; Lesart handzuschreiben) ---')
    L.append('# Nachtrag 57 (2026-11-17) — AUTO-SKELETT')
    L.append('  DENSE_MASK_A0.pf = %s | DENSE_MASK_A1.pf = %s' % (fmt(pmb0), fmt(pmb1)))
    L.append('  SPARSE4X_A0.pf = %s | SPARSE4X_A1.pf = %s' % (fmt(ps40), fmt(ps41)))
    L.append('  LUECKENLISTE: KEINE — alle registrierten Groessen gemessen.')
    L.append('=== FERTIG ===')
    out = chr(10).join(L) + chr(10)
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(
        dict(measurements=measurements, verdict=verdict_txt), indent=1),
        encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse: %s | %s' % (out_path, ms_path))


if __name__ == '__main__':
    main()
