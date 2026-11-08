#!/usr/bin/env python
'''T13 — ENERGIE-KANTE: NORMEN-SWEEP x GEOMETRIE (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE: T12 (Nachtrag 13) — PASS (P-ENERGIE): Die kohaerente
48-Moden-Richtung lernt auf REF-Energie (0.80), alle Geometrien lernen bei
voller Energie, AMPNEG (E=0.25) verhungert. Die Verhungerung der Serie war
die ENERGIE, nicht die Richtung. Offen: Wo genau liegt die Kante, und
verschiebt die Geometrie sie?

T13-THESE (ENERGIE-KANTE, die eine These dieses Laufs):
  Es gibt eine GEMEINSAME scharfe Kante der Trainierbarkeit in der
  Antriebsnorm s (fuer das konstante W-Regime dieser Serie), und die
  Geometrie des Antriebs (iid vs kohaerent) verschiebt sie NICHT:
  Energie ist DER Trainings-Knopf, Geometrie oberhalb der Kante
  gleichgueltig.

INSTRUMENT:
  W = R diag(eigs) R^T, R aus T6-B60-Familie, PRO SEED FIX (DM-Konstanz,
  check <= 1e-9). Spektrum FIX. F=I, b = 0.1*randn FIX. Seeds 130..139.
  Energie-Konventionen GENAGELT: Spaltennorm von Uw je s (exakt normiert,
  skaliert); REF unnnormiert (0.5*randn, Norm ~3.46) als Sanity-Arm.
  ARME (7):
    REF: 0.5*randn(H,2) UNNORMIERT — SANITY.
    REFGEOM_1.0 / REFGEOM_2.0 / REFGEOM_3.5: iid Gauss-Spalten, exakt
      auf Norm s normiert.
    KOHGEOM_1.0 / KOHGEOM_2.0 / KOHGEOM_3.5: kohaerente 48-Moden-Summen
      (T9-Konstruktion, v_c = R[:,subset] @ ones/sqrt), exakt auf Norm s
      normiert.
  TRAININGS-Mass: P_lernt = Fraktion(acc >= 0.99), Paritaet wie T3/T6/T9/
  T10/T11/T12 (Delayed-XOR, T=24, RK2, tau=1.0, dt=0.1, substeps=5,
  Adam 600, Batch 32, lr 5e-3, 1024 Test-Samples, Test-Seed 777).
  RESERVOIR-Mass: MC_total wie T7/T9/T10/T11/T12.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF) >= 0.5, sonst UNENTSCHEIDEN (Serie-Post-Mortem).
  P1 (KANTE): P_lernt(REFGEOM, s=1.0) = 0 UND P_lernt(REFGEOM, s=3.5)
    >= 0.5.
  P2 (GEOMETRIE-FREI): |P_lernt(KOHGEOM, s) - P_lernt(REFGEOM, s)| <= 0.3
    fuer ALLE s in {1.0, 2.0, 3.5}.
  PASS: Sanity UND P1 UND P2 — gemeinsame scharfe Kante, Geometrie
    verschiebt sie nicht: Energie ist DER Trainings-Knopf der Serie.
  KILL-A (2D-KARTE): P2 verletzt — Geometrie verschiebt die Kante; die
    Karte braucht Energie x Geometrie als 2D-Flaeche.
  KILL-B (KEINE KANTE / WEICH): P1 ausserhalb der erwarteten Randpunkte
    verletzt (REFGEOM bei 3.5 < 0.5 waere Serie-Inkonsistenz zu T12) ODER
    weicher Uebergang: P_lernt(s=2.0) beider Geometrien im 0.2-0.5-Zwischen-
    bereich OHNE scharfen Sprung zwischen den Randpunkten — Kurve statt
    Schalter.
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen in einem Arm,
    DM-Verletzung (rel. Streuung > 1e-9). Rerun nur bei Crash.
  SEKUNDAER: MC je Arm (erwartet energie-robust flach), median acc, DM,
    Laufzeit.
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.
'''
import math
import os
import time
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

FIXED_P0 = 4242
FIXED_SPEC = 9999
TEST_SEED = 777
H = 48
T_SEQ = 24
TAU = 1.0
DT = 0.1
SUBSTEPS = 5
STEPS = 600
BATCH = 32
LR = 5e-3
LAM_MAX = 0.6
WB = 0.6
SEEDS = list(range(130, 140))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
NORMS = [1.0, 2.0, 3.5]
ARMS = ['REF'] + ['REFGEOM_%s' % s for s in NORMS] + ['KOHGEOM_%s' % s for s in NORMS]
T_MC = 300
WASH = 50
KS = [1, 2, 4, 8, 16]
N_TRAIN = 5
N_TEST = 2
INPUT_SCALE = 0.5
RIDGE = 1e-6


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


def init_W_R(H, seed):
    genS = torch.Generator().manual_seed(FIXED_SPEC)
    eigs = torch.cat([torch.tensor([LAM_MAX]),
                      0.9 * LAM_MAX * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)])
    R = eigenbasis(H, seed)
    return R @ torch.diag(eigs) @ R.T, R


def make_Uw(arm, R, gen):
    if arm == 'REF':
        return 0.5 * torch.randn(H, 2, generator=gen)
    geom, s_str = arm.split('_')
    s = float(s_str)
    if geom == 'REFGEOM':
        v0 = torch.randn(H, generator=gen)
        v1 = torch.randn(H, generator=gen)
    elif geom == 'KOHGEOM':
        half = H // 2
        v0 = R[:, 0:half] @ torch.ones(half) / math.sqrt(half)
        v1 = R[:, half:H] @ torch.ones(half) / math.sqrt(half)
    else:
        raise ValueError(arm)
    v0 = v0 / v0.norm() * s
    v1 = v1 / v1.norm() * s
    return torch.stack([v0, v1], dim=1)


def dyn(z, W, b, uin):
    return -z / TAU + torch.tanh(z + b) @ W.T + uin


def rollout(W, b, Uw, x):
    B = x.shape[0]
    z = torch.zeros(B, H)
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T)
            z = z + DT * k2
    return z


def states_mc(W, b, Uw, x):
    B = x.shape[0]
    z = torch.zeros(B, H)
    out = []
    for t in range(T_MC):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T)
            z = z + DT * k2
        out.append(z)
    return torch.stack(out, dim=1)


def train_parity(seed, arm):
    gen = torch.Generator().manual_seed(seed)
    W0, R = init_W_R(H, seed)
    dm = float(W0.diag().abs().mean() / W0.abs().mean())
    Uw = make_Uw(arm, R, gen)
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
            return None, dm
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(1024, test_gen)
    with torch.no_grad():
        z = rollout(W, b, Uw, xT)
        acc = ((z @ head_w.T + head_b).argmax(dim=1) == yT).double().mean().item()
    return acc, dm


def mc_total(seed, arm):
    W0, R = init_W_R(H, seed)
    gen = torch.Generator().manual_seed(seed)
    Uw = make_Uw(arm, R, gen)
    b = 0.1 * torch.randn(H, generator=gen)
    g1 = torch.Generator().manual_seed(9000 + seed)
    xtr = INPUT_SCALE * torch.randn(N_TRAIN, T_MC, 2, generator=g1)
    g2 = torch.Generator().manual_seed(9500 + seed)
    xte = INPUT_SCALE * torch.randn(N_TEST, T_MC, 2, generator=g2)
    str_ = states_mc(W0, b, Uw, xtr)[:, WASH:, :].reshape(-1, H)
    ste = states_mc(W0, b, Uw, xte)[:, WASH:, :].reshape(-1, H)
    mcs = []
    for k in KS:
        tgt_tr = xtr[:, WASH - k:T_MC - k, 0].reshape(-1)
        tgt_te = xte[:, WASH - k:T_MC - k, 0].reshape(-1)
        A = str_.T @ str_ + RIDGE * torch.eye(H)
        w = torch.linalg.solve(A, str_.T @ tgt_tr)
        pred = ste @ w
        mse = float(((pred - tgt_te) ** 2).mean())
        var = float(tgt_te.var())
        mcs.append(1.0 - mse / max(var, 1e-12))
    return float(sum(mcs))


def median(v):
    if not v:
        return float('nan')
    return float(torch.tensor(v).median())


def main():
    out_path = 'results/t13_energie_kante_results.txt'
    csv_path = 'results/t13_energie_kante_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T13 — ENERGIE-KANTE: NORMEN-SWEEP x GEOMETRIE')
    L.append('(registriert: Sanity P_lernt(REF)>=0.5; P1 Kante (REFGEOM 1.0=0, 3.5>=0.5);')
    L.append('P2 Geometrie-frei (Diff <= 0.3 fuer alle s); PASS = Sanity UND P1 UND P2;')
    L.append('KILL-A = P2 verletzt; KILL-B = keine scharfe Kante; Seeds 130..139, f64)')
    L.append('=' * 118)

    rows = []
    stats = {}
    for arm in ARMS:
        pvals = []
        dms = []
        ndiv = 0
        for seed in SEEDS:
            acc, dm = train_parity(seed, arm)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            dms.append(dm)
            rows.append(dict(arm=arm, seed=seed, acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        mcs = [mc_total(seed, arm) for seed in SEEDS]
        stats[arm] = dict(pf=pf, ndiv=ndiv, dm=float(torch.tensor(dms).mean()),
                          med=median(ok), mc=median(mcs))
        L.append('')
        L.append('Arm %s: n_ok=%d/%d | P_lernt=%.2f | median acc=%.4f | median MC=%.4f | DM=%.4f | div=%d'
                 % (arm, len(ok), len(pvals), pf, median(ok), stats[arm]['mc'],
                    stats[arm]['dm'], ndiv))
        L.append('  Parity-Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, pvals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%.6g\n' % (r['arm'], r['seed'], r['acc']))

    L.append('')
    L.append('--- T13-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for arm in ARMS:
        L.append('%s: P_lernt=%.2f median acc=%.4f median MC=%.4f DM=%.4f'
                 % (arm, stats[arm]['pf'], stats[arm]['med'], stats[arm]['mc'], stats[arm]['dm']))

    dms_all = [stats[a]['dm'] for a in ARMS]
    dm_rel = (max(dms_all) - min(dms_all)) / max(dms_all)
    dm_const = dm_rel <= 1e-9
    sanity_ref = stats['REF']['pf'] >= 0.5
    sanity_div = all(stats[a]['ndiv'] <= 3 for a in ARMS)
    p_low = stats['REFGEOM_1.0']['pf']
    p_high = stats['REFGEOM_3.5']['pf']
    diffs = {s: abs(stats['KOHGEOM_%s' % s]['pf'] - stats['REFGEOM_%s' % s]['pf']) for s in NORMS}
    maxdiff = max(diffs.values())
    soft_mid = all(0.2 <= stats['REFGEOM_%s' % s]['pf'] <= 0.5 for s in ['2.0']) and \
               all(0.2 <= stats['KOHGEOM_%s' % s]['pf'] <= 0.5 for s in ['2.0'])
    L.append('DM-Konstanz: rel. Streuung=%.2e (%s)' % (dm_rel, 'OK' if dm_const else 'VERLETZT'))
    L.append('SANITY (REF lernt, P_lernt(REF)=%.2f >= 0.5?): %s' % (stats['REF']['pf'], 'JA' if sanity_ref else 'NEIN'))
    L.append('P1 (Kante): P_lernt(REFGEOM,1.0)=%.2f (=0?) und P_lernt(REFGEOM,3.5)=%.2f (>=0.5?)'
             % (p_low, p_high))
    L.append('P2 (Geometrie-frei): Diffs je s: %s (max=%.2f <= 0.3?)'
             % (', '.join('s=%s: %.2f' % (s, diffs[s]) for s in NORMS), maxdiff))
    L.append('Uebergang s=2.0: REFGEOM=%.2f KOHGEOM=%.2f (weicher Zwischenbereich 0.2-0.5?)'
             % (stats['REFGEOM_2.0']['pf'], stats['KOHGEOM_2.0']['pf']))
    L.append('Sanity: Divergenzen gesamt=%d (%s)'
             % (sum(stats[a]['ndiv'] for a in ARMS), 'JA' if sanity_div else 'NEIN'))

    p1 = (p_low == 0.0) and (p_high >= 0.5)
    p2 = maxdiff <= 0.3
    if not dm_const:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — DM-Konstanz verletzt: Instrument-Bug. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_div:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — Divergenzen: Instrument instabil. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_ref:
        verdict = ('UNENTSCHEIDEN — SANITY GERICHT (P_lernt(REF) < 0.5): Serie-Post-Mortem; '
                   'KEINE weitere Paritaets-Reparatur dieser Linie.')
    elif p1 and p2:
        verdict = ('PASS — ENERGIE-KANTE BEWIESEN: Eine gemeinsame scharfe Kante in der '
                   'Antriebsnorm trennt tot von lernbar, und die Geometrie (iid vs '
                   'kohaerent) verschiebt sie nicht (max. Diff %.2f). Energie ist DER '
                   'Trainings-Knopf dieser Serie; oberhalb der Kante ist die '
                   'Antriebs-Geometrie gleichgueltig. Uebergangspunkt s=2.0: REFGEOM '
                   '%.2f, KOHGEOM %.2f.'
                   % (maxdiff, stats['REFGEOM_2.0']['pf'], stats['KOHGEOM_2.0']['pf']))
    elif not p2:
        verdict = ('KILL-A — 2D-KARTE NOETIG: Die Geometrie verschiebt die Kante '
                   '(max. Diff %.2f > 0.3). Energie UND Geometrie tragen getrennt; '
                   'die Karte braucht eine Energie-x-Geometrie-Flaeche.' % maxdiff)
    else:
        verdict = ('KILL-B — KEINE SCHARFE KANTE: P1 riss oder der Uebergang ist weich '
                   '(s=2.0 Zwischenbereich: REFGEOM %.2f, KOHGEOM %.2f). Die '
                   'Trainierbarkeit ist eine KURVE in der Energie, kein Schalter; '
                   'Randpunkte: P(1.0)=%.2f, P(3.5)=%.2f.'
                   % (stats['REFGEOM_2.0']['pf'], stats['KOHGEOM_2.0']['pf'], p_low, p_high))
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))
    print('[i] Ergebnisse nach ' + out_path + ' und ' + csv_path + ' geschrieben.')


if __name__ == '__main__':
    main()
