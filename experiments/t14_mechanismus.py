#!/usr/bin/env python
'''T14 — MECHANISMUS-BEWIES-VERSUCH: NICHTLINEARITAET IST NOTWENDIG (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE: T13 (Nachtrag 14) — PASS: scharfe Energie-Kante zwischen
Antriebsnorm 2.0 und 3.5, geometrie-frei. Hypothese seit Nachtrag 13:
sub-Kante bleibt tanh im linearen Regime -> effektives lineares Netz ->
Paritaet mit linearem Readout nicht loesbar (T2b-Linie). T14 macht den
Mechanismus entscheidbar: Ersetze die AKTIVIERUNG, halte alles andere
fix (W trainierbar, Antriebs-Energie, b, Aufgabe).

T14-THESE (P-MECH, die eine These dieses Laufs):
  Die Energie-Kante ist der Eintritt ins nichtlineare Regime: Ein rein
  lineares RNN (Identity-Aktiviierung) lernt die Paritaet bei KEINER
  Antriebs-Energie (ID3.5 = 0, ID10 = 0), waehrend jede nichtlineare
  Aktiviierung oberhalb der Kante lernt (tanh: T3.5; ReLU: R3.5).

INSTRUMENT:
  W = R diag(eigs) R^T, T6-B60-Familie, PRO SEED FIX als INIT, dann
  trainierbar wie immer (Adam auf W + Readout). b = 0.1*randn FIX.
  Seeds 140..149. Energie-Konventionen wie T12/T13 (Spaltennorm exakt).
  ARME (6):
    REF: 0.5*randn(H,2) UNNORMIERT, tanh — SANITY.
    T3.5: REFGEOM Norm 3.5, tanh — Positiv-Kontrolle.
    T1.0: REFGEOM Norm 1.0, tanh — Negativ-Kontrolle.
    ID3.5: IDENTITY statt tanh, REFGEOM Norm 3.5 — KERN-ARM.
    ID10: IDENTITY statt tanh, REFGEOM Norm 10.
    R3.5: RELU statt tanh, REFGEOM Norm 3.5.
  TRAININGS-Mass: P_lernt = Fraktion(acc >= 0.99), Paritaet wie T3/T6/
  T9/T10/T11/T12/T13 (Delayed-XOR, T=24, RK2, tau=1.0, dt=0.1,
  substeps=5, Adam 600, Batch 32, lr 5e-3, 1024 Test-Samples, Test-Seed
  777). Achtung ID-Arme: lineare Dynamik kann divergieren; Divergenz-
  Zaehlung wie immer (nicht-finite Loss -> div, <= 3 erlaubt).
  SEKUNDAER: Nichtlinearitaets-Auslastung: Anteil der (B,H)-Zustands-
  eintraege mit |z+b| > 1 waehrend des Test-Rollouts, gemittelt; MC wie
  T7 (nur fuer tanh-Arme aussagekraeftig, fuer alle berichtet).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF) >= 0.5 UND P_lernt(T3.5) >= 0.5 UND
    P_lernt(T1.0) = 0. Sonst UNENTSCHEIDEN (Instrument inkonsistent zur
    T12/T13-Basis; Serie-Post-Mortem).
  PASS (P-MECH): P_lernt(ID3.5) = 0 UND P_lernt(ID10) = 0 UND
    P_lernt(R3.5) >= 0.5.
  KILL-A (IDENTITY LERNT): P_lernt(ID3.5) >= 0.3 ODER P_lernt(ID10) >=
    0.3 — Linearitaets-Story falsch.
  KILL-B (RELU TOT): ID-Arme 0 (wie erwartet), ABER P_lernt(R3.5) = 0 —
    die Form der Nichtlinearitaet traegt zusaetzlich.
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen in einem Arm,
    DM-Verletzung. Rerun nur bei Crash, Design unveraendert.
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
SEEDS = list(range(140, 150))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
ARMS = ['REF', 'T3.5', 'T1.0', 'ID3.5', 'ID10', 'R3.5']
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


def act_of(arm):
    if arm.startswith('ID'):
        return 'identity'
    if arm.startswith('R'):
        return 'relu'
    return 'tanh'


def make_Uw(arm, R, gen):
    if arm == 'REF':
        return 0.5 * torch.randn(H, 2, generator=gen)
    s = 10.0 if arm == 'ID10' else (3.5 if arm in ('T3.5', 'ID3.5', 'R3.5') else 1.0)
    v0 = torch.randn(H, generator=gen)
    v1 = torch.randn(H, generator=gen)
    v0 = v0 / v0.norm() * s
    v1 = v1 / v1.norm() * s
    return torch.stack([v0, v1], dim=1)


def dyn(z, W, b, uin, act):
    pre = z + b
    if act == 'identity':
        a = pre
    elif act == 'relu':
        a = torch.relu(pre)
    else:
        a = torch.tanh(pre)
    return -z / TAU + a @ W.T + uin


def rollout(W, b, Uw, x, act):
    B = x.shape[0]
    z = torch.zeros(B, H)
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T, act)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T, act)
            z = z + DT * k2
    return z


def rollout_with_sat(W, b, Uw, x, act):
    B = x.shape[0]
    z = torch.zeros(B, H)
    sat_num = 0.0
    sat_den = 0.0
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T, act)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T, act)
            z = z + DT * k2
        sat_num += float(((z + b).abs() > 1.0).double().sum())
        sat_den += float(z.numel())
    return z, (sat_num / max(sat_den, 1.0))


def states_mc(W, b, Uw, x, act):
    B = x.shape[0]
    z = torch.zeros(B, H)
    out = []
    for t in range(T_MC):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T, act)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T, act)
            z = z + DT * k2
        out.append(z)
    return torch.stack(out, dim=1)


def train_parity(seed, arm):
    act = act_of(arm)
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
        z = rollout(W, b, Uw, x, act)
        logits = z @ head_w.T + head_b
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            return None, dm, float('nan')
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(1024, test_gen)
    with torch.no_grad():
        z, sat = rollout_with_sat(W, b, Uw, xT, act)
        acc = ((z @ head_w.T + head_b).argmax(dim=1) == yT).double().mean().item()
    return acc, dm, sat


def mc_total(seed, arm):
    act = act_of(arm)
    W0, R = init_W_R(H, seed)
    gen = torch.Generator().manual_seed(seed)
    Uw = make_Uw(arm, R, gen)
    b = 0.1 * torch.randn(H, generator=gen)
    g1 = torch.Generator().manual_seed(9000 + seed)
    xtr = INPUT_SCALE * torch.randn(N_TRAIN, T_MC, 2, generator=g1)
    g2 = torch.Generator().manual_seed(9500 + seed)
    xte = INPUT_SCALE * torch.randn(N_TEST, T_MC, 2, generator=g2)
    str_ = states_mc(W0, b, Uw, xtr, act)[:, WASH:, :].reshape(-1, H)
    ste = states_mc(W0, b, Uw, xte, act)[:, WASH:, :].reshape(-1, H)
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
    out_path = 'results/t14_mechanismus_results.txt'
    csv_path = 'results/t14_mechanismus_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T14 — MECHANISMUS-BEWIES-VERSUCH: NICHTLINEARITAET NOTWENDIG')
    L.append('(registriert: Sanity REF>=0.5, T3.5>=0.5, T1.0=0; PASS wenn ID3.5=0, ID10=0, R3.5>=0.5;')
    L.append('KILL-A wenn ein ID-Arm >=0.3; KILL-B wenn ID ok aber R3.5=0; Seeds 140..149, f64)')
    L.append('=' * 118)

    rows = []
    stats = {}
    for arm in ARMS:
        pvals = []
        dms = []
        sats = []
        ndiv = 0
        for seed in SEEDS:
            acc, dm, sat = train_parity(seed, arm)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            dms.append(dm)
            if sat == sat:
                sats.append(sat)
            rows.append(dict(arm=arm, seed=seed, acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        mcs = [mc_total(seed, arm) for seed in SEEDS]
        stats[arm] = dict(pf=pf, ndiv=ndiv, dm=float(torch.tensor(dms).mean()),
                          med=median(ok), mc=median(mcs),
                          sat=median(sats))
        L.append('')
        L.append('Arm %s: n_ok=%d/%d | P_lernt=%.2f | median acc=%.4f | median MC=%.4f | DM=%.4f | NL-Auslastung=%.3f | div=%d'
                 % (arm, len(ok), len(pvals), pf, median(ok), stats[arm]['mc'],
                    stats[arm]['dm'], stats[arm]['sat'], ndiv))
        L.append('  Parity-Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, pvals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%.6g\n' % (r['arm'], r['seed'], r['acc']))

    L.append('')
    L.append('--- T14-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for arm in ARMS:
        L.append('%s: P_lernt=%.2f median acc=%.4f median MC=%.4f DM=%.4f NL=%.3f'
                 % (arm, stats[arm]['pf'], stats[arm]['med'], stats[arm]['mc'],
                    stats[arm]['dm'], stats[arm]['sat']))

    dms_all = [stats[a]['dm'] for a in ARMS]
    dm_rel = (max(dms_all) - min(dms_all)) / max(dms_all)
    dm_const = dm_rel <= 1e-9
    sanity_div = all(stats[a]['ndiv'] <= 3 for a in ARMS)
    s_ref = stats['REF']['pf'] >= 0.5
    s_pos = stats['T3.5']['pf'] >= 0.5
    s_neg = stats['T1.0']['pf'] == 0.0
    sanity_ok = s_ref and s_pos and s_neg
    p_id35 = stats['ID3.5']['pf']
    p_id10 = stats['ID10']['pf']
    p_r35 = stats['R3.5']['pf']
    L.append('DM-Konstanz: rel. Streuung=%.2e (%s)' % (dm_rel, 'OK' if dm_const else 'VERLETZT'))
    L.append('SANITY: REF=%.2f (>=0.5? %s), T3.5=%.2f (>=0.5? %s), T1.0=%.2f (=0? %s)'
             % (stats['REF']['pf'], 'JA' if s_ref else 'NEIN',
                stats['T3.5']['pf'], 'JA' if s_pos else 'NEIN',
                stats['T1.0']['pf'], 'JA' if s_neg else 'NEIN'))
    L.append('KERN: P_lernt(ID3.5)=%.2f, P_lernt(ID10)=%.2f, P_lernt(R3.5)=%.2f'
             % (p_id35, p_id10, p_r35))
    L.append('NL-Auslastung (sekundaer): T1.0=%.3f, T3.5=%.3f, REF=%.3f'
             % (stats['T1.0']['sat'], stats['T3.5']['sat'], stats['REF']['sat']))
    L.append('Sanity: Divergenzen gesamt=%d (%s)'
             % (sum(stats[a]['ndiv'] for a in ARMS), 'JA' if sanity_div else 'NEIN'))

    if not dm_const:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — DM-Konstanz verletzt: Instrument-Bug. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_div:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — Divergenzen: Instrument instabil. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_ok:
        verdict = ('UNENTSCHEIDEN — SANITY GERICHT (REF/T3.5/T1.0 inkonsistent zur '
                   'T12/T13-Basis): Serie-Post-Mortem.')
    elif p_id35 == 0.0 and p_id10 == 0.0 and p_r35 >= 0.5:
        verdict = ('PASS — P-MECH BEWIESEN: Das rein lineare RNN (Identity) lernt die '
                   'Paritaet bei KEINER Antriebs-Energie (ID3.5=0, ID10=0), RELU lernt '
                   '(R3.5 >= 0.5). Die Energie-Kante ist der Eintritt ins nichtlineare '
                   'Regime; die tanh-Form ist nicht spezifisch. Die Energie-Story der '
                   'Serie ist MECHANISTISCH geschlossen: sub-Kante = linear = Paritaet '
                   'unloesbar (T2b-Linie).')
    elif p_id35 >= 0.3 or p_id10 >= 0.3:
        verdict = ('KILL-A — LINEARITAETS-STORY FALSCH: Ein lineares RNN loest die '
                   'Paritaet (ID-Arm >= 0.3). Die Kante ist etwas anderes als der '
                   'Nichtlinearitaets-Eintritt (z.B. Signal-Rausch-Verhaeltnis im '
                   'Zustandsraum). Ehrliche Neuinterpretation noetig.')
    else:
        verdict = ('KILL-B — FORM DER NICHTLINEARITAET TRAEGT: ID-Arme tot wie erwartet, '
                   'ABER auch RELU versagt (R3.5 = 0). Nichtlinearitaet allein reicht '
                   'nicht; die Form (Saettigung/Befaelligung, tanh vs relu) traegt '
                   'zusaetzlich. Die Kante ist der Eintritt ins SAETTIGUNGS-faehige '
                   'Regime.')
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
