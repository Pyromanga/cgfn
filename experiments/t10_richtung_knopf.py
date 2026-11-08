#!/usr/bin/env python
'''T10 — RICHTUNGS-ZERLEGUNG: PHASEN vs. BASIS (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE: T9b (Nachtrag 10) — KILL-C. REF (iid-Koordinaten-Antrieb,
Uw=0.5*randn(H,2)) lernt P_lernt 0.90; JEDE kohaerente Eigenmoden-Summe
verhungert (alle d inkl. 48). Trainierbarkeit haengt an der RICHTUNG,
nicht an der Zahl der angeregten Moden. MC haengt umgekehrt an der
Moden-Abdeckung und ist richtungs-robust.

T10-THESE (PHASEN-THESE, die eine These dieses Laufs):
  Der Killer ist die fehlende PHASENMISCHUNG des Antriebs in der
  Eigenbasis: Ein Antrieb mit zufaelligen Phasen auf vollen 48 Moden
  (MODRAND) macht das Netz trainierbar, obwohl er — wie KOH48 — nur
  Eigenmoden-Richtungen enthaelt und mod-getrennt ist.

INSTRUMENT:
  W = R diag(eigs) R^T, R aus T6-B60-Familie (per-Modus-Winkel U(0,0.6),
  DM ~13.6), PRO SEED FIX (DM-Konstanz per Konstruktion, check <= 1e-9).
  Spektrum FIX (wie T6/T9/T9b). F=I, b = 0.1*randn FIX. Seeds 100..109.
  ARME (4), Energie pro Kanal konstant 0.5 (Spaltennorm von Uw je 0.5):
    REF: Uw = 0.5*randn(H,2) — T6-Instrument, SANITY-Arm.
    KOH48: v_c = R[:,subset_c] @ ones/sqrt(24), Kanaele mod-getrennt
      (T9-Konstruktion d=48) — Negativ-Kontrolle.
    MODRAND48: v_c = R[:,subset_c] @ g_c, g_c = randn(24) (zufaellige
      Phasen), Kanaele mod-getrennt — PHASEN-TEST.
    KOORDKOH48: v0 = ones(H)/sqrt(H), v1 = alternierendes
      Vorzeichenmuster/sqrt(H) — kohaerent in Koordinaten, in der
      Eigenbasis phasen-gemischt (R^T v ist voll) — BASIS-TEST.
  TRAININGS-Mass: P_lernt = Fraktion(acc >= 0.99) auf Paritaet
    (Delayed-XOR, Delay <=19 Frames, T=24, RK2, tau=1.0, dt=0.1,
    substeps=5, Adam 600 Schritte, Batch 32, lr 5e-3, 1024 frische
    Test-Samples, Test-Seed 777) — identisch T3/T6/T9.
  RESERVOIR-Mass: MC_total wie T7/T9 (k in {1,2,4,8,16}, Kanal 0,
    weisses Rauschen 0.5 auf beiden Kanaelen, T=300, Washout 50,
    Ridge 1e-6, 5 Train- + 2 Test-Trajektorien).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF) >= 0.5. Sonst UNENTSCHEIDEN, Serie-Post-Mortem,
    KEINE weitere Paritaets-Reparatur dieser Linie.
  PASS (PHASEN-THESE): Sanity UND P_lernt(MODRAND48) >= 0.5.
    (KOORDKOH48-Befund ist sekundaer; erwartet ebenfalls lernend.)
  KILL-A (BASIS-THESE): Sanity UND P_lernt(KOORDKOH48) >= 0.5 ABER
    P_lernt(MODRAND48) = 0.
  KILL-B (ZERLEGUNG SCHEITERT): Sanity UND MODRAND48 UND KOORDKOH48
    beide P_lernt < 0.5 (nur REF-artige iid-Koordinaten-Antriebe lernen).
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen in einem Arm oder
    DM-Verletzung (rel. Streuung > 1e-9). Rerun nur bei Crash, Design
    unveraendert.
  SEKUNDAER: MC je Arm, median acc, DM, Laufzeit. MC-Kontrolle: alle
    48-Moden-Arme erwartet ~1.65-1.70; MC(MODRAND48) ≈ MC(REF) bei
    getrenntem Training-Outcome bestaerkt die Richtungs-Robustheit von MC.
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
SEEDS = list(range(100, 110))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
ARMS = ['REF', 'KOH48', 'MODRAND48', 'KOORDKOH48']
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
    half = H // 2
    if arm == 'KOH48':
        v0 = R[:, 0:half] @ torch.ones(half) / math.sqrt(half)
        v1 = R[:, half:H] @ torch.ones(half) / math.sqrt(half)
    elif arm == 'MODRAND48':
        g0 = torch.randn(half, generator=gen)
        g1 = torch.randn(half, generator=gen)
        v0 = R[:, 0:half] @ g0 / g0.norm()
        v1 = R[:, half:H] @ g1 / g1.norm()
    elif arm == 'KOORDKOH48':
        v0 = torch.ones(H) / math.sqrt(H)
        sign = torch.tensor([1.0 if (i % 2 == 0) else -1.0 for i in range(H)])
        v1 = sign / math.sqrt(H)
    else:
        raise ValueError(arm)
    return 0.5 * torch.stack([v0, v1], dim=1)


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
    out_path = 'results/t10_richtung_results.txt'
    csv_path = 'results/t10_richtung_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T10 — RICHTUNGS-ZERLEGUNG: PHASEN vs. BASIS')
    L.append('(registriert: Sanity P_lernt(REF)>=0.5; PASS wenn P_lernt(MODRAND48)>=0.5;')
    L.append('KILL-A wenn KOORDKOH48>=0.5 und MODRAND48=0; KILL-B wenn beide <0.5; Seeds 100..109, f64)')
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
    L.append('--- T10-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for arm in ARMS:
        L.append('%s: P_lernt=%.2f median acc=%.4f median MC=%.4f DM=%.4f'
                 % (arm, stats[arm]['pf'], stats[arm]['med'], stats[arm]['mc'], stats[arm]['dm']))

    dms_all = [stats[a]['dm'] for a in ARMS]
    dm_rel = (max(dms_all) - min(dms_all)) / max(dms_all)
    dm_const = dm_rel <= 1e-9
    sanity_ref = stats['REF']['pf'] >= 0.5
    sanity_div = all(stats[a]['ndiv'] <= 3 for a in ARMS)
    mr = stats['MODRAND48']['pf']
    kk = stats['KOORDKOH48']['pf']
    L.append('DM-Konstanz: rel. Streuung=%.2e (%s)' % (dm_rel, 'OK' if dm_const else 'VERLETZT'))
    L.append('SANITY (REF lernt, P_lernt(REF)=%.2f >= 0.5?): %s' % (stats['REF']['pf'], 'JA' if sanity_ref else 'NEIN'))
    L.append('P_lernt(MODRAND48)=%.2f | P_lernt(KOORDKOH48)=%.2f | P_lernt(KOH48)=%.2f'
             % (mr, kk, stats['KOH48']['pf']))
    L.append('Sanity: Divergenzen gesamt=%d (%s)'
             % (sum(stats[a]['ndiv'] for a in ARMS), 'JA' if sanity_div else 'NEIN'))

    if not dm_const:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — DM-Konstanz verletzt: Instrument-Bug. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_div:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — Divergenzen: Instrument instabil. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_ref:
        verdict = ('UNENTSCHEIDEN — SANITY GERICHT (P_lernt(REF) < 0.5): Serie-Post-Mortem; '
                   'KEINE weitere Paritaets-Reparatur dieser Linie.')
    elif mr >= 0.5:
        verdict = ('PASS — PHASEN-THESE BEWIESEN: Zufaellige Phasen auf vollen 48 Eigenmoden '
                   '(MODRAND48, mod-getrennt, Kanaele ohne Overlap) machen das Netz '
                   'trainierbar. Die kohaerente Eigenmoden-Richtung war der Killer. '
                   'Die Verhungerungs-These von Nachtrag 8/10 ist damit MECHANISTISCH: '
                   'nicht die Basis, nicht die Zahl — die PHASENSTRUKTUR des Antriebs.')
    elif kk >= 0.5:
        verdict = ('KILL-A — BASIS-THESE GEWINNT: Kohaerente Koordinaten-Richtung '
                   '(KOORDKOH48) lernt, aber phasen-gemischte Eigenmoden (MODRAND48) '
                   'nicht. Der Schluessel liegt im Bezug zur Koordinaten-Struktur '
                   '(Anker/tanh-Schleifen leben in Koordinaten), nicht in der '
                   'Phasenmischung an sich.')
    else:
        verdict = ('KILL-B — ZERLEGUNG SCHEITERT: Weder Phasenmischung in der Eigenbasis '
                   '(MODRAND48) noch kohaerente Koordinaten-Richtung (KOORDKOH48) '
                   'lernen. Nur iid-Koordinaten-Antriebe (REF-Art) machen das Netz '
                   'trainierbar. "Genuin diffus" bleibt undifferenziert — der naechste '
                   'Stein muss die REF-Struktur weiter zerlegen (z.B. Kanal-Overlap '
                   'oder Rang der Antriebsmatrix als Variable).')
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
