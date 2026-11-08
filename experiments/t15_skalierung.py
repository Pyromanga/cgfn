#!/usr/bin/env python
'''T15 — KANTEN-SKALIERUNGSGESETZ: LIEGT DIE KANTE BEI 0.5*sqrt(H)? (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE: T13 lokalisierte die Energie-Kante bei H=48 zwischen Norm
2.0 und 3.5. Und 0.5*sqrt(48) = 3.46 — exakt die REF-Konvention. Verdacht:
Die Kante folgt dem Skalen-Gesetz ||v_c||-Kante = 0.5*sqrt(H), d.h. die
bewaehrte Konvention sitzt auf der Kante (erklaert die REF-Lotterei
0.50-1.00 der Serie). T15 prueft das Gesetz an zwei neuen H.

T15-THESE (P-SQRT, die eine These dieses Laufs):
  Die Trainierbarkeits-Kante liegt bei der Antriebsnorm 0.5*sqrt(H):
  Bei H=24 zwischen 2.0 (tot) und 3.5 (lernbar); bei H=96 zwischen
  3.5 (tot) und 4.9 (lernbar).

INSTRUMENT:
  W = R diag(eigs) R^T, T6-B60-Familie-Konstruktion bei jeweiligem H
  (per-Modus-Winkel U(0,0.6), Spektrum FIX pro H), PRO SEED INIT,
  dann trainierbar (Adam auf W + Readout) wie immer. b = 0.1*randn FIX.
  Seeds 150..159. Energie-Konvention: Spaltennorm exakt s (normiert);
  REF unnnormiert 0.5*randn(H,2).
  ARME (8):
    REF24 / REF96: SANITY je H (unnnormierte Konvention).
    R24_2.0 / R24_3.5 / R24_4.9: H=24, REFGEOM exakt normiert.
    R96_2.0 / R96_3.5 / R96_4.9: H=96, REFGEOM exakt normiert.
  TRAININGS-Mass: P_lernt = Fraktion(acc >= 0.99), Paritaet wie T3/T6/
  T9/T10/T11/T12/T13/T14 (Delayed-XOR, T=24, RK2, tau=1.0, dt=0.1,
  substeps=5, Adam 600, Batch 32, lr 5e-3, 1024 Test-Samples, Test-Seed
  777). MC entfaellt (H-abhaengig nicht vergleichbar); DM wird je H
  berichtet (sekundaer, Konstanz innerhalb H per Konstruktion).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF24) >= 0.5 UND P_lernt(REF96) >= 0.5, sonst
    UNENTSCHEIDEN (Serie-Post-Mortem).
  PASS (P-SQRT): P_lernt(R24_2.0) = 0 UND P_lernt(R24_3.5) >= 0.5
    UND P_lernt(R96_3.5) = 0 UND P_lernt(R96_4.9) >= 0.5.
  KILL-A (ABSOLUTE KANTE): P_lernt(R96_3.5) >= 0.5.
  KILL-B (H=24-KANTE HOEHER): P_lernt(R24_3.5) = 0 UND
    P_lernt(R24_4.9) >= 0.5.
  SONST: GEMISCHT — Bericht, 2D-Karten-Kante (H x s).
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen in einem Arm,
    DM-Verletzung INNERHALB eines H. Rerun nur bei Crash.
  SEKUNDAER: median acc, DM je H, Laufzeit.
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.
'''
import math
import os
import time
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

# FIX 2026-11-08 (Run-1-Readout-Bug, siehe XVII Nachtrag 16): REF-Arme
# werden in stats getrennt verschluesselt (REF24/REF96). Run-1 schloss
# beide unter 'REF' zusammen, REF96 ueberschrieb REF24, der DM-Konstanz-
# Check las falsche DM-Werte und meldete eine Verletzung (Crash-Verdikt).
# Design, Masse, Registrierung und VERDICT-Logik UNVERAENDERT.

FIXED_P0 = 4242
TEST_SEED = 777
T_SEQ = 24
TAU = 1.0
DT = 0.1
SUBSTEPS = 5
STEPS = 600
BATCH = 32
LR = 5e-3
LAM_MAX = 0.6
WB = 0.6
SEEDS = list(range(150, 160))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
NORMS = [2.0, 3.5, 4.9]
ARMS = [('REF', 24), ('REF', 96)] + [('R24_%s' % s, 24) for s in NORMS] + [('R96_%s' % s, 96) for s in NORMS]


def skey(arm, H):
    # REF existiert zweimal (H=24, H=96) — getrennt verschluesseln.
    return 'REF%d' % H if arm == 'REF' else arm


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
    dm = float(W.diag().abs().mean() / W.abs().mean())
    return W, R, dm


def make_Uw(arm, H, gen):
    if arm == 'REF':
        return 0.5 * torch.randn(H, 2, generator=gen)
    s = float(arm.split('_')[1])
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
    W0, R, dm = init_W(H, seed)
    Uw = make_Uw(arm, H, gen)
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


def median(v):
    if not v:
        return float('nan')
    return float(torch.tensor(v).median())


def main():
    out_path = 'results/t15_skalierung_results.txt'
    csv_path = 'results/t15_skalierung_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T15 — KANTEN-SKALIERUNGSGESETZ: LIEGT DIE KANTE BEI 0.5*sqrt(H)?')
    L.append('(registriert: Sanity REF24>=0.5 UND REF96>=0.5; PASS (P-SQRT) wenn R24_2.0=0, R24_3.5>=0.5,')
    L.append('R96_3.5=0, R96_4.9>=0.5; KILL-A wenn R96_3.5>=0.5; KILL-B wenn R24_3.5=0 und R24_4.9>=0.5;')
    L.append('Seeds 150..159, f64)')
    L.append('=' * 118)

    rows = []
    stats = {}
    for arm, H in ARMS:
        pvals = []
        dms = []
        ndiv = 0
        for seed in SEEDS:
            acc, dm = train_parity(seed, arm, H)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            dms.append(dm)
            rows.append(dict(arm=arm, H=H, seed=seed, acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        st = dict(pf=pf, ndiv=ndiv, dm=float(torch.tensor(dms).mean()),
                  med=median(ok), H=H)
        stats[skey(arm, H)] = st
        L.append('')
        L.append('Arm %s (H=%d): n_ok=%d/%d | P_lernt=%.2f | median acc=%.4f | DM=%.4f | div=%d'
                 % (arm, H, len(ok), len(pvals), pf, median(ok), st['dm'], ndiv))
        L.append('  Parity-Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, pvals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,H,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%d,%.6g\n' % (r['arm'], r['H'], r['seed'], r['acc']))

    L.append('')
    L.append('--- T15-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for arm, H in ARMS:
        st = stats[skey(arm, H)]
        L.append('%s (H=%d): P_lernt=%.2f median acc=%.4f DM=%.4f'
                 % (arm, H, st['pf'], st['med'], st['dm']))

    p_ref24 = stats['REF24']['pf']
    p_ref96 = stats['REF96']['pf']
    p_r24_20 = stats['R24_2.0']['pf']
    p_r24_35 = stats['R24_3.5']['pf']
    p_r24_49 = stats['R24_4.9']['pf']
    p_r96_20 = stats['R96_2.0']['pf']
    p_r96_35 = stats['R96_3.5']['pf']
    p_r96_49 = stats['R96_4.9']['pf']
    sanity_ok = (p_ref24 >= 0.5) and (p_ref96 >= 0.5)
    sanity_div = all(stats[skey(a, H)]['ndiv'] <= 3 for a, H in ARMS)
    dm_ok = True
    for h in (24, 96):
        dms_h = [stats[skey(a, H)]['dm'] for a, H in ARMS if H == h]
        if max(dms_h) > 0 and (max(dms_h) - min(dms_h)) / max(dms_h) > 1e-9:
            dm_ok = False
    L.append('SANITY: REF24=%.2f (>=0.5? %s), REF96=%.2f (>=0.5? %s)'
             % (p_ref24, 'JA' if p_ref24 >= 0.5 else 'NEIN',
                p_ref96, 'JA' if p_ref96 >= 0.5 else 'NEIN'))
    L.append('H=24: s=2.0: %.2f, s=3.5: %.2f, s=4.9: %.2f' % (p_r24_20, p_r24_35, p_r24_49))
    L.append('H=96: s=2.0: %.2f, s=3.5: %.2f, s=4.9: %.2f' % (p_r96_20, p_r96_35, p_r96_49))
    L.append('DM-Konstanz innerhalb H: %s | Divergenzen gesamt=%d (%s)'
             % ('OK' if dm_ok else 'VERLETZT', sum(stats[skey(a, H)]['ndiv'] for a, H in ARMS),
                'JA' if sanity_div else 'NEIN'))

    if not dm_ok:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — DM-Konstanz innerhalb eines H verletzt: '
                   'Instrument-Bug. Rerun erlaubt, Design unveraendert.')
    elif not sanity_div:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — Divergenzen: Instrument instabil. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_ok:
        verdict = ('UNENTSCHEIDEN — SANITY GERICHT (REF24 oder REF96 < 0.5): '
                   'Serie-Post-Mortem; keine weitere Skalierungs-Reparatur.')
    elif p_r24_20 == 0.0 and p_r24_35 >= 0.5 and p_r96_35 == 0.0 and p_r96_49 >= 0.5:
        verdict = ('PASS — P-SQRT BEWIESEN: Die Kante folgt 0.5*sqrt(H) an beiden H '
                   '(H=24: 2.0 tot, 3.5 lernt; H=96: 3.5 tot, 4.9 lernt). Das '
                   'Skalen-Gesetz der Serie: Die Antriebs-Spaltennorm-Kante liegt bei '
                   '0.5*sqrt(H) — die bewaehrte REF-Konvention sitzt exakt AUF der '
                   'Kante (erklaert die REF-Lotterei 0.50-1.00). Design-Rezept: '
                   'Antriebsnorm >= 0.5*sqrt(H), sonst ist das Netz nicht schwach, '
                   'sondern funktional null (linear).')
    elif p_r96_35 >= 0.5:
        verdict = ('KILL-A — ABSOLUTE KANTE: R96_3.5 lernt — die Kante bleibt bei ~3.5 '
                   'auch fuer H=96. Kein sqrt(H)-Gesetz; die Kante ist eine absolute '
                   'Amplituden-Schwelle (pro-Einheit-Betrag ~ 3.5/sqrt(48) ≈ 0.5 '
                   'pro Einheit). Design-Rezept absolut: Antriebsnorm >= ~3.5, '
                   'unabhaengig von H.')
    elif p_r24_35 == 0.0 and p_r24_49 >= 0.5:
        verdict = ('KILL-B — H=24-KANTE HOEHER ALS VORHERGESAGT: Auch bei H=24 liegt '
                   'die Kante zwischen 3.5 und 4.9 — naeher an der absoluten '
                   'Interpretation bzw. kraeftiger als sqrt(H). Feiner Sweep noetig.')
    else:
        verdict = ('GEMISCHT — keine der registrierten Kanten-Konstellationen sitzt '
                   'vollstaendig. Bericht: H=24: 2.0: %.2f, 3.5: %.2f, 4.9: %.2f; '
                   'H=96: 2.0: %.2f, 3.5: %.2f, 4.9: %.2f. Die Kante braucht eine '
                   '2D-Karte (H x s) als naechsten Stein.'
                   % (p_r24_20, p_r24_35, p_r24_49, p_r96_20, p_r96_35, p_r96_49))
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
