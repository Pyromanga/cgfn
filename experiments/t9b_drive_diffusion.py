#!/usr/bin/env python
'''T9b — ANTRIEBS-KNOPF v2: MODEN-ABDECKUNG vs. KOORDINATEN-DIFFUSION (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE: T9 (Nachtrag 9) mass den Antriebs-Rang d mit
kohaerenten Moden-Summen — alle 5 Arme verhungerten (P_lernt 0),
waehrend dieselbe W-Familie mit T6s Koordinaten-Zufalls-Antrieb
bei 0.70-0.80 lernt. Diagnose: T9 vergass den T6-Referenzpunkt
im selben Lauf; der Knopf drehte an der Moden-ZAHL, aber die
Lernbarkeit haengt vermutlich an RICHTUNG/PHASE. Nebenbefund:
MC stieg strikt monoton mit d (Diffusion auch fuer Ridge gut).

T9b-THESE (die eine These dieses Laufs, Doppeldissoziation
v2 mit faehigem Instrument):
  Der selbe Lauf trennt: Ist Trainierbarkeit eine Funktion der
  ANGEDECKTEN MODEN-ZAHL (d-Familie, dann lernt grosses d) oder
  der RICHTUNG/PHASEN-STRUKTUR (dann bleibt jedes d tot und nur
  REF lernt)? Und leidet die Ridge-Ernte unter Konzentration?

INSTRUMENT:
  W = R diag(eigs) R^T, R aus T6-B60-Familie (per-Modus-Winkel
  U(0,0.6), DM ~13.6), PRO SEED FIX (DM-Konstanz per
  Konstruktion, check <= 1e-9). Spektrum FIX (wie T6/T9).
  F=I (ungerahmt), b = 0.1*randn FIX (Symmetrie-Brecher).
  ARME: REF mit EXAKTEM T6-Antrieb (Uw = 0.5*randn(H,2),
    erste Generator-Ziehung, dann b — T6-Reihenfolge) plus
    die T9-d-Familie UNVERAENDERT: d in {2,4,8,16,48},
    v_c = R[:,subset] @ ones/sqrt, ||v_c||=1, Amplitude 0.5
    pro Kanal, Kanaele mod-getrennt.
  Seeds 90..99 (frisch). f64.
  TRAININGS-Mass: P_lernt = Fraktion(acc >= 0.99) auf Paritaet
    (Delayed-XOR, Delay <=19 Frames, T=24, RK2, tau=1.0,
    dt=0.1, substeps=5, Adam 600 Schritte, Batch 32, lr 5e-3,
    1024 frische Test-Samples, Test-Seed 777) — identisch T3/T6/T9.
  RESERVOIR-Mass: MC_total (Ridge-Vorhersage von u_{t-k},
    Kanal 0, k in {1,2,4,8,16}, weisses Rauschen 0.5 auf BEIDEN
    Kanaelen, T=300, Washout 50, Ridge 1e-6, 5 Train- + 2 Test-
    Trajektorien) — identisch T7/T9. Auch fuer REF gemessen.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY (VOR jeder These-Auswertung): P_lernt(REF) >= 0.5.
    Wenn NICHT: UNENTSCHEIDEN, Serie-Post-Mortem, KEINE weitere
    Paritaets-Reparatur dieser Linie. (Lehre T3: Der lernende
    Arm muss lernen; hier ist REF der lernende Arm.)
  PRIMAER, PASS genau dann wenn:
    (P1) TRAINING: P_lernt(d=48) - P_lernt(d=2) >= 0.3.
    UND (P2) RESERVOIR: MC(d=2) >= 0.90 * MC(d=48).
  PASS -> Doppeldissoziation: Training braucht Moden-Abdeckung,
    die Ridge-Ernte kommt mit Konzentration zurecht. T8/T8b als
    Regime-Konflikt erklaert; Antriebs-Knopf = Struktur-Gesetz.
  KILL-A (Verhungerungs-Story falsch): P_lernt(d=2) >= 0.5.
  KILL-B (Diffusion universell): P1 JA, aber MC(d=2) <
    0.90*MC(d=48) — auch das Reservoir braucht Verteilung.
    (T9-Nebenbefund MC-Monotonie deutet hierher.)
  KILL-C (die Phasen-Wette, NEU): Sanity JA, aber P1 NEIN —
    Lernbarkeit ist KEINE Funktion der Moden-Abdeckung. Ist
    zusaetzlich P_lernt(d=48) = 0, verhungert JEDE kohaerente
    Eigenmoden-Summe: Es ist die RICHTUNG, nicht die ZAHL.
    Verhungerungs-These verfeinert (T8/T8b vollstaendig
    erklaert: der G8-F-Transfer schleppt die Richtung mit).
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen in einem
    Arm, oder DM-Verletzung. Dann Serie-Post-Mortem.
  SEKUNDAER: MC(REF) vs. MC(d)-Kurve, median acc, DM, Laufzeit.
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
SEEDS = list(range(90, 100))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
D_GRID = [2, 4, 8, 16, 48]
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


def make_Uw(R, d):
    half = d // 2
    v0 = R[:, 0:half] @ torch.ones(half) / math.sqrt(half)
    v1 = R[:, half:d] @ torch.ones(d - half) / math.sqrt(d - half)
    return 0.5 * torch.stack([v0, v1], dim=1)  # (H, 2), Spaltennorm 0.5


def make_Uw_ref(H, gen):
    # EXAKTES T6-Instrument: erste Ziehung Uw, zweite Ziehung b.
    return 0.5 * torch.randn(H, 2, generator=gen)


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
    if arm == 'REF':
        Uw = make_Uw_ref(H, gen)   # T6-Reihenfolge: Uw zuerst
        b = 0.1 * torch.randn(H, generator=gen)
    else:
        d = arm
        Uw = make_Uw(R, d)
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
    if arm == 'REF':
        gen = torch.Generator().manual_seed(seed)
        Uw = make_Uw_ref(H, gen)
        b = 0.1 * torch.randn(H, generator=gen)
    else:
        Uw = make_Uw(R, arm)
        b = 0.1 * torch.randn(H, generator=torch.Generator().manual_seed(seed))
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
    out_path = 'results/t9b_drive_diffusion_results.txt'
    csv_path = 'results/t9b_drive_diffusion_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T9b — ANTRIEBS-KNOPF v2: MODEN-ABDECKUNG vs. KOORDINATEN-DIFFUSION')
    L.append('(registriert: Sanity P_lernt(REF)>=0.5; PASS nur wenn zusaetzlich')
    L.append('P1 P_lernt(48)-P_lernt(2)>=0.3 UND P2 MC(2)>=0.90*MC(48); Seeds 90..99, f64)')
    L.append('=' * 118)

    arms = ['REF'] + D_GRID
    rows = []
    stats = {}
    for arm in arms:
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
                 % (str(arm), len(ok), len(pvals), pf, median(ok), stats[arm]['mc'],
                    stats[arm]['dm'], ndiv))
        L.append('  Parity-Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, pvals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%.6g\n' % (str(r['arm']), r['seed'], r['acc']))

    L.append('')
    L.append('--- T9b-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for arm in arms:
        L.append('%s: P_lernt=%.2f median acc=%.4f median MC=%.4f DM=%.4f'
                 % (str(arm), stats[arm]['pf'], stats[arm]['med'], stats[arm]['mc'], stats[arm]['dm']))

    dms_all = [stats[a]['dm'] for a in arms]
    dm_rel = (max(dms_all) - min(dms_all)) / max(dms_all)
    dm_const = dm_rel <= 1e-9
    sanity_ref = stats['REF']['pf'] >= 0.5
    p1 = (stats[48]['pf'] - stats[2]['pf']) >= 0.3
    p2 = stats[2]['mc'] >= 0.90 * stats[48]['mc']
    konz_trains = stats[2]['pf'] >= 0.5
    best_d_pf = max(stats[d]['pf'] for d in D_GRID)
    sanity_div = all(stats[a]['ndiv'] <= 3 for a in arms)
    L.append('DM-Konstanz: rel. Streuung=%.2e (%s)' % (dm_rel, 'OK' if dm_const else 'VERLETZT'))
    L.append('SANITY (REF lernt, P_lernt(REF)=%.2f >= 0.5?): %s' % (stats['REF']['pf'], 'JA' if sanity_ref else 'NEIN'))
    L.append('P1 (Training: P_lernt(48)-P_lernt(2)>=0.3): %s' % ('JA' if p1 else 'NEIN'))
    L.append('P2 (Reservoir: MC(2)>=0.90*MC(48)): %s' % ('JA' if p2 else 'NEIN'))
    L.append('KILL-A-Indikator: P_lernt(d=2)=%.2f (>=0.5? %s)'
             % (stats[2]['pf'], 'JA' if konz_trains else 'NEIN'))
    L.append('KILL-C-Indikator: P_lernt(d=48)=%.2f (bester d-Arm P_lernt=%.2f)'
             % (stats[48]['pf'], best_d_pf))
    L.append('Sanity: Divergenzen gesamt=%d (%s)'
             % (sum(stats[a]['ndiv'] for a in arms), 'JA' if sanity_div else 'NEIN'))

    if not dm_const:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — DM-Konstanz verletzt: Instrument-Bug. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_div:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — Divergenzen: Instrument instabil. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_ref:
        verdict = ('UNENTSCHEIDEN — SANITY GERICHT (P_lernt(REF) < 0.5): Selbst das '
                   'exakte T6-Instrument lernt nicht mehr. Serie-Post-Mortem; KEINE '
                   'weitere Paritaets-Reparatur dieser Linie.')
    elif p1 and p2:
        verdict = ('PASS — DOPPELDISSOZIATION BEWIESEN: Trainierbarkeit steigt mit der '
                   'Moden-Abdeckung (P1), die Ridge-Ernte kommt mit Konzentration '
                   'zurecht (P2). Der Antriebs-Knopf ist ein eigenes Struktur-Gesetz; '
                   'T8/T8b vollstaendig als Regime-Konflikt erklaert.')
    elif konz_trains:
        verdict = ('KILL-A — Die Verhungerungs-Story war falsch: Auch der konzentrierteste '
                   'Moden-Antrieb (d=2) trainiert (P_lernt >= 0.5). T8b starb an einem '
                   'anderen Defekt; Neuinterpretation von Nachtrag 8 noetig.')
    elif p1 and not p2:
        verdict = ('KILL-B — DIFFUSION UNIVERSELL: Training braucht Verteilung (P1), aber '
                   'auch das Reservoir leidet unter Konzentration (MC(2) < 0.90*MC(48)). '
                   'Einheitsgesetz statt Regime-Kontrast; der G8-Konzentrations-Vorteil '
                   'war ein Artefakt des Ein-Moden-Ein-Kanal-Designs. T9s MC-Monotonie '
                   ' bestaetigt.')
    else:
        verdict = ('KILL-C — DIE PHASEN-WETTE VERLOREN: Sanity (REF lernt) sitzt, aber '
                   'P1 riss: Trainierbarkeit ist KEINE Funktion der Moden-Abdeckung.')
        if stats[48]['pf'] == 0.0:
            verdict += (' P_lernt(d=48)=0: JEDE kohaerente Eigenmoden-Summe verhungert — '
                        'es ist die RICHTUNG (Koordinaten-Diffusion, Phasenmischung, '
                        'Kanal-Overlap), nicht die ZAHL der getroffenen Moden. Die '
                        'Verhungerungs-These von Nachtrag 8 ist verfeinert und '
                        'vollstaendig: Der G8-F-Transfer schleppt die Antriebs-')
            verdict += 'Richtung mit. T8/T8b endgueltig erklaert.'
        else:
            verdict += (' P_lernt(d=48)>0 aber ohne d-Trend: Die Moden-Zahl moduliert '
                        'hoechstens schwach; die eigentliche Variable ist die '
                        'Richtungs-/Phasenstruktur des Antriebs.')
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
