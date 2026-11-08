#!/usr/bin/env python
'''T9 — DER ANTRIEBS-KNOPF: DOPPELDISSOZIATION DER REGIME (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE: T8b entdeckte als Nebenprodukt die ANTRIEBS-
VERHUNGERUNG: Das G8-Paket (Antrieb konzentriert auf Top-
Eigenmoden) toetet das Training (0/50), obwohl dieselbe W-
Familie mit diffusem Antrieb bei P_lernt 0.80 lernt (T6) —
und obwohl Konzentration im Reservoir-Ridge-Regime ein
VORTEIL ist (G8: U=Q[:,0], +16..41% MC). Nachtrag 8
formulierte die These: Die Regime verlangen ENTGEGENGESETZTE
Antriebs-Geometrie. T9 macht das zum eigenen Stein.

T9-THESE (DOPPELDISSOZIATION, die eine These dieses Laufs):
  Der Antriebs-Rang d (Zahl der getroffenen Eigenmoden, Energie
  pro Kanal KONSTANT) trennt die Regime:
    TRAINING: Lernbarkeit steigt mit d (P_lernt(48) -
      P_lernt(2) >= 0.3) — Training braucht verteilten Antrieb.
    RESERVOIR-RIDGE: MC leidet NICHT unter Konzentration
      (MC(d=2) >= 0.90 * MC(d=48)) — die Ridge-Ernte kommt mit
      Konzentration zurecht (G8-Logik).
  Wenn BEIDES sitzt, ist die Doppeldissoziation bewiesen: derselbe
  Knopf, entgegengesetzte Anforderungen pro Regime.

INSTRUMENT:
  W = R diag(eigs) R^T, R aus der T6-B60-Familie (per-Modus-
  Winkel U(0,0.6), DM ~ 13.6), PRO SEED FIX ueber alle Arme
  (DM-Konstanz per Konstruktion, check <= 1e-9). Rahmen F=I
  (T-Standard, ungerahmt). b = 0.1*randn FIX (Symmetrie-Brecher).
  ANTRIEB: Kanal 0 verteilt auf die ersten d/2, Kanal 1 auf die
  naechsten d/2 Eigenmoden: v_c = R[:, Teilmenge] @ ones/sqrt,
  ||v_c|| = 1, Amplitude 0.5 pro Kanal, UNABHAENGIG von d.
  ARME: d in {2, 4, 8, 16, 48}. FRISCHE Seeds 80..89.
  TRAININGS-Mass: P_lernt = Fraktion(acc >= 0.99) auf Paritaet
    (Delayed-XOR, Delay <=19 Frames, T=24, RK2, tau=1.0,
    dt=0.1, substeps=5, Adam 600 Schritte, Batch 32, lr 5e-3,
    1024 frische Test-Samples, Test-Seed 777).
  RESERVOIR-Mass: MC_total = Summe R^2 der Ridge-Vorhersage
    von u_{t-k} (Kanal 0), k in {1,2,4,8,16}, Antrieb weisses
    Rauschen 0.5*randn auf BEIDEN Kanaelen, T=300, Washout 50,
    Ridge 1e-6, 5 Train- + 2 frische Test-Trajektorien.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  PRIMAER, PASS genau dann wenn:
    (P1) TRAINING: P_lernt(d=48) - P_lernt(d=2) >= 0.3.
    UND (P2) RESERVOIR: MC(d=2) >= 0.90 * MC(d=48).
  PASS -> Doppeldissoziation bewiesen: Der Antriebs-Knopf
     trennt die Regime; Training braucht Verteilung, Ridge-Ernte
     braucht sie nicht. T8/T8b vollstaendig erklaert als
     Regime-Konflikt, nicht als Rahmen-Effekt.
  KILL-A (Verhungerungs-Story falsch): P_lernt(d=2) >= 0.5
     (Konzentration trainiert auch) — dann starb T8b an etwas
     anderem; Neuinterpretation noetig.
  KILL-B (keine Dissoziation): P1 sitzt, aber MC(d=2) <
     0.90*MC(d=48) — dann braucht AUCH das Reservoir Verteilung:
     Einheitsgesetz "Diffusion immer gut" statt Kontrast.
  UNENTSCHEIDEN: Sanity — kein Trainings-Arm erreicht
     P_lernt >= 0.5 (auch d=48 lernt nicht, Instrument defekt
     gegenueber T6-Basis) oder >3/10 Divergenzen in einem Arm
     oder DM-Verletzung. EINE Reparatur erlaubt (neue Linie,
     frische Strikes).
  SEKUNDAER: volle Kurven P_lernt(d), MC(d), median acc,
    DM-Check, Laufzeit.
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
SEEDS = list(range(80, 90))
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


def train_parity(seed, d):
    gen = torch.Generator().manual_seed(seed)
    W0, R = init_W_R(H, seed)
    dm = float(W0.diag().abs().mean() / W0.abs().mean())
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


def mc_total(seed, d):
    W0, R = init_W_R(H, seed)
    Uw = make_Uw(R, d)
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
    out_path = 'results/t9_drive_regime_results.txt'
    csv_path = 'results/t9_drive_regime_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T9 — ANTRIEBS-KNOPF: DOPPELDISSOZIATION (registriert: PASS nur wenn')
    L.append('P1 P_lernt(48)-P_lernt(2)>=0.3 UND P2 MC(2)>=0.90*MC(48); Seeds 80..89, f64)')
    L.append('W pro Seed fix (DM konstant), F=I, Energie pro Kanal konstant 0.5')
    L.append('=' * 118)

    rows = []
    stats = {}
    for d in D_GRID:
        pvals = []
        dms = []
        ndiv = 0
        for seed in SEEDS:
            acc, dm = train_parity(seed, d)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            dms.append(dm)
            rows.append(dict(d=d, seed=seed, acc=acc))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        mcs = [mc_total(seed, d) for seed in SEEDS]
        stats[d] = dict(pf=pf, ndiv=ndiv, dm=float(torch.tensor(dms).mean()),
                        med=median(ok), mc=median(mcs))
        L.append('')
        L.append('Arm d=%d: n_ok=%d/%d | P_lernt=%.2f | median acc=%.4f | median MC=%.4f | DM=%.4f | div=%d'
                 % (d, len(ok), len(pvals), pf, median(ok), stats[d]['mc'],
                    stats[d]['dm'], ndiv))
        L.append('  Parity-Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, pvals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('d,seed,acc\n')
        for r in rows:
            fh.write('%d,%d,%.6g\n' % (r['d'], r['seed'], r['acc']))

    L.append('')
    L.append('--- T9-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for d in D_GRID:
        L.append('d=%d: P_lernt=%.2f median acc=%.4f median MC=%.4f DM=%.4f'
                 % (d, stats[d]['pf'], stats[d]['med'], stats[d]['mc'], stats[d]['dm']))

    dms_all = [stats[d]['dm'] for d in D_GRID]
    dm_rel = (max(dms_all) - min(dms_all)) / max(dms_all)
    dm_const = dm_rel <= 1e-9
    p1 = (stats[48]['pf'] - stats[2]['pf']) >= 0.3
    p2 = stats[2]['mc'] >= 0.90 * stats[48]['mc']
    konz_trains = stats[2]['pf'] >= 0.5
    L.append('DM-Konstanz: rel. Streuung=%.2e (%s)' % (dm_rel, 'OK' if dm_const else 'VERLETZT'))
    L.append('P1 (Training: P_lernt(48)-P_lernt(2)>=0.3): %s' % ('JA' if p1 else 'NEIN'))
    L.append('P2 (Reservoir: MC(2)>=0.90*MC(48)): %s' % ('JA' if p2 else 'NEIN'))
    L.append('KILL-A-Indikator: P_lernt(d=2)=%.2f (>=0.5? %s)'
             % (stats[2]['pf'], 'JA' if konz_trains else 'NEIN'))

    best_pf = max(stats[d]['pf'] for d in D_GRID)
    sanity_div = all(stats[d]['ndiv'] <= 3 for d in D_GRID)
    L.append('Sanity: bester Trainings-Arm P_lernt=%.2f (>=0.5? %s), Divergenzen gesamt=%d (%s)'
             % (best_pf, 'JA' if best_pf >= 0.5 else 'NEIN',
                sum(stats[d]['ndiv'] for d in D_GRID),
                'JA' if sanity_div else 'NEIN'))

    if not dm_const:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — DM-Konstanz verletzt: Instrument-Bug. '
                   'Rerun erlaubt, Design unveraendert.')
    elif best_pf < 0.5 or not sanity_div:
        verdict = ('UNENTSCHEIDEN — Sanity gerissen: Auch d=48 lernt nicht '
                   '(Instrument defekt gegenueber T6-Basis) oder Divergenzen. '
                   'EINE Reparatur erlaubt (neue Linie).')
    elif p1 and p2:
        verdict = ('PASS — DOPPELDISSOZIATION BEWIESEN: Der Antriebs-Rang trennt die '
                   'Regime — Training braucht verteilten Antrieb (P_lernt steigt mit '
                   'd), die Ridge-Ernte kommt mit Konzentration zurecht (MC '
                   'unangetastet). T8/T8b vollstaendig erklaert als Regime-Konflikt. '
                   'Der Antriebs-Knopf ist ein eigenes Struktur-Gesetz dieser Serie.')
    elif konz_trains:
        verdict = ('KILL-A — Die Verhungerungs-Story war falsch: Auch konzentrierter '
                   'Antrieb trainiert (P_lernt(d=2)>=0.5). T8b starb an einem anderen '
                   'Defekt; Neuinterpretation des Nachtrags 8 noetig. Ehrliche '
                   'Selbstkorrektur, kein Rueckzug aus der Offenheit.')
    else:
        verdict = ('KILL-B — KEINE DISSOZiation: Training braucht Verteilung (P1), '
                   'aber auch das Reservoir leidet unter Konzentration (MC(2) < '
                   '0.90*MC(48)). Einheitsgesetz: Diffusion ist universell gut. '
                   'Der Regime-Kontrast von Nachtrag 8 wird auf einen Nenner '
                   'gebracht: Die G8-Antriebs-Konzentration war nie ein Vorteil.')
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
