#!/usr/bin/env python
'''T12 — ENERGIE- UND KANAL-ZERLEGUNG (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE: T11 — UNENTSCHEIDEN durch Instrument-Bug (Nachtrag 12): Der
REF-Arm wurde stillschweigend auf Spaltennorm 0.5 normiert (bewaehrt:
unnormiert, ~3.46) und verhungerte. Nebenbefund: Die Antriebs-ENERGIE ist
ein massiver Knopf — und wirft die Frage auf, ob der T9b/T10-Kontrast
REF (lernt) vs. KOH-d-Familie (verhungert) teilweise ein ENERGIE-Konfund
war: REF hatte E~12 pro Kanal, die kohaerenten Arme E=0.25.

T12-THESE (P-ENERGIE, die eine These dieses Laufs):
  Die Verhungerung der kohaerenten Moden-Arme (T9/T9b/T10) war die
  ENERGIE, nicht die Richtung: Auf REF-Energie hochskaliert lernt auch
  die kohaerente 48-Moden-Konstruktion (KOH48E, P_lernt >= 0.5).

INSTRUMENT:
  W = R diag(eigs) R^T, R aus T6-B60-Familie, PRO SEED FIX (DM-Konstanz,
  check <= 1e-9). Spektrum FIX. F=I, b = 0.1*randn FIX. Seeds 120..129.
  Spaltennorm-Konventionen GENAGELT (siehe Registrierung im Docstring und
  XVII Nachtrag 12):
    REF: 0.5*randn(H,2) UNNORMIERT (bewaehrte Konvention, Norm ~3.46).
    AMP-NEG: iid-Spalten exakt auf Norm 0.5 (T11-REF-Replikation).
    KOH48E: T9-kohaerente 48-Moden-Konstruktion, Spalten exakt auf
      Norm 0.5*sqrt(48) skaliert.
    CHANSEP-E / CHANOVLP-E / ENTROP-E: T11-Zerlegungs-Arme, Spalten exakt
      auf Norm 0.5*sqrt(48) skaliert.
  TRAININGS-Mass: P_lernt = Fraktion(acc >= 0.99), Paritaet wie T3/T6/
  T9/T10/T11 (Delayed-XOR, T=24, RK2, tau=1.0, dt=0.1, substeps=5,
  Adam 600, Batch 32, lr 5e-3, 1024 Test-Samples, Test-Seed 777).
  RESERVOIR-Mass: MC_total wie T7/T9/T10/T11.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF) >= 0.5, sonst UNENTSCHEIDEN (Serie-Post-Mortem).
  AMP-NEG-NEGATIVKONTROLLE (berichtet, vor Primaer): erwartet 0; wenn
    >= 0.5 ist die Energie-Story widerlegt.
  PASS (P-ENERGIE): Sanity UND P_lernt(KOH48E) >= 0.5: Die Verhungerung
    war die Energie — Nachtraege 10/11 relativiert (Energie-Konfund),
    ehrliche Selbstkorrektur der Richtungs-Story.
  KILL-A (RICHTUNG HAELT): Sanity UND P_lernt(KOH48E) = 0: Die Richtung
    steht ENERGIE-BEREINIGT; Nachtrag 10/11 unangreifbar. CHAN-*-E-Arme
    sekundaer (T11-Frage bei richtiger Energie).
  KILL-B (MISCHUNG): 0 < P_lernt(KOH48E) < 0.5: beide Knarren tragen.
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen, DM-Verletzung.
  SEKUNDAER: P_lernt(CHANSEP-E/CHANOVLP-E/ENTROP-E), MC je Arm, DM, Zeit.
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
SEEDS = list(range(120, 130))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
ARMS = ['REF', 'AMPNEG', 'KOH48E', 'CHANSEP_E', 'CHANOVLP_E', 'ENTROP_E']
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


def scale_col(v, target_norm):
    return v / v.norm() * target_norm


def make_Uw(arm, R, gen):
    half = H // 2
    big = 0.5 * math.sqrt(H)   # ~3.464, REF-Erwartungsnorm
    small = 0.5
    if arm == 'REF':
        return 0.5 * torch.randn(H, 2, generator=gen)
    if arm == 'AMPNEG':
        v0 = scale_col(torch.randn(H, generator=gen), small)
        v1 = scale_col(torch.randn(H, generator=gen), small)
    elif arm == 'KOH48E':
        c0 = R[:, 0:half] @ torch.ones(half) / math.sqrt(half)
        c1 = R[:, half:H] @ torch.ones(half) / math.sqrt(half)
        v0 = c0 * big
        v1 = c1 * big
    elif arm == 'CHANSEP_E':
        u0 = torch.zeros(H)
        u0[0:half] = torch.randn(half, generator=gen)
        u1 = torch.zeros(H)
        u1[half:H] = torch.randn(half, generator=gen)
        v0 = scale_col(u0, big)
        v1 = scale_col(u1, big)
    elif arm == 'CHANOVLP_E':
        u0 = torch.zeros(H)
        u0[0:half] = torch.randn(half, generator=gen)
        u1 = torch.zeros(H)
        u1[0:half] = torch.randn(half, generator=gen)
        v0 = scale_col(u0, big)
        v1 = scale_col(u1, big)
    elif arm == 'ENTROP_E':
        g0 = torch.randn(H, generator=gen)
        g1 = torch.randn(H, generator=gen)
        v0 = scale_col(torch.sign(g0) / math.sqrt(float(H)), big)
        v1 = scale_col(torch.sign(g1) / math.sqrt(float(H)), big)
    else:
        raise ValueError(arm)
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
    out_path = 'results/t12_energie_richtung_results.txt'
    csv_path = 'results/t12_energie_richtung_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T12 — ENERGIE- UND KANAL-ZERLEGUNG')
    L.append('(registriert: Sanity P_lernt(REF)>=0.5; AMP-NEG erwartet 0;')
    L.append('PASS (P-ENERGIE) wenn P_lernt(KOH48E)>=0.5; KILL-A wenn KOH48E=0;')
    L.append('KILL-B wenn 0<KOH48E<0.5; Seeds 120..129, f64)')
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
    L.append('--- T12-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for arm in ARMS:
        L.append('%s: P_lernt=%.2f median acc=%.4f median MC=%.4f DM=%.4f'
                 % (arm, stats[arm]['pf'], stats[arm]['med'], stats[arm]['mc'], stats[arm]['dm']))

    dms_all = [stats[a]['dm'] for a in ARMS]
    dm_rel = (max(dms_all) - min(dms_all)) / max(dms_all)
    dm_const = dm_rel <= 1e-9
    sanity_ref = stats['REF']['pf'] >= 0.5
    sanity_div = all(stats[a]['ndiv'] <= 3 for a in ARMS)
    p_amp = stats['AMPNEG']['pf']
    p_koh = stats['KOH48E']['pf']
    L.append('DM-Konstanz: rel. Streuung=%.2e (%s)' % (dm_rel, 'OK' if dm_const else 'VERLETZT'))
    L.append('SANITY (REF lernt, P_lernt(REF)=%.2f >= 0.5?): %s' % (stats['REF']['pf'], 'JA' if sanity_ref else 'NEIN'))
    L.append('AMP-NEG-NEGativkontrolle (erwartet 0): P_lernt(AMPNEG)=%.2f' % p_amp)
    L.append('PRIMAER: P_lernt(KOH48E)=%.2f (bekannt: KOH48 bei kleiner Energie = 0.00 aus T9b/T10)'
             % p_koh)
    L.append('SEKUNDAER (T11-Frage bei REF-Energie): CHANSEP_E=%.2f CHANOVLP_E=%.2f ENTROP_E=%.2f'
             % (stats['CHANSEP_E']['pf'], stats['CHANOVLP_E']['pf'], stats['ENTROP_E']['pf']))
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
    elif p_koh >= 0.5:
        verdict = ('PASS — P-ENERGIE BEWIESEN: Die kohaerente 48-Moden-Konstruktion lernt '
                   'auf REF-Energie (KOH48E P_lernt >= 0.5). Die Verhungerung der '
                   'T9/T9b/T10-kohaerenten Arme war die ENERGIE, nicht die Richtung. '
                   'Ehrliche Selbstkorrektur: Der Richtungs-Kontrast von Nachtrag 10/11 '
                   'ist als Energie-Konfund entlarvt und relativiert. '
                   '(AMP-NEG berichtet: %.2f)' % p_amp)
    elif p_koh == 0.0:
        verdict = ('KILL-A — RICHTUNGS-STORY HAELT: Kohaerente Moden-Summen verhungern '
                   'AUCH auf REF-Energie (KOH48E = 0). Die Richtungs-Story von Nachtrag '
                   '10/11 steht ENERGIE-BEREINIGT. AMP-NEG berichtet: %.2f. '
                   'CHANSEP_E/CHANOVLP_E/ENTROP_E sekundaer: %.2f / %.2f / %.2f '
                   '(T11-Frage bei richtiger Energie).'
                   % (p_amp, stats['CHANSEP_E']['pf'], stats['CHANOVLP_E']['pf'],
                      stats['ENTROP_E']['pf']))
    else:
        verdict = ('KILL-B — MISCHUNG: KOH48E im Zwischenbereich (0 < P_lernt < 0.5): '
                   'Die Energie verschiebt die Lernbarkeit, die Richtung moduliert '
                   'zusaetzlich. Beide Knarren tragen; die Karte braucht eine '
                   '2D-Sicht (Energie x Richtung) als naechsten Stein.')
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
