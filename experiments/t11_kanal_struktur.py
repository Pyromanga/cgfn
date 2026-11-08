#!/usr/bin/env python
'''T11 — KANAL-STRUKTUR-ZERLEGUNG: OVERLAP, TRAEGER, VORZEICHEN (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE: T10 (Nachtrag 11) — KILL-B. REF (zwei unabhaengige iid-Gauss-
Spalten in Koordinaten) lernt 0.70; MODRAND48 (iid Phasen, Kanaele getrennt),
KOORDKOH48 (DC+Nyquist, phasen-gemischt in Eigenbasis), KOH48 (kohaerent)
allesamt 0.00. Was bleibt als Unterschied: (a) KANAL-OVERLAP (bei REF teilen
beide Kanaele den ganzen R^48), (b) TRAEGER-DIMENSION je Kanal, (c) ENTROPIE/
FEINVERTEILUNG der Amplituden. T11 zerlegt genau das.

T11-THESE (OVERLAP-THESE, die eine These dieses Laufs):
  Kanal-Overlap ist der Schluessel zur Trainierbarkeit: Zwei Kanaele, die
  denselben Unterraum treiben (CHANOVLP), lernen — zwei Kanaele auf
  disjunkten Unterraumen (CHANSEP) nicht. Die Traeger-Dimension (24 statt
  48) ist dabei unschuldig.

INSTRUMENT:
  W = R diag(eigs) R^T, R aus T6-B60-Familie, PRO SEED FIX (DM-Konstanz,
  check <= 1e-9). Spektrum FIX. F=I, b = 0.1*randn FIX. Seeds 110..119.
  Energie pro Kanal konstant 0.5 (Spaltennorm von Uw je 0.5).
  ARME (4):
    REF: Uw = 0.5*randn(H,2)/norm — SANITY-Arm (T6/REF-Konvention:
      Spalten von 0.5*randn(H,2) haben Erwartungsnorm ~0.5*sqrt(48);
      zur exakten Energie-Konvention normiert jede Spalte auf 0.5).
      HINWEIS: REF in T9/T10 nutzte unnormierte 0.5*randn-Spalten.
      Energie-Konvention hier einheitlich normiert; REF-Lernbarkeit
      wurde mit beiden Konventionen beobachtet (T6 unnormiert,
      T11 normiert prueft Sanity erneut).
    CHANSEP: v0 = iid Gauss auf Koordinaten 0..23 (Null auf 24..47),
      v1 = iid Gauss auf Koordinaten 24..47 (Null auf 0..23),
      je normiert — iid Amplituden, aber KANAL-GETRENNT ohne Overlap.
    CHANOVLP: v0, v1 = zwei unabhaengige iid Gauss-Vektoren auf
      Koordinaten 0..23 (Null auf 24..47), je normiert — maximaler
      Overlap, Traeger-Dimension 24 pro Kanal.
    ENTROP: v0 = iid Vorzeichen (randn-Zeichen)/sqrt(H) auf ALLEN 48
      Koordinaten, v1 unabhaengig ebenso — deterministische Konstruktion
      mit maximaler Feinverteilung (Entropie-Knarp), voller Traeger,
      voller Overlap.
  TRAININGS-Mass: P_lernt = Fraktion(acc >= 0.99), Paritaet wie T3/T6/T9/T10
    (Delayed-XOR, T=24, RK2, tau=1.0, dt=0.1, substeps=5, Adam 600,
    Batch 32, lr 5e-3, 1024 Test-Samples, Test-Seed 777).
  RESERVOIR-Mass: MC_total wie T7/T9/T10.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF) >= 0.5, sonst UNENTSCHEIDEN, Serie-Post-Mortem,
    KEINE weitere Paritaets-Reparatur dieser Linie.
  PASS (OVERLAP-THESE): Sanity UND P_lernt(CHANOVLP) >= 0.5 UND
    P_lernt(CHANSEP) = 0.
  KILL-A (TRAEGERTHESE): Sanity UND P_lernt(CHANSEP) >= 0.5 (auch ohne
    Overlap lernbar — Geometrie des Overlaps ist nicht der Punkt;
    CHANOVLP/ENTROP berichtet als sekundaer).
  KILL-B (VORZEICHEN-THESE): Sanity UND P_lernt(ENTROP) >= 0.5 UND
    P_lernt(CHANSEP) = 0 UND P_lernt(CHANOVLP) = 0 — iid Vorzeichen
    genuegen: die FEINVERTEILUNG der Antriebsmatrix ist der Schluessel.
  KILL-C (NICHTS DAVON): Sanity UND CHANSEP, CHANOVLP, ENTROP alle
    P_lernt < 0.5 — die Kanal-Struktur-Zerlegung scheitert; verbleibende
    Variable: die Gauss-Amplituden-Verteilung (nur REF-artige
    Amplitudenspektren lernen). T12: Amplituden-Knarp.
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen in einem Arm,
    DM-Verletzung (rel. Streuung > 1e-9). Rerun nur bei Crash.
  SEKUNDAER: MC je Arm, median acc, DM, Laufzeit.
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.
'''
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
SEEDS = list(range(110, 120))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
ARMS = ['REF', 'CHANSEP', 'CHANOVLP', 'ENTROP']
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


def normcol(v):
    return v / v.norm() * 0.5


def make_Uw(arm, R, gen):
    half = H // 2
    if arm == 'REF':
        v0 = normcol(torch.randn(H, generator=gen))
        v1 = normcol(torch.randn(H, generator=gen))
    elif arm == 'CHANSEP':
        v0 = torch.zeros(H)
        v0[0:half] = torch.randn(half, generator=gen)
        v0 = normcol(v0)
        v1 = torch.zeros(H)
        v1[half:H] = torch.randn(half, generator=gen)
        v1 = normcol(v1)
    elif arm == 'CHANOVLP':
        v0 = torch.zeros(H)
        v0[0:half] = torch.randn(half, generator=gen)
        v0 = normcol(v0)
        v1 = torch.zeros(H)
        v1[0:half] = torch.randn(half, generator=gen)
        v1 = normcol(v1)
    elif arm == 'ENTROP':
        g0 = torch.randn(H, generator=gen)
        v0 = normcol(torch.sign(g0) / (2.0 * torch.sqrt(torch.tensor(float(H)))))
        v0 = torch.sign(g0) / torch.sqrt(torch.tensor(float(H))) * 0.5
        g1 = torch.randn(H, generator=gen)
        v1 = torch.sign(g1) / torch.sqrt(torch.tensor(float(H))) * 0.5
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
    out_path = 'results/t11_kanal_struktur_results.txt'
    csv_path = 'results/t11_kanal_struktur_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T11 — KANAL-STRUKTUR-ZERLEGUNG: OVERLAP, TRAEGER, VORZEICHEN')
    L.append('(registriert: Sanity P_lernt(REF)>=0.5; PASS wenn CHANOVLP>=0.5 und CHANSEP=0;')
    L.append('KILL-A wenn CHANSEP>=0.5; KILL-B wenn ENTROP>=0.5 und CHANSEP=CHANOVLP=0;')
    L.append('KILL-C wenn alle drei <0.5; Seeds 110..119, f64)')
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
    L.append('--- T11-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for arm in ARMS:
        L.append('%s: P_lernt=%.2f median acc=%.4f median MC=%.4f DM=%.4f'
                 % (arm, stats[arm]['pf'], stats[arm]['med'], stats[arm]['mc'], stats[arm]['dm']))

    dms_all = [stats[a]['dm'] for a in ARMS]
    dm_rel = (max(dms_all) - min(dms_all)) / max(dms_all)
    dm_const = dm_rel <= 1e-9
    sanity_ref = stats['REF']['pf'] >= 0.5
    sanity_div = all(stats[a]['ndiv'] <= 3 for a in ARMS)
    p_sep = stats['CHANSEP']['pf']
    p_ovlp = stats['CHANOVLP']['pf']
    p_ent = stats['ENTROP']['pf']
    L.append('DM-Konstanz: rel. Streuung=%.2e (%s)' % (dm_rel, 'OK' if dm_const else 'VERLETZT'))
    L.append('SANITY (REF lernt, P_lernt(REF)=%.2f >= 0.5?): %s' % (stats['REF']['pf'], 'JA' if sanity_ref else 'NEIN'))
    L.append('P_lernt(CHANSEP)=%.2f | P_lernt(CHANOVLP)=%.2f | P_lernt(ENTROP)=%.2f'
             % (p_sep, p_ovlp, p_ent))
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
    elif p_ovlp >= 0.5 and p_sep == 0.0:
        verdict = ('PASS — OVERLAP-THESE BEWIESEN: Zwei Kanaele im GEMEINSAMEN 24-dim '
                   'Unterraum (CHANOVLP) machen das Netz trainierbar, zwei Kanaele auf '
                   'disjunkten Unterraemen (CHANSEP) nicht. Der Kanal-Overlap ist der '
                   'Schluessel zur Trainierbarkeit; die Traeger-Dimension ist unschuldig. '
                   'Mechanismus-Platzhalter: XOR braucht Interferenz beider Kanaele auf '
                   'denselben Einheiten — disjunkte Kanaele koennen das Produkt nie '
                   'bilden.')
    elif p_sep >= 0.5:
        verdict = ('KILL-A — TRAEGER-/NICHT-OVERLAP-THESE: Auch ohne Kanal-Overlap '
                   '(CHANSEP) lernbar. Die Geometrie des Overlaps ist nicht der Punkt; '
                   'die Rest-Variable ist die Amplituden-/Vorzeichen-Verteilung. '
                   'CHANOVLP/ENTROP als sekundaer berichtet.')
    elif p_ent >= 0.5:
        verdict = ('KILL-B — VORZEICHEN-THESE: iid Vorzeichen auf allen 48 Koordinaten '
                   '(ENTROP) genuegen zum Lernen, obwohl CHANSEP und CHANOVLP (iid '
                   'Gauss, Traeger 24) versagen. Der Schluessel ist die FEINVERTEILUNG '
                   '(Entropie) der Antriebsmatrix: Maximal verteilte Amplituden machen '
                   'trainierbar, jede konzentrierte Amplitudenverteilung nicht.')
    else:
        verdict = ('KILL-C — NICHTS DAVON: Kanal-Overlap nein, Traeger-Reduktion nein, '
                   'iid Vorzeichen nein — die Kanal-Struktur-Zerlegung scheitert '
                   'vollstaendig. Verbleibende Variable: die Gauss-Amplituden-'
                   'VERTEILUNG selbst (kontinuierliche iid Amplituden vs. diskrete '
                   'Vorzeichen). T12 muesste den Amplituden-Knarp drehen '
                   '(z.B. Randomisierte Amplituden gegen festes Richtungsbild).')
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
