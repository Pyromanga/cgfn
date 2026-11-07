#!/usr/bin/env python
'''T3 — TRANSFER, TEIL 2: DER ANKER AUF PARITAET (registriert 2026-11-07, VOR dem Lauf)

VORGESCHICHTE (T2, zwei Strikes, UNENTSCHEIDEN, geschlossen):
  T2b zeigte exakt begruendbar: Mit b=0 und z0=0 ist die Dynamik
  UNGERADE in (x,z); der Aufgaben-Spiegel (a,b)->(-a,-b) dreht
  Logits um, das Label bleibt gleich — B0-Netze mit linearem
  Readout koennen Paritaet PROVABELL nicht lernen. Alle 20 B0-Runs
  bei Chance waren ein Theorem, kein Lotto. Konsequenz: Bias ist
  auf Paritaet kein Confound, sondern der Symmetrie-Brecher — er
  muss in ALLE Arme.
  Explorations-Signal aus T2b (UNREGISTRIERT, hier erstmals als
  These genagelt): BR_AL06 median 1.0000 vs BR_RND median 0.5215.

T3-THESE (die eine These dieses Laufs):
  In einem end-to-end trainierten Netz auf Paritaet (mit Bias als
  Symmetrie-Brecher in allen Armen) schlaegt der 0.6-rad
  Fehlwinkel-Anker (G8/G10-Familie) den Zufalls-Anker — das ist
  der erste registrierte TRANSFER des Anker-Befunds vom Reservoir
  ins echte Lernen.

INSTRUMENT (identisch zu T2b, nur die Rahmen lottern):
  XOR-Delay <=19 Frames, T=24, Bits 3 Frames gehalten, Positionen
  randomisiert; RK2, tau=1.0, dt=0.1, substeps=5; H=48, lam_max=0.6,
  Spektrum fix identisch; Adam, 600 Schritte, Batch 32, lr 5e-3;
  b ~ N(0,0.1) fix am Init in BEIDEN Armen; W+Readout trainiert; f64.

ARME: RND (R = QR(randn(seed))) vs AL06 (per-Modus
  theta_i ~ U(0,0.6), M = diag(cos th)+diag(sin th)@P0).
  FRISCHE Seeds 10..19 (kein Seed-Recycling aus T2b).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Messgroesse: finale Test-Accuracy (1024 frische Samples,
  fixer Test-Seed 777, identisch fuer beide Arme).
  PRIMAER, PASS genau dann wenn:
    (P1) EFFEKT: median(acc AL06) - median(acc RND) >= 0.15.
    UND (P2) PERFEKT: Fraktion perfekter Runs (acc >= 0.99) in
         AL06 >= 0.5 UND >= 2x Fraktion in RND.
  KILL: Sonst — der Anker-Befund (G8) transferiert NICHT ins
     echte Lernen; die Transfer-Linie ist ehrlich beendet.
  UNENTSCHEIDEN: Sanity — Gesamt-Median < 0.60 (Aufgabe wider
     Erwarten nicht gelernt) oder >3/10 Divergenzen in einem Arm.
     Dann Design-Post-Mortem, aber KEIN neuer Paritaets-Versuch
     (drei Versuche auf derselben Aufgabe waeren eine
     Steinchen-Serie).
  SEKUNDAER (Bericht): Mediane, Spreads, Perfekt-Fraktionen,
    volle Seed-Tabelle, Laufzeit.
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
SEEDS = list(range(10, 20))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3


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


def frame_R(H, mode, seed):
    gen = torch.Generator().manual_seed(seed)
    genP = torch.Generator().manual_seed(FIXED_P0)
    P0, _ = torch.linalg.qr(torch.randn(H, H, generator=genP))
    if mode == 'RND':
        R, _ = torch.linalg.qr(torch.randn(H, H, generator=gen))
        return R
    if mode == 'AL06':
        th = 0.6 * torch.rand(H, generator=gen)
        M = torch.diag(torch.cos(th)) + torch.diag(torch.sin(th)) @ P0
        R, _ = torch.linalg.qr(M)
        return R
    raise ValueError(mode)


def init_W(H, frame_mode, seed):
    genS = torch.Generator().manual_seed(FIXED_SPEC)
    eigs = torch.cat([torch.tensor([LAM_MAX]),
                      0.9 * LAM_MAX * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)])
    R = frame_R(H, frame_mode, seed)
    return R @ torch.diag(eigs) @ R.T


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


def run_one(seed, frame_mode):
    gen = torch.Generator().manual_seed(seed)
    W0 = init_W(H, frame_mode, seed)
    Uw = 0.5 * torch.randn(H, 2, generator=gen)
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
            return None
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(1024, test_gen)
    with torch.no_grad():
        z = rollout(W, b, Uw, xT)
        acc = ((z @ head_w.T + head_b).argmax(dim=1) == yT).double().mean().item()
    return acc


def median(v):
    if not v:
        return float('nan')
    return float(torch.tensor(v).median())


def std(v):
    if len(v) < 2:
        return float('nan')
    return float(torch.tensor(v).std())


def main():
    out_path = 'results/t3_transfer_parity_results.txt'
    csv_path = 'results/t3_transfer_parity_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T3 — ANKER AUF PARITAET (registriert: PASS nur wenn')
    L.append('P1 median(AL06)-median(RND)>=0.15 UND P2 Perfekt-Fraktion(AL06)>=0.5 und >=2x RND)')
    L.append('Bias in allen Armen (Symmetrie-Brecher, N(0,0.1)), frische Seeds 10..19, f64')
    L.append('=' * 118)

    rows = []
    accs = {}
    divs = {}
    for frame_mode in ('RND', 'AL06'):
        vals = []
        ndiv = 0
        for seed in SEEDS:
            acc = run_one(seed, frame_mode)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            vals.append(acc)
            rows.append(dict(arm=frame_mode, seed=seed, acc=acc))
        accs[frame_mode] = vals
        divs[frame_mode] = ndiv
        ok = [v for v in vals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        L.append('')
        L.append('Arm %s: n_ok=%d/%d | median=%.4f | std=%.4f | Perfekt-Fraktion=%.2f'
                 % (frame_mode, len(ok), len(vals), median(ok), std(ok), pf))
        L.append('  Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, vals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%.6g\n' % (r['arm'], r['seed'], r['acc']))

    L.append('')
    L.append('--- T3-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    okR = [v for v in accs['RND'] if v == v]
    okA = [v for v in accs['AL06'] if v == v]
    medR, medA = median(okR), median(okA)
    pfR = sum(1 for v in okR if v >= 0.99) / len(okR) if okR else 0.0
    pfA = sum(1 for v in okA if v >= 0.99) / len(okA) if okA else 0.0
    L.append('median(RND)=%.4f median(AL06)=%.4f Effekt=%.4f' % (medR, medA, medA - medR))
    L.append('Perfekt-Fraktion RND=%.2f AL06=%.2f' % (pfR, pfA))

    p1 = (medA - medR) >= 0.15
    p2 = (pfA >= 0.5) and (pfA >= 2.0 * max(pfR, 1e-9))
    L.append('P1 (Effekt >=0.15): %s | P2 (Perfekt>=0.5 und 2x RND): %s'
             % ('JA' if p1 else 'NEIN', 'JA' if p2 else 'NEIN'))

    total_div = sum(divs.values())
    all_med = float(torch.tensor([medR, medA]).median())
    sanity_learn = all_med >= 0.60
    sanity_div = all(divs[k] <= 3 for k in ('RND', 'AL06'))
    L.append('Sanity: Median-basiert %.4f (>=0.60? %s), Divergenzen=%d (%s)'
             % (all_med, 'JA' if sanity_learn else 'NEIN', total_div,
                'JA' if sanity_div else 'NEIN'))

    if not sanity_learn or not sanity_div:
        verdict = ('UNENTSCHEIDEN — Sanity gerissen (nicht gelernt oder '
                   'Divergenzen). Design-Post-Mortem, aber KEIN neuer '
                   'Paritaets-Versuch: drei Versuche auf derselben Aufgabe '
                   'waeren eine Steinchen-Serie.')
    elif p1 and p2:
        verdict = ('PASS — Der 0.6-rad-Anker schlaegt den Zufalls-Anker im '
                   'end-to-end trainierten Netz auf Paritaet: Effekt >=0.15 '
                   'Median-Differenz UND >=2x Perfekt-Fraktion. Der Anker-'
                   'Befund (G8/G10) transferiert vom Reservoir ins echte '
                   'Lernen. Erster bewiesener Transfer dieser Serie.')
    else:
        verdict = ('KILL — Der Anker-Befund transferiert NICHT ins echte '
                   'Lernen (Effekt oder Perfekt-Gate verfehlt). Die Transfer-'
                   'Linie ist ehrlich beendet; das T2b-Signal war ein '
                   'Seed-Lotto-Artefakt.')
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
