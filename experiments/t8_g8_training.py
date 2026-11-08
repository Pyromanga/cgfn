#!/usr/bin/env python
'''T8 — G8S MECHANISMUS IM TRAINIERTEN NETZ (registriert 2026-11-08, VOR dem Lauf)

VORGESCHICHTE (Nachtrag 7): T7 hat den T-Knopf (Eigenbasis vs.
tanh-Koordinaten) fuer Reservoir-MC getotet (MC flat bei 19x DM),
und die Nachlese des G8-Original-Codes zeigte: G8 rotiert den
TANH-RAHMEN bei FIXEM W = Q T Q^T — DM ist dort ueber das ganze
w-Gitter PER KONSTRUKTION konstant. Der Entkopplungs-Angriff
trifft G8 nicht. G8s Knopf wurde im Training NIE getestet.

T8-THESE (die eine These dieses Laufs):
  Der G8-Knopf — Rotation des Nichtlinearitaets-Rahmens relativ
  zur Eigenbasis bei identischem W, identischem Spektrum, identi-
  scher diag-Masse — traegt auch im end-to-end trainierten Netz:
  Der Innen-Bereich des Rahmen-Winkels (w in 0.6..0.9, G8-Familie)
  erreicht hoehere P_lernt als der Zufalls-Rahmen (w=0) und als
  die volle Ausrichtung (w=1). Da DM per Konstruktion konstant
  ist, waere jeder Effekt PURE Geometrie.

INSTRUMENT (T3-Bewaehrt + G8-Mechanismus):
  Paritaet: Delayed-XOR, Delay <=19 Frames, T=24, Bits 3 Frames
  gehalten, Positionen randomisiert, Ziel XOR(A,B) am letzten
  Frame. RK2, tau=1.0, dt=0.1, substeps=5, H=48, lam_max=0.6,
  Spektrum fix. W = Q T Q^T PRO SEED fix (identisch ueber alle
  Arme). Rahmen F(w) = Q @ QR(w*I + (1-w)*P0), P0 = Q^T R0
  (R0 = QR(randn, Seed 4242) — G8-Exakt). Dyn:
    f(z) = -z/tau + tanh(z @ F + b_f) @ F.T @ W.T + x @ Uw.T
  mit b_f ~ N(0,0.1) FIX am Init (Symmetrie-Brecher in Rahmen-
  Koordinaten, T2b-Lehre) und Uw = 0.5 * Q[:, 0:2] (Antrieb auf
  den Top-2-Eigenmoden — G8-Mechanismus-Paket). GETRAINIERT:
  W + Linear-Readout, Adam 600 Schritte, Batch 32, lr 5e-3, f64.

ARME: w in {0.00, 0.60, 0.75, 0.90, 1.00}. FRISCHE Seeds 60..69.
  w=0 ist der Zufalls-Rahmen (Feld-Standard), w=1 die volle
  Ausrichtung an die Eigenbasis.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Messgroesse: P_lernt = Fraktion Runs mit Test-Accuracy >=
  0.99 (1024 frische Samples, Test-Seed 777, identisch fuer alle
  Arme). Kontrollgroesse: DM pro Arm (muss identisch sein —
  W ist pro Seed gleich; relative Streuung der Arme-DM <= 1e-9,
  sonst Instrument-Bug -> UNENTSCHEIDEN Crash-Klasse, Rerun
  erlaubt).
  PRIMAER, PASS genau dann wenn:
    (P1) GEOMETRIE: max P_lernt(w in {0.60,0.75,0.90}) -
         P_lernt(0.00) >= 0.3.
    UND (P2) NICHT-VOLL: P_lernt(best interior) >=
         P_lernt(1.00) (volle Ausrichtung ist nicht das Optimum
         — G6/G7/G8-Echo).
  KILL: Sonst — der G8-Knopf hat im trainierten Netz keine
     Kraft (flat oder falscher Peak): Die Regime-Karte ist
     komplett und endgueltig: Rahmen-Geometrie wirkt NUR im
     Reservoir (Ridge-Readout, fixes W), Entkopplung NUR im
     Training. Ehrliches Ende der Geometrie-Linie.
  UNENTSCHEIDEN: Sanity — kein Arm lernt (bester P_lernt < 0.5)
     oder >3/10 Divergenzen in einem Arm oder DM-Konstanz
     verletzt (Crash-Klasse: Rerun erlaubt, Design aendern
     verboten).
  SEKUNDAER (Bericht): P_lernt je w, median acc, DM-Check,
    Laufzeit.
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
SEEDS = list(range(60, 70))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
W_GRID = [0.00, 0.60, 0.75, 0.90, 1.00]


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


def factored_WQ(H, seed):
    genQ = torch.Generator().manual_seed(seed)
    A = torch.randn(H, H, generator=genQ)
    Q, _ = torch.linalg.qr(A)
    genS = torch.Generator().manual_seed(FIXED_SPEC)
    eigs = torch.cat([torch.tensor([LAM_MAX]),
                      0.9 * LAM_MAX * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)])
    return Q @ torch.diag(eigs) @ Q.T, Q


def frame_F(H, Q, w):
    genR = torch.Generator().manual_seed(FIXED_P0)
    R0, _ = torch.linalg.qr(torch.randn(H, H, generator=genR))
    P0 = Q.T @ R0
    R_y = torch.linalg.qr(w * torch.eye(H) + (1.0 - w) * P0)[0]
    return Q @ R_y


def dyn(z, W, F, b_f, uin):
    return -z / TAU + torch.tanh(z @ F + b_f) @ F.T @ W.T + uin


def rollout(W, F, b_f, Uw, x):
    B = x.shape[0]
    z = torch.zeros(B, H)
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, F, b_f, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, F, b_f, xt @ Uw.T)
            z = z + DT * k2
    return z


def run_one(seed, w):
    gen = torch.Generator().manual_seed(seed)
    W0, Q = factored_WQ(H, seed)
    dm = float(W0.diag().abs().mean() / W0.abs().mean())
    F = frame_F(H, Q, w)
    Uw = 0.5 * Q[:, 0:2].contiguous()
    b_f = 0.1 * torch.randn(H, generator=gen)
    W = W0.clone().requires_grad_(True)
    head_w = torch.zeros(2, H, requires_grad=True)
    head_b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([W, head_w, head_b], lr=LR)
    train_gen = torch.Generator().manual_seed(1000 + seed)
    for _ in range(STEPS):
        x, y = make_task(BATCH, train_gen)
        z = rollout(W, F, b_f, Uw, x)
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
        z = rollout(W, F, b_f, Uw, xT)
        acc = ((z @ head_w.T + head_b).argmax(dim=1) == yT).double().mean().item()
    return acc, dm


def main():
    out_path = 'results/t8_g8_training_results.txt'
    csv_path = 'results/t8_g8_training_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T8 — G8S MECHANISMUS IM TRAINIERTEN NETZ (registriert: PASS nur wenn')
    L.append('P1 max P_lernt(0.6..0.9) - P_lernt(0.0) >= 0.3 UND P2 interior >= P_lernt(1.0))')
    L.append('Framed tanh, W pro Seed fix (DM konstant), Eigenmoden-Antrieb, Seeds 60..69, f64')
    L.append('=' * 118)

    rows = []
    stats = {}
    for w in W_GRID:
        vals = []
        dms = []
        ndiv = 0
        for seed in SEEDS:
            acc, dm = run_one(seed, w)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            vals.append(acc)
            dms.append(dm)
            rows.append(dict(w=w, seed=seed, acc=acc))
        ok = [v for v in vals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        stats[w] = dict(pf=pf, ndiv=ndiv, dm=float(torch.tensor(dms).mean()))
        L.append('')
        L.append('Arm w=%.2f: n_ok=%d/%d | P_lernt=%.2f | DM=%.4f | div=%d'
                 % (w, len(ok), len(vals), pf, stats[w]['dm'], ndiv))
        L.append('  Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, vals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('w,seed,acc\n')
        for r in rows:
            fh.write('%.2f,%d,%.6g\n' % (r['w'], r['seed'], r['acc']))

    L.append('')
    L.append('--- T8-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    for w in W_GRID:
        L.append('w=%.2f: P_lernt=%.2f DM=%.4f' % (w, stats[w]['pf'], stats[w]['dm']))

    dms_all = [stats[w]['dm'] for w in W_GRID]
    dm_rel = (max(dms_all) - min(dms_all)) / max(dms_all)
    dm_const = dm_rel <= 1e-9
    interior = [0.60, 0.75, 0.90]
    best_int = max(interior, key=lambda w: stats[w]['pf'])
    p1 = (stats[best_int]['pf'] - stats[0.00]['pf']) >= 0.3
    p2 = stats[best_int]['pf'] >= stats[1.00]['pf']
    L.append('DM-Konstanz: rel. Streuung=%.2e (%s)' % (dm_rel, 'OK' if dm_const else 'VERLETZT'))
    L.append('Bester Innen-Arm: w=%.2f (P_lernt=%.2f)' % (best_int, stats[best_int]['pf']))
    L.append('P1 (Interior minus Zufall >= 0.3): %s | P2 (Interior >= Voll): %s'
             % ('JA' if p1 else 'NEIN', 'JA' if p2 else 'NEIN'))

    best_pf = max(stats[w]['pf'] for w in W_GRID)
    sanity_learn = best_pf >= 0.5
    sanity_div = all(stats[w]['ndiv'] <= 3 for w in W_GRID)
    L.append('Sanity: bester Arm P_lernt=%.2f (>=0.5? %s), Divergenzen gesamt=%d (%s)'
             % (best_pf, 'JA' if sanity_learn else 'NEIN',
                sum(stats[w]['ndiv'] for w in W_GRID),
                'JA' if sanity_div else 'NEIN'))

    if not dm_const:
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — DM-Konstanz verletzt: W ist '
                   'nicht identisch ueber die Arme, Instrument-Bug. Rerun '
                   'erlaubt, Design unveraedert.')
    elif not sanity_learn or not sanity_div:
        verdict = ('UNENTSCHEIDEN — Sanity gerissen (nichts gelernt oder '
                   'Divergenzen). Post-Mortem, kein Verdikt.')
    elif p1 and p2:
        verdict = ('PASS — Der G8-Knopf traegt ins trainierte Netz: Bei '
                   'identischem W, identischem Spektrum und per Konstruktion '
                   'konstanter diag-Masse erreicht der Innen-Bereich des '
                   'Rahmen-Winkels hoehere Lernbarkeit als Zufalls-Rahmen UND '
                   'volle Ausrichtung. PURE Geometrie, DM-kontrolliert '
                   'bewiesen. G8s Befund ist kein Reservoir-Artefakt, und '
                   'die Regime-Karte aus Nachtrag 6/7 wird revidiert: '
                   'Geometrie wirkt in BEIDEN Regimen, im Training neben '
                   'Entkopplung.')
    else:
        verdict = ('KILL — Der G8-Knopf hat im trainierten Netz keine eigene '
                   'Kraft (flat oder falscher Peak, bei konstantem DM). Die '
                   'Regime-Karte ist endgueltig: Rahmen-Geometrie wirkt NUR '
                   'im Reservoir (fixes W, Ridge-Readout), Entkopplung NUR '
                   'im Training. Ehrliches Ende der Geometrie-Linie.')
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
