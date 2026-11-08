#!/usr/bin/env python
'''T8b — G8-KNOPF AN TRAINIERBARER BASIS (registriert 2026-11-08, VOR dem Lauf; einmalige Reparatur nach T8-UNENTSCHEIDEN, Praezedenz G4->G5)

T8-POST-MORTEM (ehrlich): T8 testete den Rahmen-Knopf an einer
W-Familie mit DM=2.55 — laut T6 ist deren Basis-Lernbarkeit
~0.10 (der RND-Punkt der T6-Kurve). 0/50 perfekte Runs sind
bei dieser Basis-Rate erwartbar (P(0/10)~0.35 je Arm). Der
Rahmen konnte sich nicht ausdrucken, weil die Basis nicht
trainierbar ist. Das ist kein These-Statement, sondern ein
Basis-Raten-Problem. DM-Konstanz funktionierte einwandfrei
(Streuung exakt 0).

T8b-REPARATUR (registriert, minimal): GLEICHE Frage, aber die
W-Familie ist die TRAINIERBARE aus T6: W = R diag(eigs) R^T
mit R aus der Budget-0.6-Konstruktion (per-Modus-Winkel
U(0,0.6)), DM ~ 13.5, T6-Basis-Lernbarkeit ~0.8. Pro Seed FIX
ueber alle Arme. Der tanh-Rahmen F(w) rotiert dagegen:
  F(w) = R @ QR(w*I + (1-w)*P0), P0 = R^T R0 (R0 = QR, Seed 4242)
  w=1: F = R = Eigenbasis (voll ausgerichtet)
  w=0: F ~ Zufalls-Rahmen
  Interior 0.6..0.9: G8-Peak-Bereich
Antrieb: Uw = 0.5 * R[:, 0:2] (Top-2-Eigenmoden). Sonst
Instrument identisch zu T8 (Paritaet, RK2, Adam, b_f ~ N(0,0.1)
fix, W+Readout trainiert, f64). FRISCHE Seeds 70..79.

T8b-THESE: Bei trainierbarer Basis (DM konstant per Konstruktion)
  moduliert der Rahmen-Winkel die Lernbarkeit UEBER die diag-
  Masse hinaus: Innen-Bereich > Zufalls-Rahmen UND > volle
  Ausrichtung. FLAT dagegen beweist: T6s Budget-Kurve war
  komplett DM-getragen, "DM ist alles".

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Messgroesse: P_lernt = Fraktion Runs mit Test-Accuracy >=
  0.99 (1024 frische Samples, Test-Seed 777, identisch fuer alle
  Arme). DM-Konstanz: rel. Streuung <= 1e-9 (sonst Crash-Klasse).
  PRIMAER, PASS genau dann wenn:
    (P1) GEOMETRIE: max P_lernt(w in {0.60,0.75,0.90}) -
         P_lernt(0.00) >= 0.3.
    UND (P2) NICHT-VOLL: P_lernt(best interior) >= P_lernt(1.00).
  KILL: Sonst (flat oder falscher Peak bei konstantem DM) —
     "DM ist alles": T6s Kurve war vollstaendig Entkopplungs-
     getragen, der Rahmen-Winkel hat keine eigene Kraft.
     Regime-Karte endgueltig, Geometrie-Linie ehrlich beendet.
  UNENTSCHEIDEN: bester Arm P_lernt < 0.5 (Basis wider Erwarten
     untrainierbar) oder >3/10 Divergenzen oder DM-Verletzung.
     Keine weitere Reparatur (ZWEI-STRIKES).
  SEKUNDAER: P_lernt je w, median acc, DM, Laufzeit.
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
SEEDS = list(range(70, 80))
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


def eigenbasis(H, seed):
    # T6-B60-Familie: per-Modus-Winkel U(0, 0.6) gegen die Achsen
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


def frame_F(H, R, w):
    genR = torch.Generator().manual_seed(FIXED_P0)
    R0, _ = torch.linalg.qr(torch.randn(H, H, generator=genR))
    P0 = R.T @ R0
    R_y = torch.linalg.qr(w * torch.eye(H) + (1.0 - w) * P0)[0]
    return R @ R_y


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
    W0, R = init_W_R(H, seed)
    dm = float(W0.diag().abs().mean() / W0.abs().mean())
    F = frame_F(H, R, w)
    Uw = 0.5 * R[:, 0:2].contiguous()
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
    out_path = 'results/t8b_g8_training_results.txt'
    csv_path = 'results/t8b_g8_training_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T8b — G8-KNOPF AN TRAINIERBARER BASIS (registriert: PASS nur wenn')
    L.append('P1 max P_lernt(0.6..0.9) - P_lernt(0.0) >= 0.3 UND P2 interior >= P_lernt(1.0))')
    L.append('W-Familie T6-B60 (DM~13.5, Basis-Lernbarkeit ~0.8), pro Seed fix; Seeds 70..79, f64')
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
    L.append('--- T8b-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
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
        verdict = ('UNENTSCHEIDEN (Crash-Klasse) — DM-Konstanz verletzt: Instrument-Bug. '
                   'Rerun erlaubt, Design unveraendert.')
    elif not sanity_learn or not sanity_div:
        verdict = ('UNENTSCHEIDEN — Sanity gerissen (Basis wider Erwarten untrainierbar '
                   'oder Divergenzen). ZWEI STRIKES: Keine weitere Reparatur. '
                   'Ehrliches Protokoll-Ergebnis: Der G8-Knopf ist im 40-EUR-'
                   'Trainingslabor nicht sauber feuerbar.')
    elif p1 and p2:
        verdict = ('PASS — Der Rahmen-Winkel moduliert Lernbarkeit UEBER die diag-'
                   'Masse hinaus: Innen-Bereich > Zufalls-Rahmen UND > volle '
                   'Ausrichtung, bei per Konstruktion konstantem DM. PURE '
                   'Geometrie im trainierten Netz. T6s Kurve war NICHT nur '
                   'Entkopplung. G8s Befund traegt in beide Regime.')
    else:
        verdict = ('KILL — FLAT oder falscher Peak bei konstantem DM: T6s Budget-'
                   'Kurve war vollstaendig Entkopplungs-getragen. "DM ist alles" '
                   'im Trainingsregime. Rahmen-Geometrie wirkt NUR im Reservoir '
                   '(fixes W, Ridge-Readout). Regime-Karte endgueltig, '
                   'Geometrie-Linie ehrlich beendet.')
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
