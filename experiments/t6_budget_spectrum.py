#!/usr/bin/env python
'''T6 — LERNBARKEITS-SPEKTRUM DES ANKER-BUDGETS (registriert 2026-11-07, VOR dem Lauf)

WARUM T6 (kritische Rueckfrage an T3): Das Punktpaar AL06 vs
RND ist ZWEI Geschichten in einem Arm verknotet: (a) kohaerenter
Rahmen mit Fehlwinkel-Budget 0.6, (b) banale Entkopplung — fast
diagonales W hat starke Selbst-Schleifen (diag ~ lam_i), RND hat
diag ~ 0. G8 hat (a) im Reservoir (Ridge, kein Training) gezeigt;
im trainierten Netz koennte (b) der einzige Grund sein. Ein
Punktpaar kann (a) und (b) prinzipiell nicht trennen. T6 loest
auf: die KURVE P(lernt) gegen das Budget.

T6-THESE (die eine These dieses Laufs):
  Die Lernbarkeit ist eine NICHT-MONOTONE Funktion des Fehlwinkel-
  Budgets mit Peak im Zwischenbereich — genug Rahmen-Mischung fuer
  nichtlineare Berechnung, wenig Interferenz. Wenn der Peak bei
  theta*=0 liegt (mehr Ausrichtung = besser), kollabiert die Anker-
  Geometrie auf die triviale Entkopplungs-Story (b). Wenn der Peak
  bei 1.2/RND liegt, war T3 ein Lotto-Artefakt.

AUFGABE: Paritaet (dort lebt die Evidenz-Wand T2b/T3: 15/20 vs
  3/20 perfekt). Delayed-XOR, Delay <=19 Frames, T=24, Bits 3
  Frames gehalten, Positionen randomisiert. ZIEL: XOR(A,B) am
  letzten Frame, 2 Klassen.

INSTRUMENT (identisch zu T3): RK2, tau=1.0, dt=0.1, substeps=5;
  H=48, lam_max=0.6, Spektrum FIX identisch in allen Armen (nur
  der Rahmen lottert); Adam, 600 Schritte, Batch 32, lr 5e-3;
  b ~ N(0,0.1) fix am Init in ALLEN Armen (Symmetrie-Brecher,
  T2b-Lehre); W+Readout trainiert; f64; frische Seeds 40..49.

ARME (6): Budget theta* in {0.00, 0.15, 0.30, 0.60, 1.20} mit
  per-Modus theta_i ~ U(0, theta*) via
  M = diag(cos th) + diag(sin th) @ P0, R = QR(M);
  plus RND (R = QR(randn)), das theta->unendlich-Ende der Kurve.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Messgroessen: P_lernt = Fraktion der Runs mit Test-Accuracy
  >= 0.99 (1024 frische Samples, Test-Seed 777, identisch fuer
  alle Arme); sekundaer median(acc) und diag-Masse
  DM = mean(|diag W|)/mean(|W|) am Init (Konfund-Indikator).
  PRIMAER, PASS genau dann wenn:
    (P1) ZWISCHEN-PEAK: Der Arm mit maximaler P_lernt liegt bei
         Budget in {0.15, 0.30, 0.60} UND P_lernt(theta*=0.60)
         - P_lernt(theta*=0.00) >= 0.3 (volle Ausrichtung ist
         NICHT optimal — Entkopplungs-Story allein widerlegt).
    UND (P2) ENDE-ABFALL: P_lernt(theta*=0.60) > P_lernt(RND)
         (Konsistenz mit T2b/T3, gleiche Richtung).
  KILL: Sonst — Peak bei 0.00 (Entkopplungs-Konfund gewinnt,
     Anker-Geometrie ohne eigene Kraft im trainierten Netz) oder
     Peak bei 1.20/RND bzw. P_lernt(0.60) <= P_lernt(RND)
     (T3-Wand war Lotto).
  UNENTSCHEIDEN: Sanity — kein Arm erreicht P_lernt >= 0.5
     (nichts gelernt, Instrument defekt) oder >3/10 Divergenzen
     in einem Arm. Dann Serie-Post-Mortem.
  SEKUNDAER (Bericht, kein Gate): volle Budget-Kurve P_lernt,
    median(acc), DM je Arm; Laufzeit. DM ermoeglicht die
    nachtraegliche Konfund-Analyse: korreliert die Kurve mit DM
     (Entkopplung) oder mit der Geometrie (Budget)?
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
SEEDS = list(range(40, 50))
A_LO, A_HI = 4, 8
B_LO, B_HI = 16, 20
HOLD = 3
BUDGETS = [('B00', 0.00), ('B15', 0.15), ('B30', 0.30),
           ('B60', 0.60), ('B120', 1.20), ('RND', None)]


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


def frame_R(H, budget, seed):
    gen = torch.Generator().manual_seed(seed)
    genP = torch.Generator().manual_seed(FIXED_P0)
    P0, _ = torch.linalg.qr(torch.randn(H, H, generator=genP))
    if budget is None:
        R, _ = torch.linalg.qr(torch.randn(H, H, generator=gen))
        return R
    if budget == 0.0:
        return torch.eye(H)
    th = budget * torch.rand(H, generator=gen)
    M = torch.diag(torch.cos(th)) + torch.diag(torch.sin(th)) @ P0
    R, _ = torch.linalg.qr(M)
    return R


def init_W(H, budget, seed):
    genS = torch.Generator().manual_seed(FIXED_SPEC)
    eigs = torch.cat([torch.tensor([LAM_MAX]),
                      0.9 * LAM_MAX * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)])
    R = frame_R(H, budget, seed)
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


def run_one(seed, budget):
    gen = torch.Generator().manual_seed(seed)
    W0 = init_W(H, budget, seed)
    Uw = 0.5 * torch.randn(H, 2, generator=gen)
    b = 0.1 * torch.randn(H, generator=gen)
    dm = float(W0.diag().abs().mean() / W0.abs().mean())
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
    out_path = 'results/t6_budget_spectrum_results.txt'
    csv_path = 'results/t6_budget_spectrum_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T6 — LERNBARKEITS-SPEKTRUM DES ANKER-BUDGETS (registriert: PASS nur wenn')
    L.append('P1 Zwischen-Peak {0.15,0.30,0.60} UND P_lernt(0.60)-P_lernt(0.00)>=0.3')
    L.append('UND P2 P_lernt(0.60) > P_lernt(RND); Seeds 40..49, f64, Bias in allen Armen)')
    L.append('=' * 118)

    rows = []
    stats = {}
    for name, budget in BUDGETS:
        vals = []
        dms = []
        ndiv = 0
        for seed in SEEDS:
            acc, dm = run_one(seed, budget)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            vals.append(acc)
            dms.append(dm)
            rows.append(dict(arm=name, seed=seed, acc=acc))
        ok = [v for v in vals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        stats[name] = dict(pf=pf, med=median(ok), ndiv=ndiv,
                           dm=float(torch.tensor(dms).mean()))
        L.append('')
        L.append('Arm %s (budget=%s): n_ok=%d/%d | P_lernt=%.2f | median acc=%.4f | DM=%.3f | div=%d'
                 % (name, 'RND' if budget is None else ('%.2f' % budget),
                    len(ok), len(vals), pf, median(ok), stats[name]['dm'], ndiv))
        L.append('  Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, vals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%.6g\n' % (r['arm'], r['seed'], r['acc']))

    L.append('')
    L.append('--- T6-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    names = [n for n, _ in BUDGETS]
    for n in names:
        L.append('%s: P_lernt=%.2f median=%.4f DM=%.3f'
                 % (n, stats[n]['pf'], stats[n]['med'], stats[n]['dm']))

    max_arm = max(names, key=lambda n: stats[n]['pf'] if stats[n]['pf'] == stats[n]['pf'] else -1)
    p1a = max_arm in ('B15', 'B30', 'B60')
    p1b = (stats['B60']['pf'] - stats['B00']['pf']) >= 0.3
    p2 = stats['B60']['pf'] > stats['RND']['pf']
    L.append('Peak-Arm: %s (P_lernt=%.2f)' % (max_arm, stats[max_arm]['pf']))
    L.append('P1a (Peak in {B15,B30,B60}): %s | P1b (P_lernt(B60)-P_lernt(B00)>=0.3): %s'
             % ('JA' if p1a else 'NEIN', 'JA' if p1b else 'NEIN'))
    L.append('P2 (P_lernt(B60) > P_lernt(RND)): %s' % ('JA' if p2 else 'NEIN'))

    any_learn = max((stats[n]['pf'] for n in names if stats[n]['pf'] == stats[n]['pf']),
                    default=float('nan')) >= 0.5
    sanity_div = all(stats[n]['ndiv'] <= 3 for n in names)
    L.append('Sanity: bester Arm P_lernt=%.2f (>=0.5? %s), Divergenzen gesamt=%d (%s)'
             % (max(stats[n]['pf'] for n in names),
                'JA' if any_learn else 'NEIN',
                sum(stats[n]['ndiv'] for n in names),
                'JA' if sanity_div else 'NEIN'))

    if not any_learn or not sanity_div:
        verdict = ('UNENTSCHEIDEN — Sanity gerissen (nichts gelernt oder Divergenzen). '
                   'Serie-Post-Mortem; keine weitere Paritaets-Messung.')
    elif p1a and p1b and p2:
        verdict = ('PASS — Lernbarkeit ist eine nicht-monotone Funktion des Anker-'
                   'Budgets mit Peak im Zwischenbereich: Volle Ausrichtung ist '
                   'NICHT optimal (Entkopplungs-Konfund allein widerlegt), und '
                   'das 0.6-Budget schlaegt den Zufalls-Rahmen. Die Kurve '
                   'verschmilzt G6 (voll symmetriert toetet), G7 (Kliff), G8 '
                   '(0.6 gewinnt) zu EINEM Gesetz im trainierten Netz. Der '
                   'Anker ist Geometrie, nicht Selbst-Schleifen-Masse.')
    elif (not p1a) and max_arm == 'B00':
        verdict = ('KILL (Entkopplungs-Konfund) — Peak bei Budget 0: Die '
                   'Trainierbarkeit steigt monoton mit der Ausrichtung. Die '
                   'T3-Wand erklaert sich durch Selbst-Schleifen-Masse, nicht '
                   'durch Anker-Geometrie. G8s Geometrie-Befund hat im '
                   'trainierten Netz keine eigene Kraft. Ehrliches Ende.')
    else:
        verdict = ('KILL (Geometrie tot oder Lotto) — Peak am falschen Ende '
                   '(B120/RND) oder P_lernt(B60) <= P_lernt(RND): Die T3-Wand '
                   'war ein Seed-Lotto-Artefakt bzw. der Anker hat keinen '
                   'Lernbarkeits-Vorteil. Ehrliches Ende der Transfer-Linie.')
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
