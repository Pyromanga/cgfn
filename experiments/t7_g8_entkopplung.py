#!/usr/bin/env python
'''T7 — DER KONFUND KEHRT ZURUECK: IST G8 NUR ENTKOPPLUNG?
(registriert 2026-11-07, VOR dem Lauf)

DIE FRAGE (hoechste Entropie im Raum): T6 hat im Trainings-Regime
gezeigt, dass die "Anker-Geometrie" von der diag-Masse DM getragen
wird (exakt diagonales W trainiert bei P_lernt 0.70). G8 — das
gefeierte positive Ergebnis des Anker-Kapitels — verglich EXAKT
DIESELBE KONSTRUKTION im Reservoir-Regime (Budget ~0.6 vs
Zufalls-Rahmen, gleiches Spektrum, Memory Capacity via Ridge):
DM WURDE NIE KONTROLLIERT. Vielleicht waren die +16..41% MC auch
nur Selbst-Schleifen. T7 greift den eigenen PASS-Befund mit dem
Messer an, das T6 geschmiedet hat.

T7-THESE (die eine These dieses Laufs):
  Im Reservoir-Regime ist MC eine Funktion der Anker-Geometrie
  JENSEITS der Entkopplung: Die Budget-Kurve MC(theta*) hat einen
  echten Innen-Peak (0.15..0.60), obwohl DM monoton mit theta*
  faellt. Wenn MC stattdessen DM monoton folgt (Peak bei
  theta*=0), war G8 Entkopplung und wird zurueckgezogen.

INSTRUMENT (G-Serie-bewaehrt, tanh-bug-immun, kein Training):
  Memory Capacity am Zustand. Drive: u_t ~ 0.5*randn, T=300,
  7 Trajektorien (5 Ridge-Train, 2 frische Test), Washout 50.
  Ridge 1e-6, fp64, MC_k = R^2 der Vorhersage von u_{t-k} aus
  z_t, k in {1,2,4,8,16}; MC_total = Summe. b=0 (G-Rezept,
  Reservoir). Dyn identisch zur T-Serie (RK2, tau=1.0, dt=0.1,
  substeps=5), H=48, lam_max=0.6, Spektrum FIX identisch in
  allen Armen — nur der Rahmen lottert.

ARME (wie T6): Budget theta* in {0.00, 0.15, 0.30, 0.60, 1.20}
  per-Modus theta_i ~ U(0, theta*) via M = diag(cos th) +
  diag(sin th) @ P0, R = QR(M); plus RND. FRISCHE Seeds 50..59.
  DM = mean(|diag W|)/mean(|W|) am Init mitgeloggt (Konfund-
  Indikator, sekundaer).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  PRIMAER, PASS genau dann wenn:
    (P1a) MC-Peak-Arm (max median MC_total) in {B15, B30, B60}.
    UND (P1b) median MC_total(B60) > 1.10 * median MC_total(B00)
         (volle Ausrichtung toetet im Reservoir — G6/G8-Echo,
         nicht-monoton UND signifikant).
    UND (P2) median MC_total(B60) > median MC_total(RND)
         (G8-Replikation unter dieser Konstruktion).
  KILL: Peak bei B00 bzw. MC monoton fallend in theta* — G8 war
     Entkopplung: Ehrliche Zurueckziehung der Geometrie-Deutung;
     Einheitsgesetz Entkopplung ueber beide Regime.
  UNENTSCHEIDEN: Sanity — median MC_total(RND) < 1.0 (Instrument
     defekt, MC untypisch niedrig gegenueber G-Serie-Werten)
     oder >3/10 Crashes in einem Arm.
  SEKUNDAER (Bericht, kein Gate): volle Kurve MC gegen theta*,
    DM je Arm, MC_k-Zerlegung. Die nachtraegliche Korrelation
    MC-vs-DM entscheidet die Geometrie-Frage nicht als Gate,
    sondern als Bericht-Anhang (Peak-Lage zaehlt, nicht die
    Korrelation).
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.
'''
import os
import time
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

FIXED_P0 = 4242
FIXED_SPEC = 9999
H = 48
T_SEQ = 300
WASH = 50
TAU = 1.0
DT = 0.1
SUBSTEPS = 5
LAM_MAX = 0.6
KS = [1, 2, 4, 8, 16]
N_TRAIN = 5
N_TEST = 2
INPUT_SCALE = 0.5
RIDGE = 1e-6
SEEDS = list(range(50, 60))
BUDGETS = [('B00', 0.00), ('B15', 0.15), ('B30', 0.30),
           ('B60', 0.60), ('B120', 1.20), ('RND', None)]


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


def drive(batch, gen):
    return INPUT_SCALE * torch.randn(batch, T_SEQ, 1, generator=gen)


def dyn(z, W, uin):
    return -z / TAU + torch.tanh(z) @ W.T + uin


def states(W, x):
    B = x.shape[0]
    z = torch.zeros(B, H)
    out = []
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, xt)
            k2 = dyn(z + 0.5 * DT * k1, W, xt)
            z = z + DT * k2
        out.append(z)
    return torch.stack(out, dim=1)  # (B, T, H)


def mc_total(W, seed):
    # Train-Trajektorien
    g1 = torch.Generator().manual_seed(9000 + seed)
    xtr = drive(N_TRAIN, g1)
    str_ = states(W, xtr)[:, WASH:, :].reshape(-1, H)
    utr = xtr[:, WASH:, 0].reshape(-1)
    # Test-Trajektorien (frisch)
    g2 = torch.Generator().manual_seed(9500 + seed)
    xte = drive(N_TEST, g2)
    ste = states(W, xte)[:, WASH:, :].reshape(-1, H)
    ute = xte[:, WASH:, 0].reshape(-1)

    mcs = []
    for k in KS:
        tgt_tr = xtr[:, WASH - k:, 0].reshape(-1)
        tgt_te = xte[:, WASH - k:, 0].reshape(-1)
        A = str_.T @ str_ + RIDGE * torch.eye(H)
        w = torch.linalg.solve(A, str_.T @ tgt_tr)
        pred = ste @ w
        mse = float(((pred - tgt_te) ** 2).mean())
        var = float(tgt_te.var())
        mcs.append(1.0 - mse / max(var, 1e-12))
    return float(sum(mcs)), [round(m, 4) for m in mcs]


def median(v):
    if not v:
        return float('nan')
    return float(torch.tensor(v).median())


def main():
    out_path = 'results/t7_g8_entkopplung_results.txt'
    csv_path = 'results/t7_g8_entkopplung_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T7 — IST G8 NUR ENTKOPPLUNG? (registriert: PASS nur wenn')
    L.append('P1a MC-Peak in {B15,B30,B60} UND P1b MC(B60)>1.10*MC(B00) UND P2 MC(B60)>MC(RND))')
    L.append('Reservoir-Regime, MC via Ridge, b=0, Seeds 50..59, f64')
    L.append('=' * 118)

    rows = []
    stats = {}
    for name, budget in BUDGETS:
        vals = []
        dms = []
        ncrash = 0
        for seed in SEEDS:
            try:
                W0 = init_W(H, budget, seed)
                dm = float(W0.diag().abs().mean() / W0.abs().mean())
                mc, ks_split = mc_total(W0, seed)
                if mc != mc:
                    ncrash += 1
                    mc = float('nan')
            except Exception:
                ncrash += 1
                mc = float('nan')
                ks_split = []
                dm = float('nan')
            vals.append(mc)
            dms.append(dm)
            rows.append(dict(arm=name, seed=seed, mc=mc))
        ok = [v for v in vals if v == v]
        stats[name] = dict(med=median(ok), ndiv=ncrash,
                           dm=float(torch.tensor([d for d in dms if d == d]).mean()))
        L.append('')
        L.append('Arm %s (budget=%s): n_ok=%d/%d | median MC_total=%.4f | DM=%.3f | crash=%d'
                 % (name, 'RND' if budget is None else ('%.2f' % budget),
                    len(ok), len(vals), median(ok), stats[name]['dm'], ncrash))
        L.append('  Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, vals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,seed,mc_total\n')
        for r in rows:
            fh.write('%s,%d,%.6g\n' % (r['arm'], r['seed'], r['mc']))

    L.append('')
    L.append('--- T7-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    names = [n for n, _ in BUDGETS]
    for n in names:
        L.append('%s: median MC_total=%.4f DM=%.3f' % (n, stats[n]['med'], stats[n]['dm']))

    max_arm = max(names, key=lambda n: stats[n]['med'] if stats[n]['med'] == stats[n]['med'] else -1)
    p1a = max_arm in ('B15', 'B30', 'B60')
    p1b = stats['B60']['med'] > 1.10 * stats['B00']['med']
    p2 = stats['B60']['med'] > stats['RND']['med']
    L.append('Peak-Arm: %s (median MC_total=%.4f)' % (max_arm, stats[max_arm]['med']))
    L.append('P1a (Peak in {B15,B30,B60}): %s | P1b (MC(B60)>1.10*MC(B00)): %s'
             % ('JA' if p1a else 'NEIN', 'JA' if p1b else 'NEIN'))
    L.append('P2 (MC(B60) > MC(RND)): %s' % ('JA' if p2 else 'NEIN'))

    sanity_instr = stats['RND']['med'] >= 1.0
    sanity_div = all(stats[n]['ndiv'] <= 3 for n in names)
    L.append('Sanity: median MC_total(RND)=%.4f (>=1.0? %s), Crashes gesamt=%d (%s)'
             % (stats['RND']['med'], 'JA' if sanity_instr else 'NEIN',
                sum(stats[n]['ndiv'] for n in names),
                'JA' if sanity_div else 'NEIN'))

    if not sanity_instr or not sanity_div:
        verdict = ('UNENTSCHEIDEN — Sanity gerissen (Instrument untypisch oder '
                   'Crashes). Post-Mortem, kein Verdikt.')
    elif p1a and p1b and p2:
        verdict = ('PASS — G8 verTEIDIGT seine Geometrie: Die Reservoir-MC-Kurve '
                   'hat einen echten Innen-Peak im Budget, obwohl DM monoton '
                   'faellt, und volle Ausrichtung kostet (G6/G8-Echo). '
                   'Der Regime-Kontrast ist geschaerft und zementiert: '
                   'Geometrie lebt NUR im Reservoir, Entkopplung NUR im '
                   'Training. Das Anker-Kapitel behaelt seinen Kern-Befund '
                   'im Reservoir, verliert ihn im trainierten Netz.')
    else:
        verdict = ('KILL — G8 war Entkopplung: Die MC-Kurve folgt dem Budget '
                   'monoton fallend bzw. der Peak liegt bei voller Ausrichtung. '
                   'Ehrliche ZURUECKZIEHUNG der Geometrie-Deutung von G8/G10: '
                   'Die +16..41% waren Selbst-Schleifen-Masse. Einheitsgesetz: '
                   'Entkopplung traegt in BEIDEN Regimen; die Anker-Geometrie '
                   'hat nirgends eigene Kraft. Das Anker-Kapitel wird auf '
                   'diesen Nenner gebracht.')
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
