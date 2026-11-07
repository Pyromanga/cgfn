#!/usr/bin/env python
'''T4b — ANKER AUF PRODUKT, DESIGN-REPARATUR (registriert 2026-11-07, VOR dem Lauf; Strike 1/1 der Produkt-Familie, Praezedenz G4->G5)

T4-DEFEKT (ehrlich): Delay 8 statt der bewiesenen f34-Basis
(Delay 3). u_{t-8} ist im Zustand nur mit ~2% Amplitude vertreten
(Zerfall ~0.61/Frame ohne Rekurrenz-Verstaerkung; Top-Mode
haelt nur ~0.20 ueber 8 Frames). Die Produkt-Aufgabe braucht
aber ZWEI Faktoren multiplikativ — beide Arme bei R2 ~ 0,
Effekt exakt 0, keine Divergenzen: Signalmangel, kein
These-Statement. Gleiche Fehlerklasse wie T2a (Uebermut
gegenueber der bewiesenen Basis), vom korrigierten Sanity-Gate
sauber gefangen.

T4b-REPARATUR (registriert, minimal):
  1. Delay zurueck auf 3 (f34-Praezedenz: prod_task = u_t *
     u_{t-3}). T=24. Ziel y_t = u_t * u_{t-3} fuer t >= 3.
  2. Training 600 -> 1200 Schritte (Regression braucht
     laengere Readout-Konvergenz als Paritaets-CE).
  3. FRISCHE Seeds 30..39 (kein Recycling aus T4).
  THESEN UND TOERE UNVERAENDERT zu T4 genagelt.

T4b-THESE (unveraendert):
  Der 0.6-rad-Fehlwinkel-Anker (G8/G10-Familie) schlaegt den
  Zufalls-Anker auf der Produkt-Familie im end-to-end
  trainierten Netz.

INSTRUMENT: RK2, tau=1.0, dt=0.1, substeps=5; H=48, lam_max=0.6,
  Spektrum fix identisch; Adam 1200 Schritte, Batch 32, lr 5e-3;
  b ~ N(0,0.1) fix am Init in BEIDEN Armen; W+Readout trainiert;
  f64. Input u ~ 0.5*randn.

ARME: RND (R = QR(randn(seed))) vs AL06 (per-Modus
  theta_i ~ U(0,0.6), M = diag(cos th)+diag(sin th)@P0).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Messgroesse: R2 = 1 - mse/var(y) auf 1024 frische Sequenzen
  (fixer Test-Seed 777, identisch fuer beide Arme), Vorhersage
  ueber Frames 3..23.
  PRIMAER, PASS genau dann wenn:
    (P1) EFFEKT: median(R2 AL06) - median(R2 RND) >= 0.15.
    UND (P2) GELOEST: Fraktion geloester Runs (R2 >= 0.9) in
         AL06 >= 0.5 UND >= 2x Fraktion in RND.
  KILL: P1 oder P2 verfehlt, waehrend Sanity haelt — kein
     Anker-Vorteil auf der Produkt-Familie; Paritaet bleibt
     Spezialfall, ehrlich verbucht. KEINE zweite Reparatur
     (ZWEI-STRIKES-REGEL fuer die Produkt-Familie).
  UNENTSCHEIDEN: median(R2 AL06) < 0.5 (auch der Anker lernt
     es nicht — Instrument-Problem) ODER >3/10 Divergenzen
     in einem Arm. Dann ist die Produkt-Familie TOT.
  SEKUNDAER: Mediane, Spreads, geloest-Fraktionen, volle
    Seed-Tabelle, Laufzeit.
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
DELAY = 3
TAU = 1.0
DT = 0.1
SUBSTEPS = 5
STEPS = 1200
BATCH = 32
LR = 5e-3
LAM_MAX = 0.6
SEEDS = list(range(30, 40))
INPUT_SCALE = 0.5


def make_seq(batch, gen):
    return INPUT_SCALE * torch.randn(batch, T_SEQ, 1, generator=gen)


def targets(x):
    return x[:, DELAY:, 0] * x[:, :-DELAY, 0]


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


def states_all(W, b, Uw, x):
    B = x.shape[0]
    z = torch.zeros(B, H)
    out = []
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T)
            z = z + DT * k2
        out.append(z)
    return torch.stack(out, dim=1)


def run_one(seed, frame_mode):
    gen = torch.Generator().manual_seed(seed)
    W0 = init_W(H, frame_mode, seed)
    Uw = 0.5 * torch.randn(H, 1, generator=gen)
    b = 0.1 * torch.randn(H, generator=gen)
    W = W0.clone().requires_grad_(True)
    head_w = torch.zeros(1, H, requires_grad=True)
    head_b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W, head_w, head_b], lr=LR)
    train_gen = torch.Generator().manual_seed(1000 + seed)
    for _ in range(STEPS):
        x = make_seq(BATCH, train_gen)
        y = targets(x)
        st = states_all(W, b, Uw, x)
        pred = st[:, DELAY:, :] @ head_w.T + head_b
        loss = torch.nn.functional.mse_loss(pred.squeeze(-1), y)
        if not torch.isfinite(loss):
            return None
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT = make_seq(1024, test_gen)
    yT = targets(xT)
    with torch.no_grad():
        st = states_all(W, b, Uw, xT)
        pred = (st[:, DELAY:, :] @ head_w.T + head_b).squeeze(-1)
        mse = float(((pred - yT) ** 2).mean())
        var = float(yT.var())
        r2 = 1.0 - mse / max(var, 1e-12)
    return r2


def median(v):
    if not v:
        return float('nan')
    return float(torch.tensor(v).median())


def std(v):
    if len(v) < 2:
        return float('nan')
    return float(torch.tensor(v).std())


def main():
    out_path = 'results/t4b_transfer_prod_results.txt'
    csv_path = 'results/t4b_transfer_prod_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T4b — ANKER AUF PRODUKT, REPARATUR (registriert: PASS nur wenn')
    L.append('P1 median(R2 AL06)-median(R2 RND)>=0.15 UND P2 geloest(AL06)>=0.5 und >=2x RND)')
    L.append('Delay 3 (f34-Basis), 1200 Schritte, frische Seeds 30..39, f64')
    L.append('=' * 118)

    rows = []
    accs = {}
    divs = {}
    for frame_mode in ('RND', 'AL06'):
        vals = []
        ndiv = 0
        for seed in SEEDS:
            r2 = run_one(seed, frame_mode)
            if r2 is None or r2 != r2:
                ndiv += 1
                r2 = float('nan')
            vals.append(r2)
            rows.append(dict(arm=frame_mode, seed=seed, r2=r2))
        accs[frame_mode] = vals
        divs[frame_mode] = ndiv
        ok = [v for v in vals if v == v]
        gf = sum(1 for v in ok if v >= 0.9) / len(ok) if ok else float('nan')
        L.append('')
        L.append('Arm %s: n_ok=%d/%d | median R2=%.4f | std=%.4f | geloest (R2>=0.9)=%.2f'
                 % (frame_mode, len(ok), len(vals), median(ok), std(ok), gf))
        L.append('  Seeds: ' + ', '.join('%d:%.4f' % (s, v) for s, v in zip(SEEDS, vals)))
        Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,seed,r2\n')
        for r in rows:
            fh.write('%s,%d,%.6g\n' % (r['arm'], r['seed'], r['r2']))

    L.append('')
    L.append('--- T4b-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    okR = [v for v in accs['RND'] if v == v]
    okA = [v for v in accs['AL06'] if v == v]
    medR, medA = median(okR), median(okA)
    gfR = sum(1 for v in okR if v >= 0.9) / len(okR) if okR else 0.0
    gfA = sum(1 for v in okA if v >= 0.9) / len(okA) if okA else 0.0
    L.append('median R2(RND)=%.4f median R2(AL06)=%.4f Effekt=%.4f' % (medR, medA, medA - medR))
    L.append('geloest-Fraktion RND=%.2f AL06=%.2f' % (gfR, gfA))

    p1 = (medA - medR) >= 0.15
    p2 = (gfA >= 0.5) and (gfA >= 2.0 * max(gfR, 1e-9))
    L.append('P1 (Effekt >=0.15): %s | P2 (geloest>=0.5 und 2x RND): %s'
             % ('JA' if p1 else 'NEIN', 'JA' if p2 else 'NEIN'))

    total_div = sum(divs.values())
    sanity_learn = medA >= 0.5
    sanity_div = all(divs[k] <= 3 for k in ('RND', 'AL06'))
    L.append('Sanity (korrigiert): median R2(AL06)=%.4f (>=0.50? %s), Divergenzen=%d (%s)'
             % (medA, 'JA' if sanity_learn else 'NEIN', total_div,
                'JA' if sanity_div else 'NEIN'))

    if not sanity_learn or not sanity_div:
        verdict = ('UNENTSCHEIDEN — Sanity gerissen: Auch der Anker-Arm lernt die '
                   'Produkt-Aufgabe nicht (R2 < 0.5) oder Divergenzen. ZWEI STRIKES: '
                   'Die Produkt-Familie ist TOT. Kein dritter Versuch. Ehrliches '
                   'Protokoll-Ergebnis: Regression unter BPTT ist im 40-EUR-Labor '
                   'fuer diese Aufgabenklasse nicht sauber feuerbar.')
    elif p1 and p2:
        verdict = ('PASS — Der 0.6-rad-Anker schlaegt den Zufalls-Anker auf der '
                   'Produkt-Familie (Effekt >=0.15 UND >=2x geloest-Fraktion). Der '
                   'Anker-Transfer ist FAMILIEN-UEBERGREIFEND (Paritaet: '
                   'T3/T2b-Exploration; Produkt: T4b), frische Seeds, konsistentes '
                   'Bild. Der Anker ist eine Aussage ueber echtes Lernen.')
    else:
        verdict = ('KILL — Kein Anker-Vorteil auf der Produkt-Familie (Effekt- oder '
                   'geloest-Gate verfehlt, waehrend der Anker-Arm lernt). Paritaet '
                   'bleibt ein Spezialfall: Der Anker-Transfer ist NICHT familien-'
                   'uebergreifend. Ehrliches Ende der Transfer-Linie.')
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
