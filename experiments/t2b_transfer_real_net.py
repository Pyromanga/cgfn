#!/usr/bin/env python
'''T2b — TRANSFER-STEIN, DESIGN-REPARATUR NACH T2a-UNENTSCHEIDEN
(registriert 2026-11-07, VOR dem Lauf; einmalige Reparatur,
Praezedenz G4->G5)

T2a-DEFEKT (ehrlich): Delayed-XOR mit Luecken von 50-90 Frames
in einem System mit Leck-Halbwertszeit ~3 Frames. Kein Arm
lernte (alle exakt 0.5000, keine Divergenzen): Signalkuez,
nicht Instabilitaet. Ich hatte eine Aufgabe jenseits des
bewiesenen Memory-Horizonts angemeldet — Uebermut, nicht
Methode. Das Sanity-Gate hat das korrekt abgefangen.

T2b-REPARATUR (registriert, minimal):
  1. AUFGABE INS BEWIESENE GEBIET: XOR-Delay <= 19 Frames
     (Repo hat MC bis Lag 16 bewiesen, f34 white_delay8 bei
     Delay 8). T=24. Bit A an zufaelliger Position in [4,8],
     Kanal 0, 3 Frames gehalten. Bit B in [16,20], Kanal 1,
     3 Frames gehalten. Ziel am letzten Frame: XOR(A,B).
  2. INSTRUMENT = model.py-Konvention: RK2, tau=1.0, dt=0.1,
     substeps=5 (120 RK2-Schritte BPTT). Keine Euler-Pfusch-
     dyn mehr. lam_max=0.6 (ruhiges, speicherfaehiges Regime,
     G-Serie-Familie), Rest 0.9*lam*U(-0.9,0.9), Spektrum fix
     identisch in allen Armen.
  3. THESEN UND TORE UNVERAENDERT (wie T2a genagelt):
     2x2-Arme, Seeds 0..9 je Arm, b fix am Init, W+Readout
     trainiert (Adam, 600 Schritte, Batch 32, lr 5e-3).

ARME (unveraendert):
  BIAS  B0: b=0  |  BR: b ~ N(0,0.1)          (G4-Confound)
  FRAME RND: QR(randn)  |  AL06: per-Modus theta_i ~ U(0,0.6)
        via M = diag(cos th)+diag(sin th)@P0 (G8/G10-Familie)

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Messgroesse: finale Test-Accuracy (1024 frische Samples,
  fixer Test-Seed 777, identisch fuer alle Arme).
  PRIMAER, PASS genau dann wenn:
    (P1) BIAS-TRANSFER: std(acc) in BR >= 1.5 * std(acc) in
         B0 (je ueber beide FRAME-Arme gemittelt).
    UND (P2) FRAME-TRANSFER: median(acc) von AL06 > median
         (acc) von RND in BEIDEN Bias-Zellen.
  KILL: Sonst — die G-Befunde waschen sich im echten Training
     weg (oder waren nie allgemein).
  UNENTSCHEIDEN: Gesamt-Median < 0.60 (nicht gelernt) oder
     >3/10 Divergenzen in einem Arm. DANN IST T2 ENDGUELTIG
     TOT: keine zweite Reparatur (ZWEI-STRIKES-REGEL).
  SEKUNDAER: Median/Spread je Arm, Divergenzen, Laufzeit.
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
SEEDS = list(range(10))
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


def run_one(seed, bias_mode, frame_mode):
    gen = torch.Generator().manual_seed(seed)
    W0 = init_W(H, frame_mode, seed)
    Uw = 0.5 * torch.randn(H, 2, generator=gen)
    if bias_mode == 'B0':
        b = torch.zeros(H)
    elif bias_mode == 'BR':
        b = 0.1 * torch.randn(H, generator=gen)
    else:
        raise ValueError(bias_mode)
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
    out_path = 'results/t2b_transfer_real_net_results.txt'
    csv_path = 'results/t2b_transfer_real_net_table.csv'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T2b — TRANSFER-STEIN, DESIGN-REPARATUR (registriert: PASS nur wenn')
    L.append('P1 std(BR)>=1.5*std(B0) UND P2 median(AL06)>median(RND) in beiden Bias-Zellen)')
    L.append('XOR-Delay <=19 Frames (bewiesenes Gebiet), RK2/tau=1.0/dt=0.1/substeps=5, f64')
    L.append('=' * 118)

    rows = []
    accs = {}
    divs = {}
    for bias_mode in ('B0', 'BR'):
        for frame_mode in ('RND', 'AL06'):
            key = bias_mode + '_' + frame_mode
            vals = []
            ndiv = 0
            for seed in SEEDS:
                acc = run_one(seed, bias_mode, frame_mode)
                if acc is None or acc != acc:
                    ndiv += 1
                    acc = float('nan')
                vals.append(acc)
                rows.append(dict(arm=key, seed=seed, acc=acc))
            accs[key] = vals
            divs[key] = ndiv
            ok = [v for v in vals if v == v]
            L.append('')
            L.append('Arm %s: n_ok=%d/%d | median=%.4f | std=%.4f | min=%.4f max=%.4f'
                     % (key, len(ok), len(vals), median(ok), std(ok),
                        min(ok) if ok else float('nan'),
                        max(ok) if ok else float('nan')))
            Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('arm,seed,acc\n')
        for r in rows:
            fh.write('%s,%d,%.6g\n' % (r['arm'], r['seed'], r['acc']))

    L.append('')
    L.append('--- T2b-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    cells = ['B0_RND', 'B0_AL06', 'BR_RND', 'BR_AL06']
    meds = {k: median([v for v in accs[k] if v == v]) for k in cells}
    stds = {k: std([v for v in accs[k] if v == v]) for k in cells}
    all_med = float(torch.tensor([meds[k] for k in cells]).median())
    for k in cells:
        L.append('Zelle %s: median=%.4f std=%.4f' % (k, meds[k], stds[k]))

    std_b0 = 0.5 * (stds['B0_RND'] + stds['B0_AL06'])
    std_br = 0.5 * (stds['BR_RND'] + stds['BR_AL06'])
    p1 = std_br >= 1.5 * max(std_b0, 1e-9)
    L.append('P1 (Bias-Transfer): std(B0)=%.4f std(BR)=%.4f -> %s'
             % (std_b0, std_br, 'JA' if p1 else 'NEIN'))

    wins = 0
    for bm in ('B0', 'BR'):
        if meds[bm + '_AL06'] > meds[bm + '_RND']:
            wins += 1
    p2 = wins >= 2
    L.append('P2 (Frame-Transfer): AL06>RND in %d/2 Bias-Zellen -> %s'
             % (wins, 'JA' if p2 else 'NEIN'))

    total_div = sum(divs.values())
    sanity_learn = all_med >= 0.60
    sanity_div = all(divs[k] <= 3 for k in cells)
    L.append('Sanity: Gesamt-Median=%.4f (>=0.60? %s), Divergenzen=%d (%s)'
             % (all_med, 'JA' if sanity_learn else 'NEIN', total_div,
                'JA' if sanity_div else 'NEIN'))

    if not sanity_learn or not sanity_div:
        verdict = ('UNENTSCHEIDEN — auch nach der Reparatur nicht gelernt oder '
                   'Divergenzen. ZWEI-STRIKES-REGEL: T2 ist damit endgueltig '
                   'geschlossen, KEINE dritte Reparatur. Ehrliches Protokoll-'
                   'Ergebnis: Der Transfer-Stein ist im 40-EUR-Labor nicht '
                   'sauber feuerbar (Aufgaben-Horizont vs. Leck-Konstante im '
                   'Spannungsfeld); die Gate-Disziplin bleibt der Gewinn.')
    elif p1 and p2:
        verdict = ('PASS — Beide G-Befunde transferieren: Der Bias-Lotto ist '
                   'im end-to-end trainierten Netz sichtbar (BR streut '
                   '>=1.5x staerker), und der 0.6-rad-Fehlwinkel-Kuchen '
                   'schlaegt den Zufalls-Rahmen selbst nach dem Training. '
                   'Der Anker ist keine Reservoir-Kuriositaet.')
    else:
        verdict = ('KILL — Die G-Befunde waschen sich im echten Training weg '
                   '(oder sind nie allgemein gewesen): Bias-Lotto unsichtbar '
                   'und/oder Anker-Rahmen ohne Vorteil nach dem Training.')
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
