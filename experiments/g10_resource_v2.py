#!/usr/bin/env python
'''G10 — RESSOURCEN-VERTEILER v2: BUDGET-MATCHED, REPARIERT, LETZTER
VERSUCH DIESER THESE. (registriert 2026-11-06, VOR dem Lauf)

VORGESCHICHTE (G9-KILL + zwei diagnostizierte Design-Defekte):
  (D1) G9s CONC-Arm war degeneriert: Konzentration auf den
       GETRIEBENEN Modus 0 richtete alle anderen Modi verse-
       hentlich SYMMETRISCH aus -> totes Reservoir (0.4722,
       std=0). Er mass "versehentliche Symmetrie", nicht
       "Konzentration". (D2) Einzel-Seed-Kalibrierung in den
       s=0.8-Zellen (std~0.74) ist Rauschen: UNI fiel dort an
       Kalibrierungs-Glueck, nicht an der These.
  REPARATUR (wie G4->G5: erlaubte Design-Reparatur, GENAU
  EINMAL, dies ist der letzte Versuch dieser These):
  (R1) CONC2 konzentriert auf EINEN UNGETRIEBENEN Modus
       (Index 1), alle anderen Modi bekommen budget-matchede
       Winkel NICHT null — kein versehentliches Symmetrie-
       Total. (R2) Kalibrierung ueber 3 Seeds (0,1,2), nicht
       einen. (R3) Alle Familien sind BUDGET-MATCHED:
       sum(theta_i) identisch ueber alle Arme.

DIE ENTEGETROPISCHE IDEE (nutzt alles Gelernte):
  Wenn der Anker ein Ressourcen-Verteiler ist (G8: +16..41%),
  warum sollte die FLACHE Verteilung optimal sein? Die Modi
  tragen unterschiedlich viel zur getriebenen Dynamik bei —
  die Ressource duerfte SPEKTRAL gerecht verteilt werden
  muessen. Zwei gerichtete Varianten (die Richtung ist die
  offene Wette, beide registriert):
    SPEC-L: langsame Modi (klein |lambda|) bekommen MEHR
       Fehlwinkel — sie muessen saettigen, um mitzuhalten.
    SPEC-H: schnelle Modi (gross |lambda|) bekommen MEHR
       Fehlwinkel — die Ressource gehoert zu den Traegern.
  theta_i = theta* * H * w_i / sum(w_j), w nach Variante.

ARME (Budget sum(theta_i) = H * theta* ueberall):
  UNI:   w_i = const (flach, G8/G9-Champion der s=0-Zellen).
  SPEC-L: w_i propto (lam_max - |lambda_i|).
  SPEC-H: w_i propto |lambda_i|.
  CONC2: theta_1 = min(H*theta*, 1.5) (numerisches Cap),
         theta_i = theta* (kleiner Rest) fuer i != 1 — KEINE
         Symmetrie-Ausrichtung, Reparatur (R1).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG, LETZTER
VERSUCH — bei KILL ist die Verteiler-These endgueltig tot):
  Zellen: H in {32,128} x s in {0.0, 0.8}. Seeds 0..19.
  Bias genullt, float64, lam=1.0, Spektrum fix, Q rotiert,
  Antrieb U=Q[:,0]. theta* EINMAL pro Zelle auf Seeds 0,1,2
  (Mittel) kalibriert aus Raster {0.3,0.6,0.9,1.2,1.5},
  gilt budget-matched fuer ALLE Arme.
  PRIMAER, PASS genau dann wenn:
    (P1)  mean_MC_UNI >= 1.05 * mean_MC_CONC2 in >= 3/4 Zellen
          (Konzentration schadet — Verteiler-These lebt)
    UND (P2)  max(mean_MC_SPEC_L, mean_MC_SPEC_H) >= 1.05 *
          mean_MC_UNI in >= 2/4 Zellen (die flache Verteilung
          ist NICHT optimal — die Struktur ist designbar)
  PASS -> "Der Anker ist ein spektral-designbarer Ressourcen-
     Verteiler: Budget, Verteilung UND Richtung sind Knoepfe."
  KILL: Sonst — flach ist optimal oder nichts zaehlt; These
     endgueltig gefallen, G8 bleibt Parameter-Befund.
  UNENTSCHEIDEN: >30% ungueltige Punkte auf einer Seite.
  SEKUNDAER (Bericht): Richtung von SPEC-L vs SPEC-H je Zelle
    (L>H oder H<L, Konsistenz ueber Zellen); std_MC je Arm.
'''
import os
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import IS
from cgfn.field import drive_input

FIXED = 9999
FRAME_SEED = 4242
THETA_GRID = [0.3, 0.6, 0.9, 1.2, 1.5]
CAL_SEEDS = [0, 1, 2]
DELAYS = [1, 2, 4, 8, 16]
T_TOTAL = 300
WASHOUT = 50
RIDGE = 1e-6
DT = 0.1
SUBSTEPS = 5


def factored_WQT(H, lam_max, s_nn, q_seed, spec_seed, noise_seed):
    genQ = torch.Generator().manual_seed(q_seed)
    A = torch.randn(H, H, generator=genQ)
    Q, _ = torch.linalg.qr(A)
    genS = torch.Generator().manual_seed(spec_seed)
    others = lam_max * 0.9 * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)
    eigs = torch.cat([torch.tensor([lam_max]), others])
    T = torch.diag(eigs)
    if s_nn > 0:
        genN = torch.Generator().manual_seed(noise_seed)
        upper = torch.triu(torch.randn(H, H, generator=genN), diagonal=1)
        T = T + s_nn * lam_max * upper
    return Q @ T @ Q.T, Q, T, eigs


def fixed_P0(H, Q):
    genR = torch.Generator().manual_seed(FRAME_SEED)
    A = torch.randn(H, H, generator=genR)
    R0, _ = torch.linalg.qr(A)
    return Q.T @ R0


def thetas_for(H, arm, theta, eigs):
    '''Budget-matched Fehlwinkel je Arm.'''
    if arm == 'UNI':
        return torch.full((H,), theta)
    if arm == 'SPEC-L':
        w = (eigs[0].item() - eigs.abs())
        w = w.clamp_min(1e-9)
    elif arm == 'SPEC-H':
        w = eigs.abs().clamp_min(1e-9)
    elif arm == 'CONC2':
        th = torch.full((H,), theta)  # kleiner Basis-Rest, keine Symmetrie
        th[1] = min(H * theta, 1.5)
        return th
    else:
        raise ValueError(arm)
    w = w / w.sum()
    return theta * H * w


def frame_F(H, Q, arm, theta, eigs):
    P0 = fixed_P0(H, Q)
    th = thetas_for(H, arm, theta, eigs)
    M = torch.diag(torch.cos(th)) + torch.diag(torch.sin(th)) @ P0
    R_y = torch.linalg.qr(M)[0]
    return Q @ R_y


def rollout_states(H, W, F, Uw, x_seq):
    B, Tn, _ = x_seq.shape
    z = torch.zeros(B, H)
    states = []
    for t in range(Tn):
        x_t = x_seq[:, t]
        for _ in range(SUBSTEPS):
            def f(zz):
                return -zz + torch.tanh(zz @ F) @ F.T @ W.T + x_t @ Uw.T
            k1 = f(z)
            k2 = f(z + 0.5 * DT * k1)
            z = z + DT * k2
        states.append(z)
    return torch.stack(states, dim=1)


def memory_capacity(states, x_seq, delays):
    H = states.shape[-1]
    z = states[0]
    u = x_seq[0, :, 0]
    total = 0.0
    for k in delays:
        rows = slice(WASHOUT + k, u.shape[0])
        S = z[rows]
        y = u[WASHOUT:u.shape[0] - k]
        A = S.T @ S + RIDGE * torch.eye(H)
        wout = torch.linalg.solve(A, S.T @ y)
        pred = S @ wout
        pc = pred - pred.mean()
        yc = y - y.mean()
        den = torch.sqrt((pc * pc).sum() * (yc * yc).sum())
        if den > 0:
            total += float(((pc * yc).sum() / den) ** 2)
    return total


def mstats(vals):
    ok = [v for v in vals if v == v and abs(v) < 1e6]
    st = float(torch.tensor(ok).std()) if len(ok) > 1 else float('nan')
    mean = float(torch.tensor(ok).mean()) if ok else float('nan')
    return mean, st, len(vals) - len(ok)


def main():
    out_path = 'results/g10_resource_v2_results.txt'
    csv_path = 'results/g10_resource_v2_table.csv'
    os.makedirs('results', exist_ok=True)
    seeds = list(range(20))
    x_mc = drive_input(1, T_TOTAL, IS)

    L = []
    L.append('=' * 118)
    L.append('G10 — RESSOURCEN-VERTEILER v2 (registriert: PASS nur wenn P1 UNI>=1.05*CONC2 in')
    L.append('>=3/4 Zellen UND P2 max(SPEC-L,SPEC-H)>=1.05*UNI in >=2/4 Zellen; 20 Seeds,')
    L.append('Budget-matched, Kalibrierung auf 3 Seeds, LETZTER VERSUCH DIESER THESE)')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            # Kalibrierung theta* auf Seeds 0,1,2 (Mittel-MC), UNI-Familie
            best_th, best_val = None, -1e9
            for th in THETA_GRID:
                vals = []
                for sd in CAL_SEEDS:
                    W, Q, T, eigs = factored_WQT(H, 1.0, s_nn, sd, FIXED, FIXED)
                    Uw = Q[:, 0:1].contiguous()
                    F = frame_F(H, Q, 'UNI', th, eigs)
                    vals.append(memory_capacity(rollout_states(H, W, F, Uw, x_mc),
                                                x_mc, DELAYS))
                m = float(torch.tensor(vals).mean())
                if m == m and m > best_val:
                    best_th, best_val = th, m
            L.append('Kalibrierung theta*=%.2f (Seeds 0-2, UNI): MC-mittel=%.4f'
                     % (best_th, best_val))

            block = {}
            for arm in ('UNI', 'SPEC-L', 'SPEC-H', 'CONC2'):
                mcs = []
                for seed in seeds:
                    W, Q, T, eigs = factored_WQT(H, 1.0, s_nn, seed, FIXED, FIXED)
                    Uw = Q[:, 0:1].contiguous()
                    F = frame_F(H, Q, arm, best_th, eigs)
                    mcs.append(memory_capacity(rollout_states(H, W, F, Uw, x_mc),
                                                x_mc, DELAYS))
                mean, st, bad = mstats(mcs)
                block[arm] = dict(mean=mean, std=st, bad=bad)
                L.append('Arm %s: MC mean=%.4f std=%.4f ungueltig=%d'
                         % (arm, mean, st, bad))
                rows.append(dict(H=H, s=s_nn, arm=arm, theta=best_th, mean=mean,
                                 std=st, n=len(seeds), bad=bad))
            cell_stats[(H, s_nn)] = dict(block=block, theta=best_th)
            Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,theta,mean_MC,std_MC,n,bad\n')
        for r in rows:
            fh.write('%d,%s,%s,%.2f,%.6g,%.6g,%d,%d\n'
                     % (r['H'], r['s'], r['arm'], r['theta'], r['mean'],
                        r['std'], r['n'], r['bad']))

    L.append('')
    L.append('--- G10-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    p1_cells = 0
    p2_cells = 0
    dir_h = {'L': 0, 'H': 0}
    for (H, s_nn), c in sorted(cell_stats.items()):
        b = c['block']
        mU, mL, mHh, mC = b['UNI']['mean'], b['SPEC-L']['mean'], \
            b['SPEC-H']['mean'], b['CONC2']['mean']
        p1 = (mU == mU and mC == mC and mU >= 1.05 * mC)
        p2 = (max(mL, mHh) >= 1.05 * mU)
        p1_cells += 1 if p1 else 0
        p2_cells += 1 if p2 else 0
        richt = 'L>H' if mL > mHh else 'H>L'
        dir_h['L' if richt == 'L>H' else 'H'] += 1
        L.append('Zelle H=%d s=%s (theta*=%.2f): UNI=%.4f SPEC-L=%.4f SPEC-H=%.4f '
                 'CONC2=%.4f | P1:%s P2:%s Richtung:%s'
                 % (H, s_nn, c['theta'], mU, mL, mHh, mC,
                    'JA' if p1 else 'NEIN', 'JA' if p2 else 'NEIN', richt))
        L.append('  SEKUNDAER: std UNI=%.4f SPEC-L=%.4f SPEC-H=%.4f CONC2=%.4f'
                 % (b['UNI']['std'], b['SPEC-L']['std'], b['SPEC-H']['std'],
                    b['CONC2']['std']))
    L.append('P1-Bilanz: %d/4 Zellen; P2-Bilanz: %d/4 Zellen; Richtung: L %d, H %d'
             % (p1_cells, p2_cells, dir_h['L'], dir_h['H']))

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if gate_bad:
        verdict = 'UNENTSCHEIDEN — zu viele ungueltige Punkte.'
    elif p1_cells >= 3 and p2_cells >= 2:
        verdict = ('PASS — Der Anker ist ein spektral-designbarer Ressourcen-'
                   'Verteiler: Konzentration schadet (P1), und die FLACHE '
                   'Verteilung ist schlagbar (P2) — Budget, Verteilung UND '
                   'Richtung sind Knoepfe. G11: Richtungsoptimum suchen und '
                   'Vorhersagekraft (Ranking aus billiger Groesse) pruefen.')
    else:
        verdict = ('KILL — Verteiler-These v2 endgueltig gefallen (letzter '
                   'Versuch war registriert): entweder flach ist optimal oder '
                   'Konzentration schadet nicht sauber. G8 bleibt als '
                   'Parameter-Befund stehen, der Mechanismus dahinter ist '
                   'nicht designbar. Kapitel schliessen.')
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + out_path + ' und ' + csv_path + ' geschrieben.')


if __name__ == '__main__':
    main()
