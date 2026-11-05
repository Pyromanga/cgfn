'''
F36 — MODEN-SUPPLY-DEMAND-MATCH (MSDM): VORHERSAGE OHNE SKALAR, OHNE
TRAINING, AUS ZWEI GETRENNTEN MESSUNGEN (registriert 2026-11-04, VOR dem Lauf).

VORGESCHICHTE (1 Stein, 1 Frage, keine Nachverhandlung):
  F34 Phase 2: TEIL-PASS — delta_H 4/10, Range-Restriktion an BEIDEN Achsen.
  F35 (laeuft parallel, eigene Registrierung): Zwei-Knopf-Varianz-Restauration.
  Recherche 2026-11-04: Skalar-Aera ist vorbei. 2607.17858 zeigt mode-
  aufgeloeste IPC; PNAS 2424991122 macht spektrales Alignment (aggregiert,
  Frequenzraum, physikspezifisch). KEINER macht: Supply ohne Aufgabe,
  Demand ohne Reservoir, Vorhersage aus ihrem Moden-Match.

IDEE (der Wurf um die Ecke):
  LEISTUNG = geometrisches Treffen zweier unabhaengiger Spektren.
    DEMAND (pur, reservoir-frei, 1x je Aufgabe): Lag-Kern d(tau) der
      Aufgabe = Ridge-Koeffizienten von [u_t ... u_{t-15}] auf y_t,
      gemessen auf den exakten Task-Generatoren aus f34_exploration.
    SUPPLY (pur, aufgaben-frei, je Modell): Modalzerlegung von W.
      Modale Koordinaten a_k(t) = Re( V^{-1} x(t) )_k unter weisser
      Anregung; modalen Input-Antwortkern h_k(tau) =
      corr( u(t-tau), a_k(t) ), tau=0..15; Energieanteil E_k = Var(a_k)/Summe.
    MSDM = Summe_k E_k * <h_k_norm, d_norm>
  Nicht-Normalitaet geht PER MODE ein (transiente Verstaerkung faehrt
  genau die Moden, die Nachfrage bedienen koennen) — nicht als Skalar.
  Wenn MSDM stark vorhersagt, ist delta_Hs Teilerfolg mechanistisch
  ERKLAERT (Moden-Match-Anteil), nicht mystisch. Und der Homeostat
  wird ein per-Moden-Supply-Regler statt einer Skalar-Kruecke.

REGISTRIERUNG (VOR DEM LAUF GENAGELT):
  Gitter: H in {32,128} x band in {1,8} x s in {0.6,1.6} x seeds
          {20,21,22,23} (frisch) = 48 Modelle, Bisektion lam* auf
          ben=-0.30 (Import aus f35_twoknob — identischer Messpfad).
  Zellen: 2 H x 5 Aufgaben = 10, n=24 je Zelle.
  PRIMAER: corr(MSDM, TASK_R2) >= +0.6 in >= 7/10 gueltigen Zellen
           -> PASS: MSDM ist theory-anchoreder Betriebszustand;
              Homeostat-Ziel wird per-Moden Supply-Regler.
           < 7/10 -> KILL dieses Ansatzes (der innovative Wurf gilt
              als gefallen; dokumentieren, nicht wandeln).
           (Schwelle 0.6 statt 0.4: MSDM ist aufgaben-BEWUSST — ein
           aufgabenagnostischer Skalar wie delta_H muss schwaecher
           sein als ein Beobachter, der die Nachfrage kennt.)
  NUMERIK-GATE: MSDM = NaN fuer >25% Modelle einer Zelle -> Zelle
           ungueltig; >5 ungueltige Zellen -> UNENTSCHEIDEN
           (Eigen-Zerfall scheitert an defektiven band=1-Ketten).
  MANIPULATIONS-CHECK: |corr(ben1, TASK_R2)| < 0.4 in allen Zellen,
           sonst UNENTSCHEIDEN.
  DESKRIPTIV (keine Schwelle, nur Bericht): head-to-head MSDM vs
           delta_H vs MC je Zelle — zaehlt NICHT ins Verdikt.
  Demand-Lags: 16 (tau=0..15), Ridge 1e-2, einmal je Aufgabe mit
           festem Seed 123, 80 Sequenzen je Aufgabe.
  Supply-Drive: weisses Rauschen IS=0.5, T=1200, burn=200,
           fester Seed 777 (identisch fuer alle Modelle).
'''
import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import (task_sinus, task_narma10, task_white_delay8,
                             task_prod, task_trend, corr2, observables,
                             BEN_TARGET, IS)
from f35_twoknob import tune_lam_band

from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity

LAGS = 16
N_DEMAND_SEQ = 80
DEMAND_SEED = 123
SUPPLY_SEED = 777
SUPPLY_T = 1200
SUPPLY_BURN = 200


def task_datasets(n, sd):
    '''Exakte Replik der f34-Task-Generatoren: liefert je Sequenz
    (u-Sequenz, y-Ziel). Reservoir-frei.'''
    out = {'sinus': [], 'wdelay8': [], 'prod': [], 'trend': [], 'narma': []}
    gen = torch.Generator().manual_seed(sd)
    for _ in range(n):
        t = torch.arange(48, dtype=torch.float32)
        freq = 0.25 + 1.75 * torch.rand(1, 1, generator=gen)
        phase = 2 * math.pi * torch.rand(1, 1, generator=gen)
        x = torch.sin(freq * t.unsqueeze(0) + phase)
        y = x.clone(); y[:, 3:] = x[:, :-3]
        out['sinus'].append((x[0], y[0, 3:]))
        u = torch.randn(64, generator=gen) * IS
        out['wdelay8'].append((u, u[:-8].clone()))
        out['prod'].append((u, (u[8:] * u[8 - 3:-3]).clone()))
        y2 = torch.stack([u[t2 - 4:t2 + 1].mean() for t2 in range(8, 64)])
        out['trend'].append((u, y2))
        un = torch.rand(160, generator=gen) * 0.5
        y3 = torch.zeros(160)
        yh = torch.zeros(170)
        for tt in range(159):
            prod = 1.0
            for i in range(10):
                prod = prod * yh[10 + tt - i]
            un9 = un[tt - 9] if tt >= 9 else un[0]
            y3[tt + 1] = 0.3 * y3[tt] + 0.05 * y3[tt] * prod + 1.5 * un9 * un[tt] + 0.1
            yh[10 + tt + 1] = y3[tt + 1]
        out['narma'].append((un, y3[41:159].clone()))
    return out


def demand_kernel(seqs):
    '''Ridge von [u_t ... u_{t-15}] auf y_t ueber alle Sequenzen
    (reservoir-frei). Rueckgabe: L2-normalisierter Kern d_norm.'''
    F, Y = [], []
    for u, y in seqs:
        T = y.shape[0]
        u_pad = torch.cat([torch.zeros(LAGS - 1), u])   # u_pad[i] = u_{i-(LAGS-1)}
        cols = []
        for j in range(LAGS):
            # Spalte j: u_{t-j} fuer t = 0..T-1 (relativ zum Start von y)
            idx = torch.arange(LAGS - 1, LAGS - 1 + T) - j
            cols.append(u_pad[idx])
        F.append(torch.stack(cols, 1)); Y.append(y)
    F = torch.cat(F, 0).to(torch.float64); Y = torch.cat(Y, 0).to(torch.float64)
    A = F.T @ F + 1e-2 * torch.eye(LAGS, dtype=torch.float64)
    d = torch.linalg.solve(A, F.T @ Y)
    dn = d / d.norm().clamp_min(1e-12)
    return dn


@torch.no_grad()
def modal_supply(m):
    '''Modale Input-Antwortkerne unter weisser Anregung (aufgaben-frei).
    Rueckgabe: (E_k-Vector, H-Matrix [LAGS x n_modes]) oder None.'''
    W = m.W.detach().to(torch.float64)
    try:
        evals, V = torch.linalg.eig(W.to(torch.complex128))
        Vinv = torch.linalg.inv(V)
    except Exception:
        return None
    gen = torch.Generator().manual_seed(SUPPLY_SEED)
    u = torch.randn(SUPPLY_T, 1, generator=gen) * IS
    _, st = m(u.unsqueeze(0), None)
    X = st[0].to(torch.complex128)
    X = X[SUPPLY_BURN:]
    u0 = u[SUPPLY_BURN:, 0].to(torch.complex128)
    A = X @ Vinv.T
    Ar = A.real
    n_modes = Ar.shape[1]
    usd = u0.real.std()
    if usd < 1e-9:
        return None
    H_mat = torch.zeros(LAGS, n_modes, dtype=torch.float64)
    for tau in range(LAGS):
        a = Ar[tau:]
        uL = u0.real[:Ar.shape[0] - tau]
        num = ((a - a.mean(0)) * (uL - uL.mean()).unsqueeze(1)).mean(0)
        den = a.std(0).clamp_min(1e-12) * usd
        H_mat[tau] = num / den
    E = Ar.var(0)
    E = E / E.sum().clamp_min(1e-12)
    if not torch.isfinite(H_mat).all() or not torch.isfinite(E).all():
        return None
    return E, H_mat


def msdm(supply, d_norm):
    if supply is None:
        return float('nan')
    E, H_mat = supply
    tot = 0.0
    for k in range(H_mat.shape[1]):
        h = H_mat[:, k]
        hn = h / h.norm().clamp_min(1e-12)
        tot = tot + float(E[k]) * float(torch.dot(hn, d_norm.to(hn.dtype)))
    return tot


def main():
    p = argparse.ArgumentParser(description='F36 mode supply-demand match')
    p.add_argument('--seeds', default='20,21,22,23')
    p.add_argument('--bands', default='1,8')
    p.add_argument('--s-values', default='0.6,1.6')
    p.add_argument('--out', default='results/f36_msdm_results.txt')
    p.add_argument('--csv', default='results/f36_msdm_table.csv')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    bands = [int(v) for v in args.bands.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    tasks = {'sinus': task_sinus, 'narma': task_narma10,
             'wdelay8': task_white_delay8, 'prod': task_prod,
             'trend': task_trend}

    ds = task_datasets(N_DEMAND_SEQ, DEMAND_SEED)
    d_norm = {t: demand_kernel(ds[t]) for t in tasks}

    xdr = drive_input(4, 48, IS)
    L = []
    L.append('=' * 118)
    L.append('F36 — MODEN-SUPPLY-DEMAND-MATCH (registriert: MSDM >= +0.6 in >= 7/10 Zellen = PASS,')
    L.append('sonst KILL dieses Ansatzes; Numerik-Gate NaN>25% je Zelle; Manipulations-Check ben1 < 0.4)')
    L.append(f'Gitter: H in {{32,128}} x band={bands} x s={s_vals} x seeds={seeds}, Bisektion auf ben={BEN_TARGET}')
    L.append('=' * 118)

    rows = []
    for H in (32, 128):
        L.append('')
        L.append(f'########## H={H} ##########')
        for band in bands:
            for s_nn in s_vals:
                for seed in seeds:
                    lam, m = tune_lam_band(H, s_nn, seed, band, xdr)
                    supply = modal_supply(m)
                    obs = observables(m, xdr)
                    torch.manual_seed(seed + 9000)
                    mc, _ = memory_capacity(m, max_lag=16, seq_len=64,
                                             input_scale=IS)
                    obs['MC'] = mc
                    msdms = {t: msdm(supply, d_norm[t]) for t in tasks}
                    t2s = {t: tfn(m, seed) for t, tfn in tasks.items()}
                    rows.append(dict(H=H, band=band, s=s_nn, seed=seed,
                                     obs=obs, msdm=msdms, t2=t2s))
                    ms = ' '.join(f"{t}={msdms[t]:+.3f}" for t in tasks)
                    L.append(f"H={H} b={band} s={s_nn:.1f} seed={seed} "
                             f"lam*={lam:.3f} ben1={obs['ben1']:+.3f} "
                             f"dH={obs['delta_H']:.3f} MC={mc:.2f} MSDM[{ms}]")
        Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(args.csv, 'w', encoding='utf-8') as fh:
        head = ['H', 'band', 's', 'seed', 'ben1', 'delta_H', 'MC'] + \
               [f'msdm_{t}' for t in tasks] + [f't2_{t}' for t in tasks]
        fh.write(','.join(head) + '\n')
        for r in rows:
            ms = [f"{r['msdm'][t]:.6g}" if r['msdm'][t] == r['msdm'][t] else ''
                  for t in tasks]
            vals = [str(r['H']), str(r['band']), str(r['s']), str(r['seed']),
                    f"{r['obs']['ben1']:.6g}", f"{r['obs']['delta_H']:.6g}",
                    f"{r['obs']['MC']:.6g}"] + ms + \
                   [f"{r['t2'][t]:.6g}" for t in tasks]
            fh.write(','.join(vals) + '\n')

    L.append('')
    L.append('--- F36-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    hits = 0
    valid_cells = 0
    invalid_cells = 0
    ben_violations = []
    for H in (32, 128):
        for tname in tasks:
            sub = [r for r in rows if r['H'] == H]
            ms_vals = [r['msdm'][tname] for r in sub]
            tvals = [r['t2'][tname] for r in sub]
            n_nan = sum(1 for v in ms_vals if v != v)
            c = corr2(ms_vals, tvals)
            cben = corr2([r['obs']['ben1'] for r in sub], tvals)
            cdh = corr2([r['obs']['delta_H'] for r in sub], tvals)
            cmc = corr2([r['obs']['MC'] for r in sub], tvals)
            if n_nan > 0.25 * len(sub):
                L.append(f'Zelle H={H} {tname}: UNGUELTIG (MSDM NaN bei '
                         f'{n_nan}/{len(sub)} Modellen)')
                invalid_cells += 1
                continue
            valid_cells += 1
            if c == c and c >= 0.6:
                hits += 1
            if cben == cben and abs(cben) >= 0.4:
                ben_violations.append(f'H={H} {tname}: {cben:+.3f}')
            L.append(f'Zelle H={H} {tname}: corr(MSDM)={c:+.3f} '
                     f'[delta_H={cdh:+.3f}, MC={cmc:+.3f} deskriptiv] '
                     f'corr(ben1)={cben:+.3f}')

    L.append('')
    L.append(f'PRIMAER MSDM: {hits}/{valid_cells} gueltige Zellen >= +0.6   (>= 7 noetig)')
    L.append(f'Numerik-Gate: {invalid_cells} Zellen ungueltig (> 5 => UNENTSCHEIDEN)')
    if ben_violations:
        L.append('MANIPULATIONS-CHECK VERLETZT: ' + '; '.join(ben_violations))
    else:
        L.append('MANIPULATIONS-CHECK: ben1 ueberall < 0.4 — Matching haelt.')

    if ben_violations:
        verdict = ('UNENTSCHEIDEN — Stabilitaets-Matching durchbrochen.')
    elif invalid_cells > 5:
        verdict = ('UNENTSCHEIDEN — Numerik-Gate: Eigen-Zerfall scheitert '
                   'flaechig an defektiven Ketten.')
    elif hits >= 7:
        verdict = ('PASS — MSDM (Moden-Supply-Demand-Match) vorhersagt '
                   'Aufgabenleistung theory-basiert und aufgaben-bewusst. '
                   'Homeostat-Ziel: per-Moden Supply-Regler.')
    else:
        verdict = ('KILL — MSDM vorhersagt NICHT staerker als Skalare. Der '
                   'innovative Wurf ist gefallen: Supply-Demand-Trennung '
                   'traegt nicht. Dokumentieren, nicht wandeln.')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} und {args.csv} geschrieben.')


if __name__ == '__main__':
    main()
