#!/usr/bin/env python
"""F27 — Optimierer-Ressourcen-Interferenz (E3b-Antagonist, Run-35-Fassung).

Frage: Erodiert der Optimierer die Nicht-Normalitaets-Ressource?
Drift von kappa(W) (kappa_T, Tiefenanalyse XI) unter SGD, SGD+Momentum,
Adam auf einer Memory-Regression, multi-seed, W trainierbar.

Fix-Historie:
  Run 29/31: Import-/Syntax-/kappa_of-Todsuenen, Rauschen als Ziel —
  Rewrite mit ECHTER Memory-Regression.
  Run 33: linalg.svd-Crash bei NaN-W — isfinite-Waechter, DIVERGED-
  Marke, letzter endlicher kappa zaehlt.
  Run 34: SGD/Momentum (lr=0.05) divergierten in 4-7 Schritten — die
  Arme waren lr-vergiftet, kein fairer Vergleich. Korrektur meines
  eigenen Fehlablesens: die scheinbar 'kaputte' kT_start-Spalte war
  KORREKT (kz0 wird VOR dem ersten Schritt gemessen; designed
  startet bei kT ~6.5e7 und der ERSTE Schritt kollabiert es auf
  ~1.6e3). Run 35: lr-Sweep pro Optimierer bis unter die Divergenz-
  kante, 5 Seeds, t=-1-Zeile (kappa vor Schritt 1), Phase-2-Statistik
  (kT-Minimum, dessen Zeitpunkt, Verhaeltnis Ende/Minimum).

Run-34-Vorab (Adam, 3 Seeds): zweiphasig — Phase 1 Erosion (kT
stuerzt ab), Phase 2 spaeter Wiederanstieg bei weiter fallendem Loss
(Seed 2: kcond 1307 -> 5163 ab t~240). Run 35 testet, ob der Knick
robust ist und ob er auch unter SGD/Momentum auftritt (dann waere er
Update-Richtung, nicht Adaptivitaet).
Vorhersage (E3b-Original): kappa_T driftet monoton auf, Adam am
staerksten. Run 34: GEFALLEN (statt monotonem Drift: Erosion dann
Wiederaufbau). Run 35 entscheidet, ob das stabil ist.
"""
import argparse
import math
import os
from pathlib import Path

import torch

from cgfn.design import kappa_of, make_designed_model
from cgfn.resource import kappa_zoo


def _params_finite(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def run_arm(opt_name, lr, steps, seed, hidden, T, lag, log_every):
    torch.manual_seed(seed)
    model = make_designed_model(hidden, 1.0, 1.0, 0.8, seed,
                                task_dim=0, freeze_w=False)
    if opt_name == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'momentum':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(opt_name)

    gen = torch.Generator().manual_seed(seed + 1000)
    x = torch.randn(1, T, 1, generator=gen) * 0.5
    y = torch.zeros_like(x)
    y[0, lag:] = x[0, :-lag]

    kz0 = kappa_zoo(model.W.detach())
    kz_last = dict(kz0)
    # t=-1-Zeile: kappa VOR dem ersten Schritt (Run-34-Lehre:
    # designed startet bei kT ~1e7; der erste Schritt kollabiert es).
    log = [(-1, float('nan'), float(kz0['kappa_T']),
            float(kz0['kappa_cond']), float(kz0['delta_H']))]
    diverged = -1
    for t in range(steps):
        opt.zero_grad()
        out, _states = model(x)
        loss = ((out[0, lag:] - y[0, lag:]) ** 2).mean()
        loss.backward()
        opt.step()
        lw = loss.item()
        finite = math.isfinite(lw) and _params_finite(model)
        if t % log_every == 0 or t == steps - 1 or not finite:
            if finite:
                kz = kappa_zoo(model.W.detach())
                kz_last = kz
                log.append((t, lw, float(kz['kappa_T']),
                            float(kz['kappa_cond']),
                            float(kz['delta_H'])))
            else:
                log.append((t, lw, float(kz_last['kappa_T']),
                            float(kz_last['kappa_cond']),
                            float(kz_last['delta_H'])))
        if not finite:
            diverged = t
            break
    if diverged < 0:
        kz1 = kappa_zoo(model.W.detach())
        k_final = float(kappa_of(model.W.detach()))
    else:
        kz1 = kz_last
        k_final = float('nan')
    return log, kz0, kz1, k_final, diverged


def _survives(opt_name, lr, steps, seeds, hidden, T, lag):
    """Probe: ueberlebt der Optimierer 'steps' Schritte auf ALLEN Seeds?"""
    for seed in seeds:
        _log, _kz0, _kz1, _k, diverged = run_arm(
            opt_name, lr, steps, seed, hidden, T, lag, steps + 1)
        if diverged >= 0:
            return False
    return True


def main():
    p = argparse.ArgumentParser(description='F27 optimizer interference')
    p.add_argument('--seeds', default='0,1,2,3,4')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--lag', type=int, default=8)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--probe-steps', type=int, default=100)
    p.add_argument('--out', default='results/f27_optimizer_interference_results.txt')
    args = p.parse_args()

    seeds = [int(v) for v in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # lr-Kandidaten pro Optimierer (fallend). Gewaehlt wird die groesste
    # lr, die die Probe (probe_steps) auf allen Seeds uebersteht.
    candidates = {
        'sgd': [0.05, 0.01, 0.005, 0.002, 0.001, 0.0005],
        'momentum': [0.05, 0.01, 0.005, 0.002, 0.001, 0.0005],
        'adam': [0.01],
    }
    lrs = {}
    sweep_log = []
    for opt_name in ('sgd', 'momentum', 'adam'):
        chosen = None
        for lr in candidates[opt_name]:
            ok = _survives(opt_name, lr, args.probe_steps, seeds,
                           args.hidden, args.T, args.lag)
            sweep_log.append((opt_name, lr, 'OK' if ok else 'DIVERGED'))
            if ok:
                chosen = lr
                break
        if chosen is None:
            chosen = candidates[opt_name][-1]  # letzte Chance; wird divergieren
        lrs[opt_name] = chosen

    L = []
    L.append('=' * 118)
    L.append('F27 OPTIMIERER-RESSOURCEN-INTERFERENZ (Run 35): lr-Sweep, kappa(W)-Drift, Phase-2-Statistik')
    L.append(f'seeds={seeds} steps={args.steps} H={args.hidden} T={args.T} lag={args.lag} '
             f'probe_steps={args.probe_steps} designed: lam=1.0 s=0.8 tau=1.0 (W trainierbar)')
    L.append('t=-1-Zeile: kappa VOR dem ersten Schritt (designed startet bei kT ~1e7).')
    L.append('')
    L.append('--- lr-Sweep (groesste ueberlebende lr gewinnt) ---')
    for (o, lr, st) in sweep_log:
        L.append(f'  {o:>9s} lr={lr:<8g} probe: {st}')
    L.append(f'gewaehlt: {lrs}')
    L.append('')
    L.append('Run-34-Vorab: Adam zweiphasig (Erosion, dann Wiederanstieg bei fallendem Loss).')
    L.append('E3b-Original-Vorhersage (monotoner Aufwaertsdrift, Adam am staerksten): Run 34 GEFALLEN.')
    L.append('Run 35 testet: Robustheit des Phase-2-Knicks und ob er Update-Richtung oder Adam-Eigentum ist.')
    L.append('=' * 118)

    rows = []
    for opt_name in ('sgd', 'momentum', 'adam'):
        L.append('')
        L.append(f'########## {opt_name} (lr={lrs[opt_name]}) ##########')
        L.append(f"{'seed':>4s} {'t':>5s} {'loss':>9s} {'kT':>8s} {'kcond':>8s} {'dH':>7s} {'note':>9s}")
        for seed in seeds:
            log, kz0, kz1, _k, diverged = run_arm(opt_name, lrs[opt_name],
                                                  args.steps, seed,
                                                  args.hidden, args.T,
                                                  args.lag, args.log_every)
            note = 'DIVERGED' if diverged >= 0 else 'ok'
            for (t, loss, kT, kc, dH) in log:
                lv = (f'{loss:9.5f}' if math.isfinite(loss)
                      else '      --')
                L.append(f'{seed:4d} {t:5d} {lv} {kT:8.3f} {kc:8.3f} {dH:7.3f} {note:>9s}')
            # Phase-2-Statistik: Minimum ueber endliche Log-Punkte t>=0.
            pts = [(t, kT) for (t, _l, kT, _kc, _dH) in log
                   if t >= 0 and math.isfinite(kT)]
            if diverged >= 0:
                L.append(f'      [!] seed {seed}: DIVERGENZ bei t={diverged} '
                         f'(letzter endlicher kappa: '
                         f"kT {kz1['kappa_T']:.3f}, kcond {kz1['kappa_cond']:.3f})")
            if pts:
                tmin, kmin = min(pts, key=lambda p: p[1])
                kend = pts[-1][1]
                ratio = kend / max(kmin, 1e-9)
                L.append(f'      [p2] seed {seed}: kT_min={kmin:.3f} bei '
                         f't={tmin}, kT_end={kend:.3f}, ende/min={ratio:.3f}')
            rows.append(dict(opt=opt_name, seed=seed,
                             loss_end=log[-1][1],
                             kT0=float(kz0['kappa_T']), kT1=float(kz1['kappa_T']),
                             kc0=float(kz0['kappa_cond']), kc1=float(kz1['kappa_cond']),
                             diverged=diverged))

    L.append('')
    L.append('--- Aggregat (Mittel ueber Seeds; divergierte Arme zaehlen mit letztem endlichen kappa) ---')
    L.append(f"{'opt':>8s} {'loss_end':>9s} {'kT_start':>10s} {'kT_end':>9s} {'dK':>7s} {'ratio':>7s} {'kc_start':>10s} {'kc_end':>9s} {'n_div':>5s}")
    for opt_name in ('sgd', 'momentum', 'adam'):
        sub = [r for r in rows if r['opt'] == opt_name]
        n = len(sub)
        le = sum(r['loss_end'] if math.isfinite(r['loss_end']) else 0.0
                 for r in sub) / n
        k0 = sum(r['kT0'] for r in sub) / n
        k1 = sum(r['kT1'] for r in sub) / n
        c0 = sum(r['kc0'] for r in sub) / n
        c1 = sum(r['kc1'] for r in sub) / n
        ndiv = sum(1 for r in sub if r['diverged'] >= 0)
        L.append(f'{opt_name:>8s} {le:9.5f} {k0:10.3f} {k1:9.3f} {k1-k0:+7.3f} '
                 f'{k1/max(k0,1e-9):7.3f} {c0:10.3f} {c1:9.3f} {ndiv:5d}')
    L.append('(Hinweis: kT_start ist VOR dem ersten Schritt — identisch fuer alle')
    L.append(' Optimierer, da dieselben designed-Matrizen; der ERSTE Schritt')
    L.append(' kollabiert kT bereits um Groessenordnungen.)')

    L.append('')
    L.append('LESEAT:')
    L.append('Phase 1 (Erosion): kT stuerzt in den ersten Schritten ab — der')
    L.append(' Optimierer normalisiert die Nicht-Normalitaet zuerst weg.')
    L.append('Phase 2 (Wiederaufbau): ende/min > 1 systematisch => Betriebs-')
    L.append(' zustand hat EIGENE Dynamik, entkoppelt vom Loss (Kernthese).')
    L.append(' Tritt Phase 2 auch unter SGD/Momentum auf, ist es Update-')
    L.append(' Richtung; nur unter Adam: Adaptivitaets-Eigentum.')
    L.append('DIVERGED-Zaehler: Explosion ist Antagonisten-Datenpunkt.')
    L.append('Runde-XI-Note: schaerfere Groesse waere g_opt (F31-P5).')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
