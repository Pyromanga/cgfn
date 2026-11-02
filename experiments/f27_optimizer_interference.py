#!/usr/bin/env python
"""F27 — Optimierer-Ressourcen-Interferenz (E3b-Antagonist, fix 2026-11-02).

Frage: Erodiert der Optimierer die Nicht-Normalitaets-Ressource?
Wir messen den Drift von kappa(W) (Eigenvektor-Kondition kappa_T,
Tiefenanalyse XI) unter drei Optimierern (SGD, SGD+Momentum, Adam)
auf einer Memory-Regression (Ziel: um 'lag' verzoegerte weisse
Eingabe), multi-seed, W trainierbar (freeze_w=False).

Fix-Historie:
  Run 29/31: ModuleNotFoundError 'src' (import gefixt) — DANACH kamen
  zwei weitere Todsuenen ans Licht: (a) Oktober-Bulk-Edit-Truemmer in
  den f.write-Zeilen (Syntax-Fehler), (b) kappa_of existierte NICHT in
  cgfn.design, (c) Input-Shape (1000,64,16) passte nicht zum Modell
  (input_dim=1) und das Ziel war reines Rauschen. Dieser Rewrite fixt
  alle drei; Aufgabe ist jetzt eine ECHTE Memory-Regression, die die
  Ressource nutzt.

Vorhersage (E3b): kappa_T(W) driftet unter Training nach oben (F16:
+0.89 unter GRU); Adam am staerksten. Falls kein Drift: Der F16-Befund
ist kein Optimizer-Effekt.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.design import kappa_of, make_designed_model
from cgfn.resource import kappa_zoo


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
    log = []
    for t in range(steps):
        opt.zero_grad()
        out, _states = model(x)
        loss = ((out[0, lag:] - y[0, lag:]) ** 2).mean()
        loss.backward()
        opt.step()
        if t % log_every == 0 or t == steps - 1:
            kz = kappa_zoo(model.W.detach())
            log.append((t, loss.item(), kz['kappa_T'], kz['kappa_cond'],
                        kz['delta_H']))
    kz1 = kappa_zoo(model.W.detach())
    return log, kz0, kz1, float(kappa_of(model.W.detach()))


def main():
    p = argparse.ArgumentParser(description='F27 optimizer interference')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--lag', type=int, default=8)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--out', default='results/f27_optimizer_interference_results.txt')
    args = p.parse_args()

    seeds = [int(v) for v in args.seeds.split(',')]
    lrs = {'sgd': 0.05, 'momentum': 0.05, 'adam': 0.01}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    L = []
    L.append('=' * 110)
    L.append('F27 OPTIMIERER-RESSOURCEN-INTERFERENZ: kappa(W)-Drift unter SGD/Momentum/Adam auf Memory-Regression')
    L.append(f'seeds={seeds} steps={args.steps} H={args.hidden} T={args.T} lag={args.lag} '
             f'lrs={lrs} designed: lam=1.0 s=0.8 tau=1.0 (W trainierbar)')
    L.append('E3b-Vorhersage: kappa_T driftet auf; Adam am staerksten. Kein Drift => F16 ist kein Optimizer-Effekt.')
    L.append('=' * 110)

    rows = []
    for opt_name in ('sgd', 'momentum', 'adam'):
        L.append('')
        L.append(f'########## {opt_name} (lr={lrs[opt_name]}) ##########')
        L.append(f"{'seed':>4s} {'t':>5s} {'loss':>9s} {'kT':>8s} {'kcond':>8s} {'dH':>7s}")
        for seed in seeds:
            log, kz0, kz1, _k = run_arm(opt_name, lrs[opt_name], args.steps,
                                       seed, args.hidden, args.T, args.lag,
                                       args.log_every)
            for (t, loss, kT, kc, dH) in log:
                L.append(f'{seed:4d} {t:5d} {loss:9.5f} {kT:8.3f} {kc:8.3f} {dH:7.3f}')
            rows.append(dict(opt=opt_name, seed=seed,
                             loss_end=log[-1][1],
                             kT0=kz0['kappa_T'], kT1=kz1['kappa_T'],
                             kc0=kz0['kappa_cond'], kc1=kz1['kappa_cond']))

    L.append('')
    L.append('--- Aggregat (Mittel ueber Seeds) ---')
    L.append(f"{'opt':>8s} {'loss_end':>9s} {'kT_start':>9s} {'kT_end':>9s} {'dK':>7s} {'ratio':>7s} {'kc_start':>9s} {'kc_end':>9s}")
    for opt_name in ('sgd', 'momentum', 'adam'):
        sub = [r for r in rows if r['opt'] == opt_name]
        n = len(sub)
        le = sum(r['loss_end'] for r in sub) / n
        k0 = sum(r['kT0'] for r in sub) / n
        k1 = sum(r['kT1'] for r in sub) / n
        c0 = sum(r['kc0'] for r in sub) / n
        c1 = sum(r['kc1'] for r in sub) / n
        L.append(f'{opt_name:>8s} {le:9.5f} {k0:9.3f} {k1:9.3f} {k1-k0:+7.3f} '
                 f'{k1/max(k0,1e-9):7.3f} {c0:9.3f} {c1:9.3f}')

    L.append('')
    L.append('LESEAT:')
    L.append('dK > 0 systematisch => Optimierer erodiert die Nicht-Normalitaets-')
    L.append('Ressource (E3b bestaetigt); Vergleich der drei Optimierer sagt, ob')
    L.append('das an Adaptivitaet (Adam) oder an der Update-Richtung selbst liegt.')
    L.append('Runde-XI-Note: die schaerfere Groesse waere g_opt (Update-Kokkyklus,')
    L.append('F31-P5) — F27 misst den klassischen kappa(W)-Drift als F16-Replikat.')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
