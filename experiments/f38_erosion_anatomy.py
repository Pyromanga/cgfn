#!/usr/bin/env python
'''F38 — EROSIONS-ANATOMIE: GREIFT DER OPTIMIERER GERICHTET AN?
(registriert 2026-11-05, VOR dem Lauf)

VORGESCHICHTE (F27, Run 34/35, hart und repliziert):
  designed W startet bei kappa_T ~1e7; der ERSTE Schritt kollabiert
  es um Groessenordnungen; spaeter zweiphasige Eigendynamik (Erosion,
  dann Wiederanstieg bei weiter fallendem Loss). E3b-Original
  (monotoner Aufwaertsdrift) GEFALLEN.
  Was FEHLT: die Anatomie. Weiss der Optimierer, WAS er zerstoert?
  kappa_T ist ein Skalar — F35-F37 haben gezeigt, dass Skalare
  Systemeigenschaften verfehlen. Der naechste Schritt ist nicht
  ein neuer Skalar, sondern die modale Aufloesung des F27-Befunds.

WERKZEUG (aus dem heutigen Methoden-Kapital):
  Original-Eigenbasis V von W_0 (t=-1, designed, s=0.8):
    lambda_k, rechte Eigenvektoren v_k (Spalten V),
    linke Eigenvektoren w_k (Zeilen V^-1).
  Modale Kondition g_k = ||v_k|| * ||w_k||  (>= 1; =1 iff normal) —
    per-Mode-Nicht-Normalitaet, der modale Baustein von kappa_T.
  Modaler Gain ueber die Training: D(t) = V^-1 W_t V;
    alpha_k(t) = Re(D_kk(t)).  Bei t=0 ist alpha_k = Re(lambda_k).
  Erosion (Phase 1):   e_k = 1 - alpha_k(t_eros)/Re(lambda_k)
    mit t_eros = Zeitpunkt des kappa_T-Minimums.
  Wiederaufbau (Phase 2): r_k = alpha_k(t_end) - alpha_k(t_eros).
  Spearman statt Pearson (g_k ist schwermaeenig — Rank-Korr).

REGISTRIERUNG (VOR DEM LAUF GENAGELT):
  Design: designed W (lam=1.0, s=0.8, tau=1.0, H=32, W trainierbar),
  Memory-Regression (T=64, lag=8) — IDENTISCH zu F27 run_arm.
  Optimierer: Adam lr=0.01 (5 Seeds 0-4) + Kontrast-Arme
  SGD/Momentum (groesste ueberlebende lr, F27-Sweep-Rezept).
  300 Steps, log alle 10, Divergenz-Waechter wie F27.
  PRIMAER (Adam): spearman(g_k, e_k) >= +0.4 in >= 4/5 Seeds
    -> PASS: Die Erosion ist GERICHTET — der Optimierer reisst
       zuerst die transient-gewinnstarken Moden ab. F27 bekommt
       eine Anatomie; die kappa_T-Erosion ist kein Beifang, sondern
       gezielte Zerstoerung von Ressourcen-Moden.
    < 4/5 -> KILL: Erosion ist ungerichtet/phaenomenologisch —
       F27 bleibt Kern-Befund ohne Mechanismus, wir hoeren auf,
       hinter ihm her zu graben.
  SEKUNDAER (Adam): spearman(e_k, r_k) >= +0.4 in >= 3/5 Seeds
    -> REKAPITULATION: Phase 2 baut genau die Moden wieder auf,
       die Phase 1 abriss (Wiederaufbau = Umkehrung, kein Neubau).
  KONTRAST (KEINE Schwelle, nur Bericht): SGD/Momentum mit
    gleicher modaler Statistik. Gleiche Richtung dort = Update-
    Geometrie-Eigentum; nur unter Adam = Adaptivitaets-Eigentum.
    (Ersetzt die offene F27-Entzensierung: die Kontrast-Arme
    antworten die gleiche Frage schaerfer.)
  GATES: Seed DIVERGED (F27-Waechter) -> Seed ungueltig.
    Moden mit |Re(lambda_k)| < 1e-6 oder Nicht-Diagonalisierbarkeit
    -> Modus ungueltig. >30% ungueltige Moden -> Seed ungueltig.
    < 3 gueltige Adam-Seeds -> UNENTSCHEIDEN (Design, keine Schwelle).
'''
import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgfn.design import make_designed_model
from cgfn.resource import kappa_zoo


def _params_finite(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def rank_corr(a, b):
    '''Spearman: Pearson auf Rangen; NaN-sicher (ungen zu NaNs).'''
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    ok = torch.isfinite(a) & torch.isfinite(b)
    if ok.sum() < 5 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
        return float('nan')
    ra = a[ok].argsort().argsort().to(torch.float64)
    rb = b[ok].argsort().argsort().to(torch.float64)
    c = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    return c


def modal_probe(W0):
    '''Original-Eigenbasis + g_k + Maske gueltiger Moden.
    Rueckgabe: dict oder None bei Nicht-Diagonalisierbarkeit.'''
    W64 = W0.detach().to(torch.complex128)
    try:
        lam, V = torch.linalg.eig(W64)
        Vinv = torch.linalg.inv(V)
    except Exception:
        return None
    right_norm = V.norm(dim=0)
    left_norm = Vinv.norm(dim=1)
    g = (right_norm * left_norm).real
    alpha0 = lam.real
    valid = (alpha0.abs() > 1e-6) & torch.isfinite(g) & (g > 0)
    return dict(lam=lam, V=V, Vinv=Vinv, g=g.real, alpha0=alpha0,
                valid=valid)


def alphas(probe, Wt):
    D = probe['Vinv'] @ Wt.detach().to(torch.complex128) @ probe['V']
    return torch.diagonal(D).real


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

    probe = modal_probe(model.W.detach())
    if probe is None:
        return None
    if probe['valid'].float().mean() < 0.7:
        return None
    valid = probe['valid']

    log = []          # (t, loss, kT, alpha_k bei gueltigen Moden)
    diverged = -1
    for t in range(steps):
        opt.zero_grad()
        out, _ = model(x)
        loss = ((out[0, lag:] - y[0, lag:]) ** 2).mean()
        loss.backward()
        opt.step()
        lw = loss.item()
        finite = math.isfinite(lw) and _params_finite(model)
        if t % log_every == 0 or t == steps - 1 or not finite:
            if finite:
                kz = kappa_zoo(model.W.detach())
                al = alphas(probe, model.W.detach())
                log.append((t, lw, float(kz['kappa_T']), al))
            if not finite:
                diverged = t
                break
    if not log:
        return None
    return dict(probe=probe, log=log, diverged=diverged, valid=valid)


def analyse(res):
    '''e_k, r_k, S (=spearman(g,e)), R (=spearman(e,r)) je Seed.'''
    probe, log, diverged, valid = res['probe'], res['log'], res['diverged'], res['valid']
    alpha0 = probe['alpha0'][valid]
    g = probe['g'][valid]
    # t_eros: Log-Punkt mit kappa_T-Minimum (nur endliche, t>=0)
    kts = [(t, kT) for (t, _l, kT, _al) in log if math.isfinite(kT)]
    t_eros, kT_min = min(kts, key=lambda p: p[1])
    al_end = log[-1][3][valid]
    al_eros = next(al for (t, _l, _kT, al) in log if t == t_eros)[valid]
    e = 1.0 - al_eros / alpha0
    r = al_end - al_eros
    S = rank_corr(g.tolist(), e.tolist())
    R = rank_corr(e.tolist(), r.tolist())
    return dict(t_eros=t_eros, kT_min=kT_min, S=S, R=R,
                loss_end=log[-1][1], diverged=diverged)


def survives(opt_name, lr, steps, seed, hidden, T, lag):
    res = run_arm(opt_name, lr, steps, seed, hidden, T, lag, steps + 1)
    return res is not None and res['diverged'] < 0


def main():
    p = argparse.ArgumentParser(description='F38 erosion anatomy')
    p.add_argument('--seeds', default='0,1,2,3,4')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--lag', type=int, default=8)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--probe-steps', type=int, default=100)
    p.add_argument('--out', default='results/f38_erosion_anatomy_results.txt')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    candidates = {'sgd': [0.05, 0.01, 0.005, 0.002, 0.001, 0.0005],
                  'momentum': [0.05, 0.01, 0.005, 0.002, 0.001, 0.0005],
                  'adam': [0.01]}
    lrs = {}
    sweep = []
    for opt_name in ('sgd', 'momentum', 'adam'):
        chosen = candidates[opt_name][-1]
        for lr in candidates[opt_name]:
            ok = survives(opt_name, lr, args.probe_steps, seeds[0],
                          args.hidden, args.T, args.lag)
            sweep.append((opt_name, lr, 'OK' if ok else 'DIVERGED'))
            if ok:
                chosen = lr
                break
        lrs[opt_name] = chosen

    L = []
    L.append('=' * 118)
    L.append('F38 — EROSIONS-ANATOMIE (registriert: Adam spearman(g_k, e_k) >= +0.4 in')
    L.append('>= 4/5 Seeds = PASS; Sekundaer Rekapitulation >= +0.4 in >= 3/5; Kontrast ohne Schwelle)')
    L.append(f'seeds={seeds} steps={args.steps} H={args.hidden} T={args.T} lag={args.lag}')
    L.append('designed: lam=1.0 s=0.8 tau=1.0 — IDENTISCH zu F27 (W trainierbar, Memory-Regression)')
    L.append('')
    L.append('--- lr-Sweep (F27-Rezept) ---')
    for (o, lr, st) in sweep:
        L.append(f'  {o:>9s} lr={lr:<8g} probe: {st}')
    L.append(f'gewaehlt: {lrs}')
    L.append('=' * 118)

    stats = {}
    for opt_name in ('adam', 'sgd', 'momentum'):
        L.append('')
        L.append(f'########## {opt_name} (lr={lrs[opt_name]}) ##########')
        arm = []
        for seed in seeds:
            res = run_arm(opt_name, lrs[opt_name], args.steps, seed,
                          args.hidden, args.T, args.lag, args.log_every)
            if res is None:
                L.append(f'seed {seed}: UNGUELTIG (Nicht-Diagonalisierbarkeit oder >30% Moden ungueltig)')
                arm.append(None)
                continue
            a = analyse(res)
            arm.append(a)
            note = 'DIVERGED' if a['diverged'] >= 0 else 'ok'
            L.append(f"seed {seed}: t_eros={a['t_eros']} kT_min={a['kT_min']:.3f} "
                     f"S(g,e)={a['S']:+.3f} R(e,r)={a['R']:+.3f} "
                     f"loss_end={a['loss_end']:.5f} {note}")
        stats[opt_name] = arm
        Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    L.append('')
    L.append('--- F38-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    adam_arm = [a for a in stats['adam'] if a is not None and a['diverged'] < 0]
    n_adam = len(adam_arm)
    hits_S = sum(1 for a in adam_arm if a['S'] == a['S'] and a['S'] >= 0.4)
    hits_R = sum(1 for a in adam_arm if a['R'] == a['R'] and a['R'] >= 0.4)
    L.append(f'Adam: {n_adam}/5 gueltige Seeds (>= 3 gefordert)')
    L.append(f'PRIMAER   Gerichtetheit S(g,e) >= +0.4: {hits_S}/{n_adam} Seeds   (>= 4 noetig)')
    L.append(f'SEKUNDAER Rekapitulation R(e,r) >= +0.4: {hits_R}/{n_adam} Seeds   (>= 3)')
    for opt_name in ('sgd', 'momentum'):
        sub = [a for a in stats[opt_name] if a is not None and a['diverged'] < 0]
        if not sub:
            L.append(f'Kontrast {opt_name}: keine gueltigen Seeds')
            continue
        sm = sum(a['S'] for a in sub if a['S'] == a['S']) / max(1, sum(1 for a in sub if a['S'] == a['S']))
        rm = sum(a['R'] for a in sub if a['R'] == a['R']) / max(1, sum(1 for a in sub if a['R'] == a['R']))
        L.append(f'Kontrast {opt_name}: mean S(g,e)={sm:+.3f}, mean R(e,r)={rm:+.3f} '
                 f'(KEINE Schwelle: Richtung dort = Update-Geometrie; nur Adam = Adaptivitaet)')

    if n_adam < 3:
        verdict = ('UNENTSCHEIDEN — zu wenige gueltige Adam-Seeds '
                   '(Divergenz/Modal-Gate), Design-Frage, keine Schwelle.')
    elif hits_S >= 4:
        verdict = ('PASS — die kappa_T-Erosion ist GERICHTET: der Optimierer '
                   'reisst zuerst die transient-gewinnstarken Moden ab. F27 '
                   'hat eine Anatomie; die Nicht-Normalitaets-Ressource wird '
                   'gezielt geopfert, nicht zufaellig abgenutzt.')
    else:
        verdict = ('KILL — die Erosion ist ungerichtet. kappa_T-Kollaps ist '
                   'ein Beifang, keine gezielte Ressourcen-Zerstoerung. '
                   'F27 bleibt phaenomenologischer Kern-Befund; wir graben '
                   'nicht weiter hinter ihm.')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append(f'REKAPITULATION (nur Bericht): {"BEJAHT" if hits_R >= 3 else "VERNEINT"} — Phase-2-Wiederaufbau '
             f'{"in denselben Moden" if hits_R >= 3 else "nicht modengebunden"} ({hits_R}/{n_adam} Seeds >= +0.4)')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
