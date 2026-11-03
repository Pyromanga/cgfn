'''
F30.3 — NULL-TEST, OPTIMIZER-FREI (Tiefenanalyse XII, Ratte Nr. 5).

Vorgeschichte: F30 v2 (Run 26) mass 'MC sagt Adam-Adaption voraus' —
KILL (corr = -0.499). Zerlegung: speed war zu 99% der loss0-Kanal
(Anfangszustands-Explosion), lossE flach, 48/48 ohne Threshold. Und:
Messung DURCH den Optimierer ist Messung durch den Antagonisten (E3b).

F30.3 misst dasselbe Konstrukt ohne den Antagonisten:
    - Task: sinus_memory (delay 3, freq 0.25-2.0 rad/step), identisch
      zur Distribution aus cgfn.continuous
    - Ridge-Readout in GESCHLOSSENER FORM (kein Gradient, kein Adam,
      keine Lernrate) auf disjunkten Train/Test-Sequenzen
    - TASK_R2 = 1 - MSE/Var(Ziel) auf Testset

VORAB FESTGENAGELT (2026-11-03, vor dem ersten Lauf):
    KILL:  corr(MC, TASK_R2) < +0.4
    PASS:  corr(MC, TASK_R2) >= +0.4
    Sekundaer: corr(NLM, TASK_R2) > 0 erwartet; corr(ben, TASK_R2) < 0
    erwartet (Chaos schadet der Task-Leistung).
Keine Nachverhandlung. Beide Ausgaenge sind Erkenntnis:
    PASS  => Ressource task-validiert (optimizer-frei); F30-v2-Kill
             = Antagonisten-Kanal (E3b-Staerkung).
    KILL  => Ressource = Reservoir-Folklore; Pivot.
'''
import argparse
import math
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max
from cgfn.design import make_designed_model
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity

DELAY = 3


@torch.no_grad()
def task_readout_r2(m, seed, n_train=64, n_test=32, seq_len=48, ridge=1e-2):
    '''Ridge-Readout (geschlossen) auf sinus_memory; R2 auf Testset.

    Nur Positionen t >= DELAY (Delay-Ziel aktiv). Deterministisch
    ueber eigenen Generator — dieselbe Verteilung wie
    cgfn.continuous._sinus_memory, ohne globalem RNG.
    '''
    def collect(n, sd):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            t = torch.arange(seq_len, dtype=torch.float32)
            freq = 0.25 + 1.75 * torch.rand(1, 1, generator=gen)
            phase = 2 * math.pi * torch.rand(1, 1, generator=gen)
            x = torch.sin(freq * t.unsqueeze(0) + phase)  # (1, T)
            y = x.clone()
            y[:, DELAY:] = x[:, :-DELAY]
            _, states = m(x.unsqueeze(-1), None)
            feats.append(states[0, DELAY:])
            targs.append(y[0, DELAY:])
        return torch.cat(feats, 0), torch.cat(targs, 0)

    Ftr, Ytr = collect(n_train, seed + 555)
    Fte, Yte = collect(n_test, seed + 777)
    A = Ftr.T @ Ftr + ridge * torch.eye(Ftr.shape[1])
    W = torch.linalg.solve(A, Ftr.T @ Ytr.unsqueeze(1))
    pred = (Fte @ W).squeeze(1)
    denom = ((Yte - Yte.mean()) ** 2).mean().clamp_min(1e-9)
    r2 = 1.0 - ((Yte - pred) ** 2).mean() / denom
    return r2.item()


@torch.no_grad()
def nonlinear_memory(m, seed, seq_len=96, n_seq=40):
    '''Wie F30 v2: Ridge-Readout auf nichtlineare verz. Input-Features.'''
    gen = torch.Generator().manual_seed(seed + 4242)
    feats, targets = [], []
    for _ in range(n_seq):
        u = torch.randn(seq_len, 1, generator=gen) * 0.5
        _, states = m(u.unsqueeze(0), None)
        feats.append(states[0])
        u0 = u[:, 0]
        targets.append(torch.stack([u0 * torch.roll(u0, 1, 0),
                                    u0 * torch.roll(u0, 2, 0),
                                    u0 * torch.roll(u0, 4, 0)], dim=1))
    F = torch.cat(feats, 0)
    T = torch.cat(targets, 0)
    n = F.shape[0] // 2
    lam_r = 1e-2
    A = F[:n].T @ F[:n] + lam_r * torch.eye(F.shape[1])
    W = torch.linalg.solve(A, F[:n].T @ T[:n])
    pred = F[n:] @ W
    denom = ((T[n:] - T[n:].mean(0)) ** 2).mean(0).clamp_min(1e-9)
    r2 = 1.0 - ((T[n:] - pred) ** 2).mean(0) / denom
    return r2.mean().item()


def main():
    p = argparse.ArgumentParser(description='F30.3 optimizer-free null test')
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--lam-values', default='0.7,0.9,1.1,1.3')
    p.add_argument('--s-values', default='0.0,0.8,1.6')
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--out', default='results/f303_taskreadout_results.txt')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    lams = [float(v) for v in args.lam_values.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, 0.5)
    L = []
    L.append('=' * 112)
    L.append('F30.3 NULL-TEST OPTIMIZER-FREI: Ridge-Readout-Leistung (sinus_memory, delay 3) vs MC')
    L.append(f'Grid lam={lams} x s={s_vals} seeds={seeds} | VORAB: KILL falls corr(MC, TASK_R2) < +0.4')
    L.append('Ratte-5-Fix: kein Adam, kein Gradient — geschlossene Loesung, Antagonist ausgeschaltet.')
    L.append('=' * 112)

    rows = []
    L.append(f"{'lam':>5s} {'s':>4s} {'seed':>4s} {'ben':>7s} {'MC':>6s} {'NLM_R2':>7s} {'TASK_R2':>8s}")
    for lam in lams:
        for s in s_vals:
            for seed in seeds:
                m = make_designed_model(args.hidden, 1.0, lam, s, seed, task_dim=0)
                ben = benettin_ftle_max(m, xdr)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=0.5)
                nlm = nonlinear_memory(m, seed)
                tr2 = task_readout_r2(m, seed)
                L.append(f'{lam:5.2f} {s:4.1f} {seed:4d} {ben:+7.3f} {mc:6.2f} {nlm:7.3f} {tr2:+8.3f}')
                rows.append((lam, s, seed, ben, mc, nlm, tr2))

    mc = torch.tensor([r[4] for r in rows])
    nlm = torch.tensor([r[5] for r in rows])
    tr2 = torch.tensor([r[6] for r in rows])
    ben = torch.tensor([r[3] for r in rows])

    def corr(a, b):
        if a.std() < 1e-6 or b.std() < 1e-6:
            return float('nan')
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    c_mt = corr(mc, tr2)
    c_nt = corr(nlm, tr2)
    c_bt = corr(ben, tr2)
    L.append('')
    L.append(f'corr(MC, TASK_R2)    = {c_mt:+.3f}   <<< KILL-GROESSE (>= +0.4 noetig)')
    L.append(f'corr(NLM, TASK_R2)   = {c_nt:+.3f}   (registriert: > 0 erwartet)')
    L.append(f'corr(ben, TASK_R2)   = {c_bt:+.3f}   (registriert: < 0 erwartet)')
    L.append(f'TASK_R2 Spannweite: [{tr2.min().item():+.3f}, {tr2.max().item():+.3f}] — '
             f'Instrument hat Varianz: {"JA" if tr2.std().item() > 0.01 else "NEIN (Deckel-Verdacht!)"}')
    verdict = ('PASS — Ressource task-validiert (optimizer-frei); F30-v2-Kill = Antagonisten-Kanal (E3b-Staerkung)'
               if c_mt == c_mt and c_mt >= 0.4 else
               'KILL — Ressource = Reservoir-Folklore; Pivot auf deskriptive Dynamik')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
