'''
F30.4 — TRANSFER-KILL-TEST (F30.3-Generalisierung, vorab genagelt 2026-11-03).

Vorgeschichte: F30.3 PASS — corr(MC, TASK_R2) = +0.748 bei H=32,
EINER Aufgabe (sinus_memory), 48 Konfigs. Die einzige harte Zahl des
Projekts. Tiefenanalyse XIII + Selbstkritik: Das war ein Ein-Fenster-
Resultat; H=32-Designs und eine Aufgabe koennen eine Korrelation
tragen, die bei Transfer zerfaellt.

F30.4 prueft die zwei Transfer-Achsen, die die Wette toeten koennen:
    BEDINGUNG A (Skalierung): H=256 (8x), sinus_memory delay 3 —
        identisches Instrument wie F30.3 (task_readout_r2), nur H anders.
    BEDINGUNG B (Task-Familie): H=32, NARMA-10 — nichtlineare
        autoregressive Aufgabe, Standard-RC-Benchmark, andere
        Aufgabenart (nichtlinear + rekurrentes Ziel vs. linearer Delay).

VORAB FESTGENAGELT (vor dem ersten Lauf, keine Nachverhandlung):
    TRANSFER-PASS: corr(MC, TASK_R2) >= +0.4 in BEIDEN Bedingungen.
    TEIL-PASS: nur eine Bedingung >= +0.4 (Ein-Fenster-Risiko real).
    KILL: keine Bedingung >= +0.4 — F30.3 war Folklore des kleinen
          Gitters; Wette-Basis faellt, Pivot.
    Sekundaer je Bedingung (registriert): corr(NLM, TASK_R2) > 0,
        corr(ben, TASK_R2) < 0. Beide wie in F30.3.
    Deckel-Check je Bedingung: TASK_R2-std > 0.01, sonst ist das
        Instrument deckiert und die corr-Aussage wertlos.

Instrument: nur geschlossene Loesungen (Ridge), kein Adam, kein
Gradient — Antagonist bleibt ausgeschaltet (F30.3-Muster).
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
def sinus_readout_r2(m, seed, n_train=64, n_test=32, seq_len=48, ridge=1e-2):
    '''Identisch zu F30.3 task_readout_r2 (sinus_memory, delay 3).'''
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
def narma10_readout_r2(m, seed, n_train=50, n_test=25, seq_len=160,
                       washout=40, ridge=1e-2):
    '''NARMA-10, geschlossener Ridge-Readout — zweite Task-Familie.

    y_{t+1} = 0.3 y_t + 0.05 y_t * prod_{i=0..9} y_{t-i}
              + 1.5 u_{t-9} u_t + 0.1,  u ~ U[0, 0.5]
    Features: Zustaende nach Washout; Ziel: y_{t+1} (ein Schritt voraus).
    '''
    def collect(n, sd):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            u = torch.rand(seq_len, generator=gen) * 0.5
            y = torch.zeros(seq_len)
            yh = torch.zeros(seq_len + 10)  # yh[10+t] = y_t, t<0 => 0
            for t in range(seq_len - 1):
                prod = 1.0
                for i in range(10):
                    prod = prod * yh[10 + t - i]
                un = u[t - 9] if t >= 9 else u[0]
                y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * prod + 1.5 * un * u[t] + 0.1
                yh[10 + t + 1] = y[t + 1]
            _, states = m(u.unsqueeze(0).unsqueeze(-1), None)
            feats.append(states[0, washout:seq_len - 1])
            targs.append(y[washout + 1:])
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
    '''Wie F30.3: Ridge-Readout auf nichtlineare verz. Input-Features.'''
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


def corr_of(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float('nan')
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_condition(tag, hidden, task_fn, seeds, lams, s_vals, xdr, args):
    rows = []
    Lc = []
    Lc.append(f'########## BEDINGUNG {tag}: H={hidden}, task={task_fn.__name__} ##########')
    Lc.append(f"{'lam':>5s} {'s':>4s} {'seed':>4s} {'ben':>7s} {'MC':>6s} {'NLM_R2':>7s} {'TASK_R2':>8s}")
    for lam in lams:
        for s in s_vals:
            for seed in seeds:
                m = make_designed_model(hidden, 1.0, lam, s, seed, task_dim=0)
                ben = benettin_ftle_max(m, xdr)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=0.5)
                nlm = nonlinear_memory(m, seed)
                tr2 = task_fn(m, seed)
                Lc.append(f'{lam:5.2f} {s:4.1f} {seed:4d} {ben:+7.3f} {mc:6.2f} {nlm:7.3f} {tr2:+8.3f}')
                rows.append(dict(lam=lam, s=s, seed=seed, ben=ben,
                                 mc=mc, nlm=nlm, tr2=tr2))
    c_mt = corr_of([r['mc'] for r in rows], [r['tr2'] for r in rows])
    c_nt = corr_of([r['nlm'] for r in rows], [r['tr2'] for r in rows])
    c_bt = corr_of([r['ben'] for r in rows], [r['tr2'] for r in rows])
    tr2s = torch.tensor([r['tr2'] for r in rows], dtype=torch.float64)
    deckel = tr2s.std().item() > 0.01
    Lc.append('')
    Lc.append(f'Bedingung {tag}: corr(MC, TASK_R2)  = {c_mt:+.3f}   (>= +0.4 noetig)')
    Lc.append(f'Bedingung {tag}: corr(NLM, TASK_R2) = {c_nt:+.3f}   (registriert: > 0)')
    Lc.append(f'Bedingung {tag}: corr(ben, TASK_R2) = {c_bt:+.3f}   (registriert: < 0)')
    Lc.append(f'Bedingung {tag}: TASK_R2-Spannweite [{tr2s.min().item():+.3f}, '
              f'{tr2s.max().item():+.3f}] — Instrument deckiert: '
              f'{"NEIN" if deckel else "JA (corr-Aussage wertlos!)"}')
    return Lc, dict(tag=tag, c_mt=c_mt, c_nt=c_nt, c_bt=c_bt, deckel=deckel)


def main():
    p = argparse.ArgumentParser(description='F30.4 transfer kill test')
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--lam-values', default='0.7,0.9,1.1,1.3')
    p.add_argument('--s-values', default='0.0,0.8,1.6')
    p.add_argument('--out', default='results/f304_transfer_results.txt')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    lams = [float(v) for v in args.lam_values.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, 0.5)
    L = []
    L.append('=' * 112)
    L.append('F30.4 TRANSFER-KILL-TEST: F30.3-Generalisierung (H=256 Skalierung + NARMA-10 Task-Familie)')
    L.append(f'Grid lam={lams} x s={s_vals} seeds={seeds} | VORAB: TRANSFER-PASS nur wenn corr(MC, TASK_R2) >= +0.4 in BEIDEN Bedingungen')
    L.append('Bedingung A: H=256, sinus_memory (identisches Instrument wie F30.3)')
    L.append('Bedingung B: H=32, NARMA-10 (nichtlineare Task-Familie, Standard-RC-Benchmark)')
    L.append('Keine Nachverhandlung. Alle Ausgaenge sind Erkenntnis.')
    L.append('=' * 112)

    L_a, res_a = run_condition('A (H=256 sinus)', 256, sinus_readout_r2,
                               seeds, lams, s_vals, xdr, args)
    L += L_a
    # Snapshot nach Bedingung A (Run-21/33-Lehre).
    Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    L_b, res_b = run_condition('B (H=32 NARMA-10)', 32, narma10_readout_r2,
                               seeds, lams, s_vals, xdr, args)
    L += L_b

    passed = [r for r in (res_a, res_b) if r['c_mt'] == r['c_mt'] and r['c_mt'] >= 0.4 and r['deckel']]
    L.append('')
    L.append('--- TRANSFER-VERDICT (vorab genagelt: BEIDE >= +0.4 noetig) ---')
    L.append(f"Bedingung A: {'>=' if res_a['c_mt'] >= 0.4 else '<'} +0.4 (corr {res_a['c_mt']:+.3f})")
    L.append(f"Bedingung B: {'>=' if res_b['c_mt'] >= 0.4 else '<'} +0.4 (corr {res_b['c_mt']:+.3f})")
    if len(passed) == 2:
        verdict = ('TRANSFER-PASS — F30.3 generalisiert ueber Skalierung UND '
                  'Task-Familie. Ressource task-validiert ausserhalb des '
                  'Ein-Fenster-Betrachtungswinkels. Wette-Basis haelt.')
    elif len(passed) == 1:
        which = res_a['tag'] if res_a in passed else res_b['tag']
        verdict = (f'TEIL-PASS — nur Bedingung {which} >= +0.4. '
                   'Ein-Fenster-Risiko real: eine Achse traegt die '
                   'Korrelation, die andere nicht. Kein FULL-PASS der Wette-Basis.')
    else:
        verdict = ('KILL — keine Bedingung >= +0.4. F30.3 (+0.748) war '
                   'Folklore des kleinen Gitters/einer Aufgabe. Wette-Basis '
                   'faellt: Pivot auf deskriptive Dynamik.')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
