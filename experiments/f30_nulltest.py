'''
F30 — NULL-TEST DER METRIK: Misst Memory Capacity etwas, das echte
Task-Adaption vorhersagt? Der schnellste Kill des Projekts.

ANGRIFF (Tiefenanalyse IX, Saeule 1):
  2607.17858: nominelle Kapazitaet kann in schwachen Moden liegen
    und in der Praxis schlechter sein.
  2405.06561: lineare-Readout-Metriken fragil, NARMA vorsichtig.
  Jaeger-Vermutung: lineare Netze optimal fuer MC — kappa-Vergleiche
    koennen Nichtlinearitaet confoundieren.

TEST: Grid (lam x s), pro Konfiguration DREI Groessen am selben Netz:
  (a) MC (Standard, lineare Readout, max_lag 16) — unsere Metrik
  (b) NLM: nichtlineare Gedaechtnis-Readout (Ridge auf Zustände)
      fuer zusaetzliche nichtlineare Features des Inputs
  (c) TASK: echte Adaptionszeit (Steps bis Loss-Schwelle) beim
      sinus_memory-Training des NETZES (nicht nur Readouts)

KILL-Kriterium (vorab festgelegt, keine Nachverhandlung):
  corr(MC, TASK) < +0.4 => 'Ressource' ist nicht task-validiert;
  die Papier-Basis schrumpft auf deskriptive Dynamik-Aussagen.
  corr(MC, TASK) >= +0.4 => Saeule 1 steht, F24 darf als
  Ressourcen-Test laufen.
'''
import argparse
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max
from cgfn.continuous import make_continuous_batch
from cgfn.design import make_designed_model
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity

TASK = 'sinus_memory'
THRESH = 0.02


@torch.no_grad()
def nonlinear_memory(m, seed, seq_len=96, n_seq=40, max_lag=8):
    '''Ridge-Readout auf Zustände: Ziel = nichtlineare Funktion
    verzögerter Inputs (u_{t-1}*u_{t-2}, u_{t-3}*u_{t-5}, ...).'''
    torch.manual_seed(seed + 31337)
    gen = torch.Generator().manual_seed(seed + 4242)
    feats, targets = [], []
    for _ in range(n_seq):
        u = torch.randn(seq_len, 1, generator=gen) * 0.5
        _, states = m(u.unsqueeze(0), None)
        Z = states[0]  # (T, H)
        feats.append(Z)
        targets.append(torch.stack([u[:, 0] * torch.roll(u[:, 0], 1, 0),
                                    u[:, 0] * torch.roll(u[:, 0], 2, 0),
                                    u[:, 0] * torch.roll(u[:, 0], 4, 0)], dim=1))
    F = torch.cat(feats, 0)
    T = torch.cat(targets, 0)
    n = F.shape[0] // 2
    Ftr, Ttr, Fte, Tte = F[:n], T[:n], F[n:], T[n:]
    # Ridge (geschlossene Loesung, klein und deterministisch)
    lam_r = 1e-2
    A = Ftr.T @ Ftr + lam_r * torch.eye(Ftr.shape[1])
    Wout = torch.linalg.solve(A, Ftr.T @ Ttr)
    pred = Fte @ Wout
    denom = ((Tte - Tte.mean(0)) ** 2).mean(0).clamp_min(1e-9)
    r2 = 1.0 - ((Tte - pred) ** 2).mean(0) / denom
    return r2.mean().item()


def adapt_steps(m, seed, max_steps=200):
    torch.manual_seed(seed + 4242)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    lossfn = torch.nn.MSELoss()
    for i in range(max_steps):
        x, y = make_continuous_batch(TASK, 16, 16)
        out, _ = m(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < THRESH:
            return i + 1
    return max_steps


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--lam-values', default='0.7,0.9,1.1,1.3')
    p.add_argument('--s-values', default='0.0,0.8,1.6')
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--out', default='results/f30_nulltest_results.txt')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    lams = [float(v) for v in args.lam_values.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, 0.5)
    L = []
    L.append('=' * 112)
    L.append('F30 NULL-TEST: korreliert MC mit NICHTLINEAREM Gedaechtnis und echter Task-Adaption?')
    L.append(f'Grid lam={lams} x s={s_vals} seeds={seeds} | KILL: corr(MC,TASK) < +0.4')
    L.append('=' * 112)

    rows = []
    L.append(f"{'lam':>5s} {'s':>4s} {'seed':>4s} {'ben':>7s} {'MC':>6s} {'NLM_R2':>7s} {'adapt':>6s}")
    for lam in lams:
        for s in s_vals:
            for seed in seeds:
                m = make_designed_model(args.hidden, 1.0, lam, s, seed, task_dim=0)
                ben = benettin_ftle_max(m, xdr)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=0.5)
                nlm = nonlinear_memory(m, seed)
                m2 = make_designed_model(args.hidden, 1.0, lam, s, seed, task_dim=0)
                ad = adapt_steps(m2, seed)
                L.append(f'{lam:5.2f} {s:4.1f} {seed:4d} {ben:+7.3f} {mc:6.2f} {nlm:7.3f} {ad:6d}')
                rows.append((lam, s, seed, ben, mc, nlm, ad))

    mc = torch.tensor([r[4] for r in rows])
    nlm = torch.tensor([r[5] for r in rows])
    ad = torch.tensor([float(r[6]) for r in rows])
    ben = torch.tensor([r[3] for r in rows])

    def corr(a, b):
        if a.std() < 1e-6 or b.std() < 1e-6:
            return float('nan')
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    c_mn = corr(mc, nlm)
    c_ma = corr(mc, -ad)  # weniger Steps = besser; negativ drehen
    c_na = corr(nlm, -ad)
    c_ba = corr(ben, -ad)
    L.append('')
    L.append(f'corr(MC, NLM_R2)      = {c_mn:+.3f}   (Ressource soll nichtlineares Gedaechtnis tragen)')
    L.append(f'corr(MC, TASK-Speed)  = {c_ma:+.3f}   <<< KILL-GROESSE (>= +0.4 noetig)')
    L.append(f'corr(NLM, TASK-Speed) = {c_na:+.3f}')
    L.append(f'corr(ben, TASK-Speed) = {c_ba:+.3f}   (Kante/Fenster-Vergleich)')
    verdict = 'METRIK HAELT — Ressource task-validiert' if c_ma >= 0.4 else 'METRIK FAELLT — Ressource nicht task-validiert'
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
