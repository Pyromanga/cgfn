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
  (b) NLM: nichtlineare Gedaechtnis-Readout (Ridge auf Zustande)
      fuer zusaetzliche nichtlineare Features des Inputs
  (c) TASK: Lerngeschwindigkeit beim sinus_memory-Training

INSTRUMENTEN-FIX v2 (Run 24, 2026-11-02): Der erste Lauf zeigte
adapt=200 in ALLEN 48 Zeilen — Steps-bis-Schwelle(0.02) saturiert
(Deckel-Effekt), corr=nan, das 'VERDICT: METRIK FAELLT' war ein
Messinstrument-Artefakt, KEIN Befund. Fix: TASK-speed = logarithmische
Lernreduktion speed = log(loss_anfang/loss_ende) ueber 400 Steps —
varianzgarantiert, auch wenn die Schwelle nie erreicht wird.
Steps-bis-Schwelle bleibt als Sekundaerspalte sichtbar.
DAS KRITERIUM IST UNVERAENDERT: corr(MC, TASK-speed) >= +0.4
(vorab genagelt, keine Nachverhandlung — nur das Instrument wurde
gegen ein nicht-saturierendes getauscht).

KILL-Kriterium:
  corr(MC, TASK) < +0.4 => 'Ressource' ist nicht task-validiert;
  die Papier-Basis schrumpft auf deskriptive Dynamik-Aussagen.
  corr(MC, TASK) >= +0.4 => Saeule 1 steht, F24 darf als
  Ressourcen-Test laufen.
'''
import argparse
import math
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
    '''Ridge-Readout auf Zustande: Ziel = nichtlineare Funktion
    verzogerter Inputs (u_{t-1}*u_{t-2}, u_{t-3}*u_{t-5}, ...).'''
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
    lam_r = 1e-2
    A = Ftr.T @ Ftr + lam_r * torch.eye(Ftr.shape[1])
    Wout = torch.linalg.solve(A, Ftr.T @ Ttr)
    pred = Fte @ Wout
    denom = ((Tte - Tte.mean(0)) ** 2).mean(0).clamp_min(1e-9)
    r2 = 1.0 - ((Tte - pred) ** 2).mean(0) / denom
    return r2.mean().item()


def train_curve(m, seed, steps=400, batch=16, seq_len=24, lr=3e-3):
    '''Lernkurve auf sinus_memory. Rueckgabe:
       (speed, steps2thresh, loss0, loss_end)
       speed = log(mean(loss[0:5]) / mean(loss[-5:]))  — varianzgarantiert.
    '''
    torch.manual_seed(seed + 4242)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    lossfn = torch.nn.MSELoss()
    losses = []
    steps2thresh = -1
    for i in range(steps):
        x, y = make_continuous_batch(TASK, batch, seq_len)
        out, _ = m(x)
        loss = lossfn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = loss.item()
        losses.append(v)
        if steps2thresh < 0 and v < THRESH:
            steps2thresh = i + 1
    loss0 = sum(losses[:5]) / 5.0
    loss_end = sum(losses[-5:]) / 5.0
    speed = math.log(loss0 / max(loss_end, 1e-9))
    return speed, steps2thresh, loss0, loss_end


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--lam-values', default='0.7,0.9,1.1,1.3')
    p.add_argument('--s-values', default='0.0,0.8,1.6')
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--out', default='results/f30_nulltest_results.txt')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    lams = [float(v) for v in args.lam_values.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, 0.5)
    L = []
    L.append('=' * 112)
    L.append('F30 NULL-TEST v2: korreliert MC mit NICHTLINEAREM Gedaechtnis und echter Task-Adaption?')
    L.append(f'Grid lam={lams} x s={s_vals} seeds={seeds} steps={args.steps} | KILL: corr(MC,TASK-speed) < +0.4')
    L.append('Instrument v2: TASK-speed = log(loss0/loss_end) — nicht-saturierend (Run 24: adapt=200 konstant, Artefakt)')
    L.append('=' * 112)

    rows = []
    L.append(f"{'lam':>5s} {'s':>4s} {'seed':>4s} {'ben':>7s} {'MC':>6s} {'NLM_R2':>7s} {'speed':>7s} {'s2thr':>6s} {'loss0':>7s} {'lossE':>7s}")
    for lam in lams:
        for s in s_vals:
            for seed in seeds:
                m = make_designed_model(args.hidden, 1.0, lam, s, seed, task_dim=0)
                ben = benettin_ftle_max(m, xdr)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=0.5)
                nlm = nonlinear_memory(m, seed)
                m2 = make_designed_model(args.hidden, 1.0, lam, s, seed, task_dim=0)
                speed, s2t, l0, le = train_curve(m2, seed, steps=args.steps)
                L.append(f'{lam:5.2f} {s:4.1f} {seed:4d} {ben:+7.3f} {mc:6.2f} {nlm:7.3f} '
                         f'{speed:+7.3f} {s2t:6d} {l0:7.4f} {le:7.4f}')
                rows.append((lam, s, seed, ben, mc, nlm, speed, s2t))

    mc = torch.tensor([r[4] for r in rows])
    nlm = torch.tensor([r[5] for r in rows])
    speed = torch.tensor([r[6] for r in rows])
    ben = torch.tensor([r[3] for r in rows])
    s2t = torch.tensor([float(max(r[7], 0)) for r in rows])

    def corr(a, b):
        if a.std() < 1e-6 or b.std() < 1e-6:
            return float('nan')
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    c_mn = corr(mc, nlm)
    c_ma = corr(mc, speed)
    c_na = corr(nlm, speed)
    c_ba = corr(ben, speed)
    L.append('')
    L.append(f'corr(MC, NLM_R2)      = {c_mn:+.3f}   (Ressource soll nichtlineares Gedaechtnis tragen)')
    L.append(f'corr(MC, TASK-speed)  = {c_ma:+.3f}   <<< KILL-GROESSE (>= +0.4 noetig)')
    L.append(f'corr(NLM, TASK-speed) = {c_na:+.3f}')
    L.append(f'corr(ben, TASK-speed) = {c_ba:+.3f}   (Kante/Fenster-Vergleich)')
    n_satur = int((s2t <= 0).sum())
    L.append(f'Schwellen-Reichweite: {n_satur}/{len(rows)} Zeilen ohne Threshold (Sekundaer-Mass, speed unabhaengig)')
    verdict = ('METRIK HAELT — Ressource task-validiert'
              if c_ma == c_ma and c_ma >= 0.4 else
              'METRIK FAELLT — Ressource nicht task-validiert')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
