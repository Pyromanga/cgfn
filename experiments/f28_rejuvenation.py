'''
F28 — REJUVENATION: Ist die dynamische Ressource (kappa, getriebenes
lam) KAUSAL fuer Plastizitaet? Der Kopf-an-Kopf-Test gegen die
Zufalls-Loesungen der Industrie.

Literatur-Spannung (Tiefenanalyse VII):
  Sutton (Nature 2024): Plastizitaet stirbt unter Gradient-Training;
    'sustained deep learning requires a random, non-gradient
    component' (Continual Backprop, Shrink & Perturb).
  UNSERE These: Was da erodiert, ist die dynamische Ressource in
    ihren Koordinaten (kappa, getriebenes lam). Dann muss GEZIELTE
    Wiederherstellung des Fensters Zufall schlagen — und Praevention
    (Homeostat) beide schlagen.

VORHERSAGEN:
  P1: Task-Serie erodiert Adaptionsfaehigkeit (Loss of Plasticity
      im CT-RNN) UND kappa/ben wandern aus dem Fenster.
  P2: Targeted Rejuvenation (Fenster-Rueckschub, IDENTISCHE
      Stoerungsnorm wie random) stellt Adaptionsfaehigkeit und MC
      besser wieder her als Zufall.
  P3: Durchlaufender Homeostat verliert Plastizitaet erst gar nicht.

DESIGN (4 Arme x 4 Seeds, 10 Bloecke a 150 Steps):
  Block-Tasks: sinus_memory/lorenz/ou_noise zyklisch (alle real,
  crash-Lehre beachtet).
  Nach Block 5 (Erosions-Haelfte): Arm-Intervention.
    A continue: keine
    B random:   NUR Gauss-Stoerung auf W
    C targeted: NUR Fenster-Restauration (Spektral-Renorm Richtung
                kappa0/lam0), Frobenius-Norm identisch zu B
    D homeo:    ab Block 1 Homeostat-Penalty (F21-Stil, alle 50 Steps)
  Metrik: Steps bis Loss-Schwelle im jeweiligen neuen Block
  (Plastizitaet), MC nach Block 10, kappa/ben-Trajektorie.
'''
import argparse
import os
from pathlib import Path

import torch

from cgfn.continuous import make_continuous_batch
from cgfn.benettin import benettin_ftle_max
from cgfn.design import make_designed_model
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity

TASKS = ['sinus_memory', 'lorenz', 'ou_noise']


def kappa_lam(W):
    sv = torch.linalg.svdvals(W)[0].item()
    lam = torch.linalg.eigvals(W).abs().max().item()
    return sv / max(lam, 1e-9), lam


def spectral_renorm(W, k_target, lam_target, step=0.5):
    '''Schiebe kappa und lam zurueck Richtung Ziel (normierende
    Skalierung + Symmetrisierungs-Mischung). Die Frobenius-Norm der
    Aenderung wird vom Aufrufer auf Arm-B-Norm genormt.'''
    k, lam = kappa_lam(W)
    scale = (lam_target / max(lam, 1e-9)) ** step
    W2 = W * scale
    k2, _ = kappa_lam(W2)
    if abs(k2 - k_target) > 0.05:
        alpha = 0.15
        W2 = (1 - alpha) * W2 + alpha * 0.5 * (W2 + W2.T)
    return W2


def adapt_steps(m, task, thresh, seed, max_steps=120):
    '''Plastizitaet: Steps bis Loss unter Schwelle.'''
    torch.manual_seed(seed + 4242)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    lossfn = torch.nn.MSELoss()
    for i in range(max_steps):
        x, y = make_continuous_batch(task, 16, 16)
        out, _ = m(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < thresh:
            return i + 1
    return max_steps


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--blocks', type=int, default=10)
    p.add_argument('--steps-per-block', type=int, default=150)
    p.add_argument('--thresh', type=float, default=0.02)
    p.add_argument('--out', default='results/f28_rejuvenation_results.txt')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, 0.5)
    L = []
    L.append('=' * 112)
    L.append('F28 REJUVENATION: Gezielte Fenster-Restauration vs Zufall (Shrink&Perturb-Aequivalent)')
    L.append(f'seeds={seeds} blocks={args.blocks} steps/block={args.steps_per_block} thresh={args.thresh}')
    L.append('P1: Serie erodiert | P2: targeted > random | P3: Homeostat praeventiert')
    L.append('=' * 112)

    agg = {}
    for arm in ('A', 'B', 'C', 'D'):
        for seed in seeds:
            torch.manual_seed(seed)
            m = make_designed_model(32, 1.0, 1.0, 0.8, seed, task_dim=0)
            k0, lam0 = kappa_lam(m.W.detach())
            ben0 = benettin_ftle_max(m, xdr)
            opt = torch.optim.Adam(m.parameters(), lr=2e-3)
            lossfn = torch.nn.MSELoss()
            adapt_hist = []
            k_hist = []
            for blk in range(args.blocks):
                task = TASKS[blk % 3]
                torch.manual_seed(seed + 100 * blk)
                a = adapt_steps(m, task, args.thresh, seed + 100 * blk)
                adapt_hist.append(a)
                torch.manual_seed(seed + 555 * blk)
                for _ in range(args.steps_per_block):
                    x, y = make_continuous_batch(task, 16, 16)
                    out, _ = m(x)
                    loss = lossfn(out, y)
                    if arm == 'D' and (_ % 50 == 0):
                        ben_now = benettin_ftle_max(m, xdr)
                        loss = loss + 0.05 * (ben_now - (-0.3)) ** 2
                    opt.zero_grad(); loss.backward(); opt.step()
                k_now, lam_now = kappa_lam(m.W.detach())
                k_hist.append((k_now, lam_now))
                # Intervention NACH Block 5 (Index 4): Arm B NUR Zufall,
                # Arm C NUR Ziel-Korrektur bei IDENTISCHER Stoerungsnorm.
                if arm in ('B', 'C') and blk == 4:
                    with torch.no_grad():
                        W = m.W
                        dB = torch.randn_like(W) * 0.02
                        dC = spectral_renorm(W, k0, lam0) - W
                        nb = dB.norm()
                        nc = dC.norm().clamp_min(1e-9)
                        if arm == 'B':
                            W.add_(dB)
                        else:
                            W.add_(dC * (nb / nc))
            ben_end = benettin_ftle_max(m, xdr)
            torch.manual_seed(seed + 9000)
            mc, _ = memory_capacity(m, max_lag=12, seq_len=64, input_scale=0.5)
            pre = sum(adapt_hist[2:5]) / 3.0
            post = sum(adapt_hist[5:8]) / 3.0
            k_end = k_hist[-1][0]
            L.append(f'Arm {arm} seed {seed}: adapt pre {pre:5.1f} | post {post:5.1f} | '
                     f'kappa {k0:5.2f}->{k_end:5.2f} | ben {ben0:+.3f}->{ben_end:+.3f} | MC {mc:6.2f}')
            agg.setdefault(arm, []).append((pre, post, mc, k_end, ben_end))

    L.append('')
    L.append('--- Aggregate (Seed-Mittel) ---')
    for arm in ('A', 'B', 'C', 'D'):
        sub = agg[arm]
        pre = sum(v[0] for v in sub) / len(sub)
        post = sum(v[1] for v in sub) / len(sub)
        mc = sum(v[2] for v in sub) / len(sub)
        L.append(f'Arm {arm}: adapt pre {pre:5.1f} -> post {post:5.1f} | dPlast {post-pre:+6.1f} | MC {mc:6.2f}')

    a = {arm: (sum(v[1] for v in agg[arm]) / len(agg[arm])) for arm in ('A', 'B', 'C', 'D')}
    p2 = a['C'] < a['B']
    p3 = a['D'] < a['A']
    L.append('')
    L.append(f'P1 (Erosion): continue adapt pre {sum(v[0] for v in agg["A"])/len(agg["A"]):.1f} vs post {a["A"]:.1f}')
    L.append(f'P2 (targeted schlaegt random): {"JA" if p2 else "NEIN"} (C {a["C"]:.1f} vs B {a["B"]:.1f} Steps)')
    L.append(f'P3 (Praevention): {"JA" if p3 else "NEIN"} (D {a["D"]:.1f} vs A {a["A"]:.1f} Steps)')
    L.append('')
    L.append('LESEAT: P2 JA => die Ressourcen-Richtung ist die Information — Suttons')
    L.append('"Zufall ist noetig" wird widerlegt durch "Gezielt ist besser".')
    L.append('P2 NEIN => Plasticity-Erosion ist keine Ressourcen-Frage; These VII faellt.')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
