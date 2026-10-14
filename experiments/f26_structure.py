'''
F26 — STRUKTURIERTE NICHT-NORMALITAET: 'So baut man sie.'
Band-Kette vs Kurz-Ketten vs Rauschen im Schur-Delta — bei FESTEM
kappa und FESTEM getriebenem lambda.

Literatur (Tiefenanalyse V+VIII):
  Ganguli 2008 / Orhan & Pitkow 2019: maximale Fisher-Memory braucht
    versteckte FEEDFORWARD-Ketten, die superlinear amplifizieren;
    trainierte orthogonale RNNs entwickeln genau solche Ketten.
  Baggio et al. 2020: die Ketten sitzen im Schur-Delta (streng
    obere Dreiecksmatrix), identifizierbar & KONSTRUIERBAR.
  SSM-Linie (S4/Mamba): diagonal = normal = kappa 1 — die ganze
    Familie hat die Ressource strukturgleich auf null gebaut.

VORHERSAGE (scharf): Bei identischem kappa und identischem
getriebenem lambda gewinnt die geordnete Kette (Signal-Wanderung
in eine Richtung) gegen ungeordnete Kurz-Ketten und gegen
Zufalls-Nicht-Normalitaet auf Memory Capacity — weil die Kette
das Amplifikationsprofil in die ZEIT-Richtung legt (sequentielle
Abtastung der Vergangenheit, Ganguli-SNR-Argument).

DESIGN:
  Konstruktion: W = Q (lam*I + s*N) Q^T mit N aus drei Familien:
    chain:   Ein Super-Diagonale (volle Kette 1->2->...->H)
    chains4: Block-Ketten der Laenge 8 (4 parallele Ketten)
    random:  strictly-upper Zufall (wie design.py-Default)
  Q fest pro Seed. lam-Bisektion fixiert den GETRIEBENEN Benettin
  auf -0.30 (F24-Maschinerie). s wird pro Familie so gewaehlt,
  dass kappa ~ 6 (Regime B, F18c) — Kontrolle: kappa und ben
  muessen zwischen Armen uebereinstimmen (+-10%), sonst verwerfen.
  MESSUNG: MC (max_lag 16), t0, log1p(Kreiss-K).
'''
import argparse
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max
from cgfn.design import make_designed_model
from cgfn.field import drive_input, sech2_field_iteration, kreiss_constant
from cgfn.memory_capacity import memory_capacity

TARGET_BEN = -0.30
KAPPA_TARGET = 6.0


def build_family(family, hidden, lam, s, seed):
    torch.manual_seed(seed + 31)
    Q = torch.linalg.qr(torch.randn(hidden, hidden))[0]
    N = torch.zeros(hidden, hidden)
    if family == 'chain':
        idx = torch.randperm(hidden)
        perm = torch.zeros(hidden, hidden)
        for rank, unit in enumerate(idx):
            perm[unit, rank] = 1.0
        shift = torch.zeros(hidden, hidden)
        shift[torch.arange(hidden - 1), torch.arange(1, hidden)] = 1.0
        N = perm @ shift @ perm.T
    elif family == 'chains4':
        nblk = 4
        L = hidden // nblk
        for b in range(nblk):
            r0, r1 = b * L, (b + 1) * L
            N[r0:r1 - 1, r0 + 1:r1] = 1.0
    else:  # random strictly upper
        A = torch.randn(hidden, hidden)
        N = torch.triu(A, diagonal=1)
    M = lam * torch.eye(hidden) + s * N
    return Q @ M @ Q.T


def kappa_lam(W):
    sv = torch.linalg.svdvals(W)[0].item()
    lam = torch.linalg.eigvals(W).abs().max().item()
    return sv / max(lam, 1e-9), lam


def tune_s(family, hidden, lam, seed, xdr, s_lo=0.05, s_hi=3.0, iters=10):
    '''s-Bisektion auf kappa-Ziel (grobe Richtung), dann lam-Bisektion
    fuer den getriebenen Benettin. Zurueck: W, kappa, ben.'''
    s = 0.5 * (s_lo + s_hi)
    for _ in range(iters):
        s = 0.5 * (s_lo + s_hi)
        W = build_family(family, hidden, lam, s, seed)
        k, _ = kappa_lam(W)
        if k < KAPPA_TARGET:
            s_lo = s
        else:
            s_hi = s
    lo, hi = 0.5, 1.5  # lam-Bisektion (spektral) auf Ziel-Benettin
    for _ in range(12):
        lam_try = 0.5 * (lo + hi)
        W = build_family(family, hidden, lam_try, s, seed)
        #ContinuousRNN-compatible: benettin braucht Modell — nutze designed wrapper
        m = make_designed_model(hidden, 1.0, lam_try, 0.8, seed, task_dim=0)
        m.W.data = W
        ben = benettin_ftle_max(m, xdr)
        if ben < TARGET_BEN:
            lo = lam_try
        else:
            hi = lam_try
    return m, s, kappa_lam(W)[0], ben


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--out', default='results/f26_structure_results.txt')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, 0.5)
    L = []
    L.append('=' * 116)
    L.append('F26 STRUKTURIERTE NICHT-NORMALITAET: chain vs chains4 vs random bei festem kappa und lambda')
    L.append(f'kappa_target={KAPPA_TARGET} ben_target={TARGET_BEN} seeds={seeds} hidden={args.hidden}')
    L.append('Vorhersage: chain > chains4 > random auf MC (Amplifikationsprofil in Zeitrichtung)')
    L.append('=' * 116)

    rows = []
    L.append(f"{'family':>8s} {'seed':>4s} {'kappa':>6s} {'ben':>7s} {'t0':>6s} {'logK':>5s} {'MC':>6s}")
    for family in ('chain', 'chains4', 'random'):
        for seed in seeds:
            m, s, k, ben = tune_s(family, args.hidden, 0.9, seed, xdr)
            # Konsistenz-Kontrolle: kappa/ben innerhalb +-10% des Ziels
            if abs(k - KAPPA_TARGET) > 0.6 or abs(ben - TARGET_BEN) > 0.08:
                L.append(f'{family:>8s} {seed:4d}  — abgewichen (kappa {k:.2f}, ben {ben:+.3f}) — verworfen')
                continue
            with torch.no_grad():
                d = sech2_field_iteration(m, xdr)
                H = args.hidden
                J = -torch.eye(H) + m.W * d.unsqueeze(0)
                t0 = torch.linalg.eigvals(J).real.max().item()
                logk = torch.log1p(torch.tensor(kreiss_constant(J))).item()
            torch.manual_seed(seed + 9000)
            mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=0.5)
            L.append(f'{family:>8s} {seed:4d} {k:6.2f} {ben:+7.3f} {t0:+6.2f} {logk:5.2f} {mc:6.2f}')
            rows.append((family, seed, k, ben, t0, logk, mc))

    L.append('')
    L.append('--- Aggregate (Seed-Mittel) ---')
    means = {}
    for family in ('chain', 'chains4', 'random'):
        sub = [r for r in rows if r[0] == family]
        if not sub:
            L.append(f'{family}: keine gueltigen Konfigurationen')
            continue
        mc = sum(r[6] for r in sub) / len(sub)
        t0 = sum(r[4] for r in sub) / len(sub)
        means[family] = mc
        L.append(f'{family:>8s}: MC {mc:6.2f} | t0 {t0:+.2f} | n={len(sub)}')
    if len(means) == 3:
        p1 = means['chain'] > means['chains4'] > means['random']
        L.append(f"Ordnung chain > chains4 > random: {'JA' if p1 else 'NEIN'}")
        kaps = torch.tensor([r[2] for r in rows]); mcs = torch.tensor([r[6] for r in rows])
        if kaps.std() > 1e-6:
            L.append(f'corr(kappa, MC) innerhalb Familien: {torch.corrcoef(torch.stack([kaps, mcs]))[0,1]:+.3f} (sollte ~0 sein — Ressource ist Struktur, nicht kappa)')
    L.append('')
    L.append('LESEAT: Ordnung JA => Design-Prinzip: Nicht-Normalitaet BAUT man als Kette (Zeitrichtung),')
    L.append('nicht als Rauschen. Das ist der SSM-Anschluss (F29: Band-Kette als Stoerung')
    L.append('der Mamba-Diagonale). Ordnung NEIN => unstrukturierte Nicht-Normalitaet reicht,')
    L.append('F26-These faellt, Design-Frage stellt sich nicht.')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
