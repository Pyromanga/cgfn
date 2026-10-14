'''
F25 — REVERBERATING MASS: Kortex-Konsens (m in 0.94-0.998) im CT-RNN
reproduzieren — mit voller Crackling-Batterie statt nur Branching-Ratio.

Literatur-Lage (Tiefenanalyse I-V):
  Hirnkritikalitaets-Review: Kortex ist 'reverberating', subkritisch,
    Branching-Ratio m in 0.94-0.998 — exakt unsere F12-Linie.
  Beggs & Plenz 2003: m = Verhaeltnis Aktivitaet(bin t+1)/Aktivitaet(bin t)
    innerhalb von Lawinen; kritisch bei m=1, Exponent size ~ -3/2.
  Clauset/Shalizi/Newman 2009: Power-Law-Exponenten per MLE schaetzen,
    xmin per KS-Minimierung — NIE per Log-Log-Regression (least squares).
  Quasi-Kritikalitaet (IOP Nanowire-Perspektive): getriebene Systeme
    liegen nicht auf einem Punkt, sondern auf einer (tauS, tauT)-Linie,
    die die Skalierungsrelation (tauT-1)/(tauS-1) ~ gamma NAHE erfuellt.
  eLife 'Neural criticality from effective latent variables':
    Lawinen-Kritikalitaet kann als BEIPRODUKT latenter Variablen ohne
    Feintuning entstehen — m im Fenster ist NOTWENDIG, nicht hinreichend.
  => KONSEQUENZ FUER F25: m wird ueber einen lam-Sweep gemessen
     (kontrollierte Distanz zur getriebenen Kante). Nur wenn m
     MONOTON mit lam wandert und das MC-Optimum im/rechts vom
     m-Fenster liegt, ist m eine echte Regime-Koordinate und kein
     Beiprodukt.

DESIGN (Reservoir, freeze_w, kein Training):
  - Designed models (Schur, s als kappa-Regler), lam-Sweep um die
    getriebene Kante (benettin gemessen).
  - Langer weisser Rausch-Rollout (T=4000, burn-in 200).
  - Ereignis-Definition: pro Unit |z| > per-unit 99%-Quantile (MEA-
    Analogon: duenne Ereignisrate, Lawinen = Binde mit >0 Events).
  - BATTERIE: (a) m_autocorr = C1/C0 (Priesemann-Standard), (b) m_bp
    (Beggs-Plenz Bin-Ratio in Lawinen), (c) MLE tauS/tauT mit KS-
    gewaehltem xmin, (d) gamma-Fit (log<S> vs log T) gegen
    gamma_pred = (tauT-1)/(tauS-1), (e) MC am selben Netz.

VORHERSAGEN:
  P1: m wandert monoton mit lam durch das Kortex-Fenster (0.94-0.998).
  P2: MC-Optimum liegt bei m < 1 (subkritisch, F12-Kollision).
  P3 (LSTM-Branching-Analog, arXiv:2606.10384): m rueckt mit
    hidden_dim Richtung 1 — gemessen im hidden-Sweep.
'''
import argparse
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max
from cgfn.design import make_designed_model
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity

M_WINDOW = (0.94, 0.998)


@torch.no_grad()
def rollout_states(model, T, seed, scale=0.5, burn=200):
    gen = torch.Generator().manual_seed(seed + 777)
    x = torch.randn(1, T, 1, generator=gen) * scale
    _, states = model(x, None)
    return states[0, burn:]  # (T-burn, H)


def avalanche_stats(states, q=0.99):
    '''Lawinen-Statistik aus Zustandstraektorie (T, H).'''
    thr = states.quantile(q, dim=0)
    active = (states.abs() > thr).sum(dim=1).float()  # a_t (T,)
    # Lawinen: maximale Binde mit a_t > 0
    av_sizes, av_durs = [], []
    t = 0
    T = active.shape[0]
    while t < T:
        if active[t] > 0:
            t0 = t
            while t < T and active[t] > 0:
                t += 1
            av_sizes.append(float(active[t0:t].sum()))
            av_durs.append(float(t - t0))
        else:
            t += 1
    # (a) m via Autokovarianz (Priesemann): m = C1/C0
    a = active - active.mean()
    c0 = (a * a).mean().item()
    c1 = (a[:-1] * a[1:]).mean().item()
    m_autocorr = c1 / max(c0, 1e-12)
    # (b) Beggs-Plenz: Verhaeltnis aufeinanderfolgender Bins, gemittelt
    ratios = []
    for i, d in enumerate(av_durs):
        pass
    idx = 0
    for d in av_durs:
        seg = active[idx: idx + int(d)]
        for k in range(len(seg) - 1):
            if seg[k] > 0:
                ratios.append(float(seg[k + 1] / seg[k]))
        idx += int(d)
    m_bp = sum(ratios) / max(len(ratios), 1) if ratios else float('nan')
    return active, av_sizes, av_durs, m_autocorr, m_bp


def pl_mle(xs):
    '''Diskreter Power-Law-MLE mit KS-gewaehltem xmin (Clauset et al.).
    Rueckgabe: (alpha, xmin, ks) oder (nan, nan, nan).'''
    xs = [v for v in xs if v >= 1.0]
    n = len(xs)
    if n < 80:
        return float('nan'), float('nan'), float('nan')
    cands = sorted(set(xs))
    cands = [c for c in cands if c >= 1 and sum(1 for v in xs if v >= c) >= 60]
    if not cands:
        cands = [max(1.0, min(xs))]
    best = None
    for xmin in cands:
        tail = [v for v in xs if v >= xmin]
        nt = len(tail)
        s = sum(__import__('math').log(v / (xmin - 0.5)) for v in tail)
        if s <= 0:
            continue
        alpha = 1.0 + nt / s
        # KS gegen CCDF (x/xmin)^-(alpha-1)
        tail_s = sorted(tail)
        ks = 0.0
        for i, v in enumerate(tail_s):
            f_emp = 1.0 - (i + 1) / nt
            f_mod = (v / xmin) ** (-(alpha - 1.0))
            ks = max(ks, abs(f_emp - f_mod))
        if best is None or ks < best[2]:
            best = (alpha, xmin, ks)
    if best is None:
        return float('nan'), float('nan'), float('nan')
    return best


def gamma_fit(av_sizes, av_durs):
    '''gamma = Steigung log<S>(T) vs log T; Vergleich: (tauT-1)/(tauS-1).'''
    by_d = {}
    for s, d in zip(av_sizes, av_durs):
        by_d.setdefault(int(d), []).append(s)
    pts = [(d, sum(v) / len(v)) for d, v in sorted(by_d.items()) if len(v) >= 3]
    if len(pts) < 3:
        return float('nan')
    import math
    xs = [math.log(d) for d, _ in pts]
    ys = [math.log(s) for _, s in pts]
    n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        return float('nan')
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def measure_config(hidden, lam, s, seed, T, q):
    torch.manual_seed(seed)
    m = make_designed_model(hidden, 1.0, lam, s, seed, task_dim=0)
    xdr = drive_input(4, 48, 0.5)
    ben = benettin_ftle_max(m, xdr)
    states = rollout_states(m, T, seed)
    active, sizes, durs, m_auto, m_bp = avalanche_stats(states, q)
    tauS, xminS, ksS = pl_mle(sizes)
    tauT, xminT, ksT = pl_mle(durs)
    g_fit = gamma_fit(sizes, durs)
    if not (tauS != tauS) and not (tauT != tauT) and tauS > 1.001 and tauT > 1.001:
        g_pred = (tauT - 1.0) / (tauS - 1.0)
    else:
        g_pred = float('nan')
    torch.manual_seed(seed + 9000)
    mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=0.5)
    return dict(ben=ben, m_auto=m_auto, m_bp=m_bp, n_av=len(sizes),
                tauS=tauS, tauT=tauT, g_fit=g_fit, g_pred=g_pred, mc=mc)


def fmt(v, w=7, p=3):
    if v != v:
        return ' ' * (w - 1) + '-'
    return f'{v:+{w}.{p}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seeds', default='0,1')
    p.add_argument('--lam-values', default='0.8,1.0,1.2,1.4,1.6')
    p.add_argument('--s-values', default='0.8,1.6')
    p.add_argument('--T', type=int, default=4000)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--quantile', type=float, default=0.99)
    p.add_argument('--hidden-sweep', default='16,32,64')
    p.add_argument('--out', default='results/f25_reverberation_results.txt')
    args = p.parse_args()

    seeds = [int(v) for v in args.seeds.split(',')]
    lams = [float(v) for v in args.lam_values.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    h_sweep = [int(v) for v in args.hidden_sweep.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    L = []
    L.append('=' * 120)
    L.append('F25 REVERBERATING MASS: Branching-Ratio-Batterie im CT-RNN (Kortex-Fenster 0.94-0.998)')
    L.append(f'lam={lams} s={s_vals} seeds={seeds} T={args.T} hidden={args.hidden} q={args.quantile}')
    L.append('P1: m wandert monoton mit lam | P2: MC-Optimum bei m<1 | P3: m -> 1 mit hidden_dim')
    L.append('=' * 120)

    rows = []
    for s in s_vals:
        L.append('')
        L.append(f'########## s (Nicht-Normalitaet) = {s} ##########')
        L.append(f"{'lam':>5s} {'seed':>4s} {'ben':>7s} {'m_auto':>7s} {'m_bp':>7s} {'tauS':>6s} {'tauT':>6s} {'g_fit':>6s} {'g_pred':>6s} {'n_av':>6s} {'MC':>6s}")
        for lam in lams:
            for seed in seeds:
                r = measure_config(args.hidden, lam, s, seed, args.T, args.quantile)
                rows.append(dict(s=s, lam=lam, seed=seed, **r))
                L.append(f'{lam:5.2f} {seed:4d} {fmt(r["ben"])} {fmt(r["m_auto"])} {fmt(r["m_bp"])} '
                         f'{fmt(r["tauS"], 6, 2)} {fmt(r["tauT"], 6, 2)} {fmt(r["g_fit"], 6, 2)} '
                         f'{fmt(r["g_pred"], 6, 2)} {r["n_av"]:6d} {r["mc"]:6.2f}')

    # --- Monotonie P1: m_auto vs lam (pro s, gemittelt ueber Seeds) ---
    L.append('')
    L.append('--- P1: m_autocorr (Seed-Mittel) vs lam ---')
    mono_ok = 0
    mono_tot = 0
    best_lam = None
    for s in s_vals:
        ms = []
        for lam in lams:
            sub = [r for r in rows if r['s'] == s and r['lam'] == lam]
            ms.append(sum(r['m_auto'] for r in sub) / len(sub))
        inc = sum(1 for a, b in zip(ms, ms[1:]) if b > a)
        mono_tot += len(ms) - 1
        mono_ok += inc
        in_win = [lams[i] for i, mv in enumerate(ms) if M_WINDOW[0] <= mv <= M_WINDOW[1]]
        L.append(f's={s}: m(lam) = ' + ' '.join(f'{mv:+.3f}' for mv in ms) +
                 f' | steigend: {inc}/{len(ms)-1} | im Fenster: {in_win}')
        if in_win:
            best_lam = in_win[0]

    # --- P2: MC vs m_auto ueber alle rows ---
    msv = torch.tensor([r['m_auto'] for r in rows])
    mcv = torch.tensor([r['mc'] for r in rows])
    if msv.std() > 1e-6 and mcv.std() > 1e-6:
        corr = torch.corrcoef(torch.stack([msv, mcv]))[0, 1].item()
        L.append(f'corr(m_auto, MC) = {corr:+.3f} (P2: MC-Peak bei m<1 heisst negativ/quadratisch)')
        im = int(msv.argmax())
        L.append(f'MC-max bei m_auto={msv[im].item():+.3f}, MC={mcv[im].item():.2f}')

    # --- P3: hidden-Sweep am lam mit m nahe Fenstermitte ---
    L.append('')
    L.append('--- P3: hidden-Sweep (s=0.8, lam am naechsten an Fenstermitte) ---')
    target_m = 0.5 * (M_WINDOW[0] + M_WINDOW[1])
    if best_lam is None:
        sub0 = [r for r in rows if r['s'] == s_vals[0]]
        best_lam = min(sub0, key=lambda r: abs(r['m_auto'] - target_m))['lam']
    L.append(f'lam={best_lam}')
    L.append(f"{'H':>4s} {'seed':>4s} {'m_auto':>7s} {'m_bp':>7s} {'tauS':>6s} {'tauT':>6s} {'MC':>6s}")
    h_rows = []
    for H in h_sweep:
        for seed in seeds:
            r = measure_config(H, best_lam, 0.8, seed, args.T, args.quantile)
            h_rows.append(dict(H=H, **r))
            L.append(f'{H:4d} {seed:4d} {fmt(r["m_auto"])} {fmt(r["m_bp"])} '
                     f'{fmt(r["tauS"], 6, 2)} {fmt(r["tauT"], 6, 2)} {r["mc"]:6.2f}')
    for H in h_sweep:
        sub = [r for r in h_rows if r['H'] == H]
        mm = sum(r['m_auto'] for r in sub) / len(sub)
        L.append(f'H={H:3d}: m_auto(seed-mittel) = {mm:+.3f}')
    L.append('')
    L.append('LESEAT:')
    L.append('P1 ok  => m ist eine Regime-Koordinate (kein Latent-Variablen-Beiprodukt).')
    L.append('P2 ok  => MC-Optimum subkritisch von m=1 — Kortex-Konsens trifft auf RNN (F12 x Beggs-Plenz).')
    L.append('P3 ok  => m rueckt mit H Richtung 1 — LSTM-Branching-Analog im CT-RNN.')
    L.append('g_fit ~ g_pred => Quasi-Kritikalitaet (Skalierungsrelation auf der (tauS, tauT)-Linie).')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
