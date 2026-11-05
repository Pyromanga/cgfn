#!/usr/bin/env python
'''G1 — KONTROLL-AUTORITAET: KANN MAN DEN LYAPUNOV-EXPONENTEN UEBERHAUPT
TUNEN? (neue Serie G [Kontrolle], registriert 2026-11-05, VOR dem Lauf)

DER OFFENSICHTLICHE BLINDE FLECK DES FELDES:
  Das gesamte Reservoir-Tuning (Edge of Chaos, SETI, praktische
  Guides, tausende Papiere) benutzt EINEN Knopf: die spektrale
  Skalierung (lambda_max bzw. spektraler Radius). Implizite
  Annahme: Der Knopf hat KONTROLL-AUTORITAET ueber den (getriebenen)
  Lyapunov-Exponenten — man kann ein Ziel anfahren.
  Sornette et al. (arXiv 2603.08191) haben gezeigt: Nicht-Normalitaet
  treibt lambda_1 bei FIXEM Spektrum durch null (Chaos ohne spektrale
  Kritikalitaet). Die Umkehrung bleibt ungeprueft — und unsere
  F35/F36/F37-Daten sind dreifache Anzeichen, dass sie bricht:
  unter band-strukturierter Nicht-Normalitaet war ben=-0.30 nicht
  mehr sauber anfahrbar (F37: 11/30 Modelle bei H=32 akzeptiert).
  G1 prueft die Kontroll-Autoritaet SYSTEMATISCH und QUER ueber
  drei Konstruktionen — inklusive des Feld-Standard-ESN.

DREI ARME (gleiche CT-RNN-Zelle, nur W unterscheidet sich):
  A1 designed-normal : Schur-Konstruktion s=0 -> W SYMMETRISCH
      (normal). Kontrollgruppe: Hier MUSS der Knopf gehen.
  A2 designed-nn     : Schur-Konstruktion s in {0.8, 1.6}
      (unsere Nicht-Normalitaets-Skala).
  A3 random-esn      : Gauss-Zufalls-W, skaliert auf spektralen
      Radius rho — SO BAUT ES DIE WELT. Zufalls-Matrizen sind mit
      Wahrscheinlichkeit 1 nicht-normal.

MESSUNG (identisches Instrument wie die gesamte Frontier):
  getriebener Benettin (drive_input wie f34..f37), H in {32,128},
  Knob-Raster 10 Punkte (0.2..2.0), Seeds 0..9.
  KONTROLL-AUTORITAET pro (Arm, H):
    TREFFERQUOTE : Anteil der (Knob, Seed)-Punkte mit
                   |ben1 - (-0.30)| <= 0.05  (Ziel anfahrbar?)
    KONTROLLRAUSCHEN: std(ben1) ueber Seeds bei festem Knob,
                   gemittelt ueber Knob-Punkte.
    MONOTONIE    : Spearman(knob, mean_ben1) — ist die Karte
                   ueberhaupt geordnet?

REGISTRIERUNG (VOR DEM LAUF GENAGELT, KEINE NACHVERHANDLUNG):
  PRIMAER (Kontroll-Kollaps): PASS genau dann, wenn in BEIDEN H:
    TREFFERQUOTE(A1 designed-normal) >= 0.50
    UND TREFFERQUOTE(A3 random-esn) < 0.25
    UND TREFFERQUOTE(A2 s=1.6) < 0.25
  -> Der Standard-Knopf hat unter Nicht-Normalitaet KEINE
     Kontroll-Autoritaet mehr; Kritikalitaets-Tuning auf
     Zufalls-Reservoirs ist Lotterie. (G2 prueft dann registriert
     den Ersatzknopf.)
  Sonst KILL: Der Knopf ist besser als unsere Daten vermuten
  liessen — auch ein klares, brauchbares Ergebnis.
  SEKUNDAER: KONTROLLRAUSCHEN(A3) > KONTROLLRAUSCHEN(A1) in
  beiden H (Seed-Streuung als Kontroll-Unschaerfe).
  DESKRIPTIV (keine Schwelle): Spearman(g_transient, ben1)
  ueber alle Modelle — Ersatz-Koordinaten-Kandidat; wird erst
  in G2 zu einer registrierten Wette, wenn G1 PASS liefert.
  GATES: ben1 nicht endlich -> Punkt ungueltig (gezaehlt);
  >30% ungueltig in einem Arm -> Arm verfaellt zurueck auf
  UNENTSCHEIDEN-Kriterium; fehlender Vergleichsarm ->
  UNENTSCHEIDEN (Design, keine Schwelle).
'''
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import IS
from cgfn.benettin import benettin_spectrum
from cgfn.design import make_designed_model
from cgfn.field import drive_input
from cgfn.resource import transient_growth
from cgfn.model import ContinuousRNN

BEN_TARGET = -0.30
BEN_TOL = 0.05
KNOBS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def rank_corr(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    ok = torch.isfinite(a) & torch.isfinite(b)
    if ok.sum() < 5 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
        return float('nan')
    ra = a[ok].argsort().argsort().to(torch.float64)
    rb = b[ok].argsort().argsort().to(torch.float64)
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def make_random_esn(H, rho, seed):
    '''Feld-Standard: Gauss-W, skaliert auf spektralen Radius rho.'''
    model = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(H, H, generator=gen)
    with torch.no_grad():
        W = W * rho
        model.W.copy_(W)
    model.W.requires_grad_(False)
    return model


def ben_of_model(m, xdr):
    spec, _ = benettin_spectrum(m, xdr)
    return float(spec[0])


def main():
    p = argparse.ArgumentParser(description='G1 control authority audit')
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--out', default='results/g1_control_authority_results.txt')
    p.add_argument('--csv', default='results/g1_control_authority_table.csv')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, IS)
    L = []
    L.append('=' * 118)
    L.append('G1 — KONTROLL-AUTORITAET (registriert: PASS nur wenn Trefferquote normal >= 0.50,')
    L.append(f'nicht-normale Arme < 0.25, in BEIDEN H; Ziel ben={BEN_TARGET} +- {BEN_TOL}; Seeds 0-9)')
    L.append(f'Knob-Raster: {KNOBS}')
    L.append('=' * 118)

    rows = []
    for H in (32, 128):
        L.append('')
        L.append(f'########## H={H} ##########')
        arms = [('normal', None), ('nn_s0.8', 0.8), ('nn_s1.6', 1.6),
                ('random-esn', 'rand')]
        for arm_name, s_nn in arms:
            for knob in KNOBS:
                bens, gs = [], []
                for seed in seeds:
                    if arm_name == 'random-esn':
                        m = make_random_esn(H, knob, seed)
                    else:
                        m = make_designed_model(H, 1.0, knob,
                                                0.0 if s_nn is None else s_nn,
                                                seed, task_dim=0)
                    ben = ben_of_model(m, xdr)
                    J = -torch.eye(H, dtype=torch.float64) / 1.0 + \
                        m.W.detach().to(torch.float64)
                    g, _t = transient_growth(J)
                    bens.append(ben)
                    gs.append(float(g))
                ok = [b for b in bens if b == b and abs(b) < 1e6]
                n_bad = len(bens) - len(ok)
                mean_b = sum(ok) / max(1, len(ok)) if ok else float('nan')
                std_b = (torch.tensor(ok, dtype=torch.float64).std().item()
                         if len(ok) > 1 else float('nan'))
                hits = sum(1 for b in ok if abs(b - BEN_TARGET) <= BEN_TOL)
                rate = hits / len(bens) if bens else float('nan')
                rows.append(dict(H=H, arm=arm_name, knob=knob, bens=bens,
                                 gs=gs, mean=mean_b, std=std_b, rate=rate,
                                 n_bad=n_bad))
                L.append(f"{arm_name:>10s} knob={knob:<4g} mean(ben1)={mean_b:+.3f} "
                         f"std(Seed)={std_b:.3f} Treffer={hits}/{len(bens)} "
                         f"ungueltig={n_bad} mean(g)={sum(gs)/len(gs):.2f}")
            Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(args.csv, 'w', encoding='utf-8') as fh:
        fh.write('H,arm,knob,mean_ben1,std_ben1,hit_rate,mean_g\n')
        for r in rows:
            fh.write(f"{r['H']},{r['arm']},{r['knob']},{r['mean']:.6g},"
                     f"{r['std']:.6g},{r['rate']:.6g},"
                     f"{sum(r['gs'])/max(1,len(r['gs'])):.6g}\n")

    # --- Registrierte Auswertung ---
    L.append('')
    L.append('--- G1-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')

    def arm_rate(H, arm):
        rs = [r for r in rows if r['H'] == H and r['arm'] == arm]
        tot = sum(len(r['bens']) for r in rs)
        hits = sum(sum(1 for b in r['bens']
                       if b == b and abs(b - BEN_TARGET) <= BEN_TOL)
                   for r in rs)
        bad = sum(r['n_bad'] for r in rs)
        return hits / tot, bad / tot

    def arm_noise(H, arm):
        rs = [r for r in rows if r['H'] == H and r['arm'] == arm
              and r['std'] == r['std']]
        return sum(r['std'] for r in rs) / max(1, len(rs))

    results = {}
    for H in (32, 128):
        block = {}
        for arm in ('normal', 'nn_s0.8', 'nn_s1.6', 'random-esn'):
            rate, bad = arm_rate(H, arm)
            noise = arm_noise(H, arm)
            rs = [r for r in rows if r['H'] == H and r['arm'] == arm]
            mmono = rank_corr([r['knob'] for r in rs],
                              [r['mean'] for r in rs])
            block[arm] = dict(rate=rate, bad=bad, noise=noise, mono=mmono)
            L.append(f"H={H} {arm:>10s}: TREFFERQUOTE={rate:.3f} "
                     f"ungueltig={bad:.3f} Kontrollrauschen={noise:.4f} "
                     f"Monotonie(Knob->ben)={mmono:+.3f}")
        results[H] = block

    pass_all = True
    for H in (32, 128):
        b = results[H]
        cond = (b['normal']['rate'] >= 0.50 and
                b['random-esn']['rate'] < 0.25 and
                b['nn_s1.6']['rate'] < 0.25 and
                b['normal']['bad'] <= 0.30 and
                b['random-esn']['bad'] <= 0.30 and
                b['nn_s1.6']['bad'] <= 0.30)
        L.append(f"Bedingung H={H}: normal>={b['normal']['rate']:.2f} (>=0.50), "
                 f"random={b['random-esn']['rate']:.2f} (<0.25), "
                 f"nn1.6={b['nn_s1.6']['rate']:.2f} (<0.25) -> "
                 f"{'ERFUELLT' if cond else 'NICHT ERFUELLT'}")
        pass_all = pass_all and cond

    sek_ok = all(results[H]['random-esn']['noise'] > results[H]['normal']['noise']
                 for H in (32, 128))
    L.append(f"SEKUNDAER Kontrollrauschen random > normal (beide H): "
             f"{'ERFUELLT' if sek_ok else 'NICHT ERFUELLT'}")

    all_g, all_b = [], []
    for r in rows:
        for g, b in zip(r['gs'], r['bens']):
            if g == g and b == b:
                all_g.append(g); all_b.append(b)
    cg = rank_corr(all_g, all_b)
    L.append(f"DESKRIPTIV Ersatzknopf-Kandidat: Spearman(g_transient, ben1) "
             f"ueber alle Punkte: {cg:+.3f} (keine Schwelle; G2-Wette nur bei PASS)")

    if pass_all:
        verdict = ('PASS — der spektrale Knopf verliert unter Nicht-Normalitaet '
                   'die Kontroll-Autoritaet: das Feld tuned auf ein Ziel, das '
                   'es mit diesem Knopf nicht anfahren kann. G2: registrierte '
                   'Wette auf den Ersatzknopf (transiente Verstaerkung).')
    elif any(results[H][arm]['bad'] > 0.30 for H in (32, 128)
             for arm in ('normal', 'nn_s1.6', 'random-esn')):
        verdict = ('UNENTSCHEIDEN — zu viele ungueltige Punkte (Benettin-'
                   'Instabilitaet), Design-Frage, keine Schwelle.')
    else:
        verdict = ('KILL — der Knopf ist kontrollfaehiger als die F35/F36/F37-'
                   'Anzeichen vermuten liessen. Auch das ist ein Ergebnis: '
                   'dann lag das Anfahrungs-Problem in den Band-Ketten, '
                   'nicht in der Nicht-Normalitaet als solcher.')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} und {args.csv} geschrieben.')


if __name__ == '__main__':
    main()
