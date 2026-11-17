'''T32-Contract — RUECKMELDUNGS-DICHTE BEI FIXEM HORIZONT v1
(registriert XVII Nachtrag 56, 2026-11-17, VOR dem Lauf;
korrigiert nach externem Einwand: EINE Variable pro Test).
HYPOTHESE: Die Rueckmeldungs-Dichte der Aufgabe (sparse:
ein Label am Ende, dense: ein Label je Frame) — NICHT die
Gedaechtnis-Forderung — ist der Traeger des Reservoir-
Schalters (Kollaps von P unter Konjugation).
TEST: Gedaechtnis-Horizont FIX bei D = 20 Frames (wie
delayed recall T23). Zwei Zellen:
  SPARSE/D20 — exakt das T20/T23-MEM: ein Label am Ende,
    haengt an den zwei Pulsen (Referenz, Kollaps bekannt).
  DENSE/D20 — an JEDEM Frame t >= D ein Label, abhaengig
    vom Input EXAKT D Frames vorher (Zufalls-Lehrer je
    Seed: tanh-Lehrer auf dem 2-dim Input, Seed 50000+s).
    Gleiche Gedaechnis-Forderung (Horizont 20), dichte
    Rueckmeldung. Frames t < D tragen nicht zum Loss bei.
Stufen a in {0, 1}: SPARSE 1350..1369 / 1370..1389,
DENSE 1390..1409 / 1410..1429. 80 Runs, f64, ~30 min,
Gradienten-Maschine exakt T20/T23 (H=48, s=2.2, wb=0.6,
600 Steps).
FALSIFIKATION: DENSE/D20 kollabiert wie SPARSE/D20
(P_DENSE(1) <= 0.15) → Hypothese gefallen; die
Gedaechtnis-Forderung ist der Traeger.
  PASS — DICHTE-TRAEGER: P_DENSE(1) >= 0.5 UND
    P_SPARSE(1) <= 0.15 (Kollaps repliziert) UND
    P_DENSE(0) >= 0.5 — der Schalter folgt der Dichte,
    nicht dem Horizont.
  KILL — HORIZONT-TRAEGER: P_DENSE(1) <= 0.15 (bei
    intakter Sanity) — der Schalter folgt der
    Gedaechtnis-Forderung; Dichte-Hypothese gefallen.
  GEMISCHT: P_DENSE(1) in (0.15, 0.50) (intakte Sanity) —
    beide Mechanismen tragen bei; registriert VOR dem
    Lauf, keine Nachdeutung.
  UNENTSCHEIDEN (Crash): Sanity verletzt (P_SPARSE(0) <
    0.25 oder P_DENSE(0) < 0.5), ndiv > 6 je Block.
Band (R8): P_SPARSE(0) in [0.25, 0.64], P_SPARSE(1) in
[0.00, 0.22] (T23/T24/T25). DENSE-Arme: ERSTFAHRT ohne
Band (registriert).
Recon (2026-11-17): Why Code, Why Now (arXiv 2602.13934)
— Informationsstruktur der Aufgabe als Obergrenze; hier
auf Mini-Skala kontrolliert getestet. Kein Doppel
bekannt. Instrument: experiments/t32_dichte_horizont.py;
Verdikt generiert vom W1-Kernel. Rollen: block.''';

H = 48
Q_SEED = 2020
BLOCKS = {('sparse', 0): (1350, 1369), ('sparse', 1): (1370, 1389),
          ('dense', 0): (1390, 1409), ('dense', 1): (1410, 1429)}
DELAY = 20


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t32():
    arms = []
    for (cell, a), rng in sorted(BLOCKS.items()):
        arms.append(dict(name='%s_A%d' % (cell.upper(), a), H=H,
                         kind='norm', role='block', s=2.2, wb=0.6,
                         conj_a=a, cell=cell,
                         seeds=_seeds(rng),
                         invariants=[dict(metric='ndiv', op='le',
                                          expect=6)]))
    return dict(
        measures=['pf', 'colnorm', 'ndiv'],
        params=dict(
            blocks={str(k): list(v) for k, v in BLOCKS.items()},
            q_seed=Q_SEED, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            delay=DELAY, teacher_seed0=50000, t_hidden=8),
        arms=arms,
        sanity=[dict(id='sanity-sparse', text='SPARSE a=0 im '
                     'T22-Band (Referenz)', predicate=[dict(
                         metric='SPARSE_A0.pf', op='ge', value=0.25)]),
                dict(id='sanity-dense', text='DENSE a=0 lernt '
                     '(Kontrolle)', predicate=[dict(
                         metric='DENSE_A0.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — DICHTE-TRAEGER',
                 text=('P_DENSE(1) >= 0.5 UND P_SPARSE(1) <= 0.15 '
                       'UND P_DENSE(0) >= 0.5: der Schalter folgt '
                       'der Rueckmeldungs-Dichte, nicht dem '
                       'Gedaechtnis-Horizont'),
                 predicate=[dict(metric='SPARSE_A0.pf', op='ge',
                                 value=0.0),
                            dict(metric='DENSE_A0.pf', op='ge',
                                 value=0.0)]),
            dict(id='KILL — HORIZONT-TRAEGER',
                 text=('P_DENSE(1) <= 0.15 bei intakter Sanity: '
                       'der Schalter folgt der Gedaechtnis-'
                       'Forderung; Dichte-Hypothese gefallen'),
                 predicate=[dict(metric='SPARSE_A0.pf', op='lt',
                                 value=-1.0)]),
        ])