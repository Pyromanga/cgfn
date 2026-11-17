'''T33-Contract — SIGNALSTRAEKE-MATCH (T32-Konfund) v1
(registriert XVII Nachtrag 57, 2026-11-17, VOR dem Lauf).
HYPOTHESE (aus T32, PASS DICHTE-TRAEGER): Die Rueckmeldungs-
Dichte der Aufgabe, nicht die Gedaechtnis-Forderung, ist der
Traeger des Schalters. OFFENER KONFUND: T32 variierte auch die
Optimierungs-Dynamik (dense: 4 Label-Frames je Sequenz statt 1;
Gradienten-VARIANZ sinkt durch Mittelung ueber 4 Lehrer-Terme;
Magnitude je Step ist bei Mean-Reduction vergleichbar — Korrektur
gegenueber der ersten Lesart, die faelschlich ~20x sagte: korrekt
ist 4x).
TEST: Maschine exakt T20/T23/T32 (H=48, s=2.2, wb=0.6, D=20,
lr=5e-3, Adam). Vier Arme, a in {0,1}:
  MB (DENSE_MASK, 600 Steps): dichte Aufgabe wie T32, aber je
    Step geht nur EINES der 4 Label-Frames in den Loss (zyklisch
    k mod 4). Matched Terme je Step (32 = sparse), Gradienten-
    Varianz und totale Signalmenge (600 x 32) bei erhaltener
    dichter Aufgabenstruktur (jeder Frame alle 4 Steps).
  S4X (SPARSE, 2400 Steps): sparse Aufgabe mit 4x Training —
    testet "zu wenig Training" als Alternative.
Bloecke: MB 1430..1469, S4X 1470..1509. 80 Runs, f64, ~75-90 min.
Referenz aus T32 (P_SPARSE=0.05/0.75, P_DENSE=1.00/1.00), keine
Referenz-Arme neu gefahren.
LESARTEN (eingefroren): Sanity P_MB(0) >= 0.5, P_S4X(0) >= 0.5.
  PASS — DICHTE-TRAEGER (These geschlossen): P_MB(1) >= 0.5 UND
    P_S4X(1) <= 0.15.
  KILL — OPTIMIERUNGS-TRAEGER (These gekillt): P_MB(1) <= 0.15
    UND P_S4X(1) >= 0.5.
  GEMISCHT: jede andere Konstellation bei intakter Sanity;
    Divergenzen einzeln gedeutet (registriert).
  UNENTSCHEIDEN: Sanity verletzt, ndiv > 6 je Block.
Baender: ALLE Arme ERSTFAHRT ohne Band (neue Optimierungs-Regime).
Instrument: experiments/t33_signalstaerke_match.py. Rollen: block.''';

H = 48
Q_SEED = 2020
BLOCKS = {('dense_mask', 0): (1430, 1449), ('dense_mask', 1): (1450, 1469),
          ('sparse4x', 0): (1470, 1489), ('sparse4x', 1): (1490, 1509)}
DELAY = 20


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t33():
    arms = []
    for (cell, a), rng in sorted(BLOCKS.items()):
        arm = dict(name='%s_A%d' % (cell.upper(), a), H=H,
                   kind='norm', role='block', s=2.2, wb=0.6,
                   conj_a=a, cell=cell,
                   seeds=_seeds(rng),
                   invariants=[dict(metric='ndiv', op='le',
                                    expect=6)])
        if cell == 'sparse4x':
            arm['steps'] = 2400
            arm['cell_task'] = 'sparse'
        else:
            arm['steps'] = 600
            arm['cell_task'] = 'dense'
        arms.append(arm)
    return dict(
        measures=['pf', 'colnorm', 'ndiv'],
        params=dict(
            blocks={str(k): list(v) for k, v in BLOCKS.items()},
            q_seed=Q_SEED, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            delay=DELAY, teacher_seed0=50000, t_hidden=8,
            steps4x=2400, mask_period=4),
        arms=arms,
        sanity=[dict(id='sanity-mb', text='MB a=0 lernt (dichte '
                     'Aufgabe, Kontrolle)', predicate=[dict(
                         metric='DENSE_MASK_A0.pf', op='ge', value=0.5)]),
                dict(id='sanity-s4x', text='S4X a=0 lernt (sparse '
                     'mit 4x Training, Kontrolle)', predicate=[dict(
                         metric='SPARSE4X_A0.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — DICHTE-TRAEGER (KONFUND GESCHLOSSEN)',
                 text=('P_MB(1) >= 0.5 UND P_S4X(1) <= 0.15: '
                       'dichte Aufgabenstruktur traegt bei gematchtem '
                       'Signal; mehr Training rettet sparse nicht'),
                 predicate=[dict(metric='DENSE_MASK_A0.pf', op='ge',
                                 value=0.0)]),
            dict(id='KILL — OPTIMIERUNGS-TRAEGER',
                 text=('P_MB(1) <= 0.15 UND P_S4X(1) >= 0.5: '
                       'Optimierungs-Dynamik traegt, die Dichte als '
                       'Aufgabenstruktur nicht — T32-These gekillt'),
                 predicate=[dict(metric='DENSE_MASK_A0.pf', op='lt',
                                 value=-1.0)]),
        ])