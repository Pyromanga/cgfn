'''T23-Contract — SCHALTER-DOSIS v1.1 (registriert XVII Nachtrag 40,
maschinengeneriert vom P4-Generator v1.1, L5-Durchlauf 2).
EINE Zelle (H=48, s=2.2, wb=0.6), Konjugationstiefe a in
{0, 1, 4, 24} via W' = Q_a W Q_a^T (Spektrum identisch, Q aus
QR(I + a*G), Q_SEED=2020 wie T20). Je Stufe 2 disjunkte
20er-Bloecke, frische Seeds 430..589; Anchor HIGH48 (s=5.5,
Seeds 590..609). Verdikt-Mass drop1 = P(0) - P(1), registriert
an JEDEM Arm (R2); P_Stufe = Mittel der 2 Block-P je Stufe.
  PASS (SCHALTER-BESTAETIGT): drop1 >= 0.30 (>= 3.1 SE)
  KILL (KEIN-KOLLAPS):        drop1 < 0.30 (T20-Artefakt-These)
  UNENTSCHEIDEN (Crash): R6-Sanity (Anchor pf < 0.5),
    ndiv > 6 je Block, std(P_Block) < 0.02 je Stufe,
    colnorm-Verletzung. EINE Reparatur (R5), dann Linie tot.
Band (mechanisch aus Block-Statistik, Nachtrag 40): P(0) in
[0.31, 0.58] | P(1) in [0.00, 0.25] | P(4) in [0.00, 0.30] |
P(24) in [0.00, 0.25] | drop1 in [0.05, 0.55] | drop124 in
[-0.10, 0.20] (Form, SEKUNDAER).
Instrument: experiments/t23_schalter_dosis.py; Verdikt generiert
vom W1-Kernel. Rollen: anchor (SANITY), block (Stufen-Bloecke).'''

H = 48
S = 2.2
WB = 0.6
Q_SEED = 2020
S_ANCHOR = 5.5
STAGES = [0, 1, 4, 24]
BLOCKS = {0: [(430, 449), (450, 469)], 1: [(470, 489), (490, 509)],
          4: [(510, 529), (530, 549)], 24: [(550, 569), (570, 589)]}


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='ndiv', op='le', expect=6)]


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t23():
    arms = [dict(name='HIGH48', H=H, kind='norm', role='anchor',
                 s=S_ANCHOR, wb=WB, conj_a=0, seeds=_seeds((590, 609)),
                 invariants=_inv_norm(S_ANCHOR))]
    for a in STAGES:
        for bi, rng in enumerate(BLOCKS[a]):
            arms.append(dict(name='A%d_B%d' % (a, bi), H=H, kind='norm',
                             role='block', s=S, wb=WB, conj_a=a,
                             seeds=_seeds(rng),
                             invariants=_inv_norm(S)))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'pstage', 'drop1'],
        params=dict(
            stages=list(STAGES), blocks={str(k): [list(v) for v in BLOCKS[k]] for k in BLOCKS},
            seeds_anchor=_seeds((590, 609)), q_seed=Q_SEED,
            s_anchor=S_ANCHOR, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=WB,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            drop1_threshold=0.30, block_std_min=0.02),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit oberhalb '
                     'der Rampe, R6)', predicate=[dict(
                         metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — SCHALTER-BESTAETIGT',
                 text='drop1 >= 0.30: Der a=1-Kollaps (T20) repliziert '
                      'auf frischen Bloecken — der Schalter ist real und '
                      'extrem scharf (Strukturbruch bei a=1)',
                 predicate=[dict(metric='A0_B0.drop1', op='ge', value=0.30)]),
            dict(id='KILL — KEIN-KOLLAPS',
                 text='drop1 < 0.30: Der T20-Schalter war ein Artefakt '
                      'des Ausreisser-Blocks 190..209 (T22-Lektion); '
                      'Schalter-Linie TOT',
                 predicate=[dict(metric='A0_B0.drop1', op='lt', value=0.30)]),
        ])
