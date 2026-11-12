'''T26-Contract — PER-SEED-ORDNUNG v1 (registriert XVII
Nachtrag 47, maschinengeneriert vom P4-Generator v1.4,
L5-Durchlauf 5). EINE Zelle (H=48, s=2.2, wb=0.6), Stufen a in
{0, 1} via W' = Q_a W Q_a^T (Q_SEED=2020 wie T20/T25), je 1
frischer 20er-Block (a=0: 850..869, a=1: 870..889); Anchor
HIGH48 (s=5.5, Seeds 890..909). FRAGE: Trennt UEBERHAUPT eine
messbare Groesse gelernte von gescheiterten Seeds INNERHALB
eines Blocks? T25-Beyfang: DM gelernt 13.133 (n=11) vs
gescheitert 12.974 (n=9), Differenz 0.159 < SE — kein Trenner.
Kandidaten (VOR dem Readout fixiert, Nachtrag 47): dm,
initloss, gainmean, gainstd, wnorm.
SEP-ENKODING (Doppelklausel R1): je Kandidat c wird
t_c = |m_gelernt - m_gescheitert| / SE_diff (im Lauf gemessen,
zweiseitig — Richtung ist NICHT vorab bekannt und wird
nicht nachverhandelt) berechnet; sepmax = max t_c ueber
Kandidaten mit t_c >= 3 UND gleichem Vorzeichen der Differenz
in a=0 und a=1 (Richtungs-Konstanz), sonst -1. Kodierung als
Verdict-Mass sepmax an JEDEM Arm (R2).
  PASS — INTRA-BLOCK-MARKER: sepmax >= 3.0
  KILL — INTRA-BLOCK-LOTTO: sepmax < 3.0
  UNENTSCHEIDEN (Crash): R6-Sanity (Anchor pf < 0.5),
    ndiv > 6 je Block, colnorm-Verletzung. EINE Reparatur
    (R5), dann Linie tot. R4 erfuellt (PASS -> T27-Kippregel;
    KILL -> Per-Seed NEGATIV, Gain als Ebenen-Kandidat).
Band (mechanisch, R8-Haertung, Nachtrag 47): DM gelernt in
[8.80, 19.70] | DM gescheitert in [8.60, 19.50] | Differenz
DM in [0.11, 0.24] | P(0) in [0.25, 0.64] | P(1) in [0.00,
0.22]. Instrument: experiments/t26_per_seed_ordnung.py;
Verdikt generiert vom W1-Kernel. Rollen: anchor (SANITY),
block (Stufen-Bloecke).'''

H = 48
S = 2.2
WB = 0.6
Q_SEED = 2020
S_ANCHOR = 5.5
STAGES = [0, 1]
BLOCKS = {0: (850, 869), 1: (870, 889)}
ANCHOR_RANGE = (890, 909)
SEP_PASS = 3.0


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='ndiv', op='le', expect=6)]


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t26():
    arms = [dict(name='HIGH48', H=H, kind='norm', role='anchor',
                 s=S_ANCHOR, wb=WB, conj_a=0,
                 seeds=_seeds(ANCHOR_RANGE),
                 invariants=_inv_norm(S_ANCHOR))]
    for a in STAGES:
        arms.append(dict(name='A%d' % a, H=H, kind='norm',
                         role='block', s=S, wb=WB, conj_a=a,
                         seeds=_seeds(BLOCKS[a]),
                         invariants=_inv_norm(S)))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'dm', 'initloss',
                  'gainmean', 'gainstd', 'wnorm', 'sepmax'],
        params=dict(
            stages=list(STAGES),
            blocks={str(k): list(BLOCKS[k]) for k in BLOCKS},
            seeds_anchor=_seeds(ANCHOR_RANGE), q_seed=Q_SEED,
            s_anchor=S_ANCHOR, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=WB,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            sep_pass=SEP_PASS,
            candidates=['dm', 'initloss', 'gainmean',
                        'gainstd', 'wnorm']),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit '
                     'oberhalb der Rampe, R6)', predicate=[dict(
                         metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — INTRA-BLOCK-MARKER',
                 text=('sepmax >= 3.0: mindestens EINE Kandidat-'
                       'Groesse trennt gelernt vs gescheitert im '
                       'a=0-Block mit >= 3 * SE (im Lauf gemessen), '
                       'Richtung konstant in a=1 — T27-Kippregel '
                       'pro Seed (Vorhersage vor dem Training)'),
                 predicate=[dict(metric='A0.sepmax', op='ge',
                                 value=3.0)]),
            dict(id='KILL — INTRA-BLOCK-LOTTO',
                 text=('sepmax < 3.0: keine Kandidat-Groesse trennt '
                       '>= 3 * SE — Per-Seed-Frage NEGATIV '
                       'geschlossen; DM bleibt Ebenen-Marker; '
                       'naechste Frage: Gain-Verteilung als '
                       'Ebenen-Kandidat'),
                 predicate=[dict(metric='A0.sepmax', op='lt',
                                 value=3.0)]),
        ])
