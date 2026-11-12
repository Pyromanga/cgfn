'''T25-Contract — ORDNUNGSPARAMETER v1 (registriert XVII
Nachtrag 45, maschinengeneriert vom P4-Generator v1.3,
L5-Durchlauf 4). EINE Zelle (H=48, s=2.2, wb=0.6), Stufen a in
{0, 1, 4, 24} via W' = Q_a W Q_a^T (Spektrum identisch,
Q_SEED=2020 wie T20), je 1 frischer 20er-Block (750..829);
Anchor HIGH48 (s=5.5, Seeds 830..849). DM-Definition
unveraendert aus T20: dm_of(W) = W.diag().abs().mean() /
W.abs().mean() (Kollaps dort 13.21 -> 1.40 bei a=1).
PRIMAER: dmdrop = DM(0) - DM(1) mit DOPPELKLAUSEL (R1, da
DM-SE ungetestet): PASS nur wenn dmdrop >= 5.0 UND
dmdrop >= 3 * SE(dmdrop). Kodierung als Verdict-Mass
dmeff = dmdrop falls Doppelklausel erfuellt, sonst -1
(registriert an JEDEM Arm, R2).
  PASS — ORDNUNGSPARAMETER-BESTAETIGT: dmeff >= 5.0
  KILL — DM-NEBENPRODUKT: dmeff < 5.0
  UNENTSCHEIDEN (Crash): R6-Sanity (Anchor pf < 0.5),
    ndiv > 6 je Block, colnorm-Verletzung. EINE Reparatur
    (R5), dann Linie tot. R4 erfuellt (PASS -> DM-Kippregel;
    KILL -> Gain-Verteilung als naechster Kandidat).
Band (mechanisch, R8-Haertung, Nachtrag 45): DM(0) in
[8.80, 19.80] | DM(1) in [0.90, 2.10] | dmdrop in
[5.90, 17.70] (alle: T20 n=1, +50% Weite) | P(0) in
[0.25, 0.64] | P(1) in [0.00, 0.22] | P(4) in [0.00, 0.30] |
P(24) in [0.00, 0.25] (T23 frisch, n=1, R8).
KEINE std-Klausel: dmeff traegt die SE-Klausel in sich
(Praezedenz Nachtrag 42). Instrument:
experiments/t25_ordnungsparameter.py; Verdikt generiert vom
W1-Kernel. Rollen: anchor (SANITY), block (Stufen-Bloecke).'''

H = 48
S = 2.2
WB = 0.6
Q_SEED = 2020
S_ANCHOR = 5.5
STAGES = [0, 1, 4, 24]
BLOCKS = {0: (750, 769), 1: (770, 789), 4: (790, 809),
          24: (810, 829)}
ANCHOR_RANGE = (830, 849)
DM_PASS = 5.0
DM_SE_FACTOR = 3


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='ndiv', op='le', expect=6)]


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t25():
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
        measures=['pf', 'colnorm', 'ndiv', 'dm', 'dmdrop', 'dmeff'],
        params=dict(
            stages=list(STAGES),
            blocks={str(k): list(BLOCKS[k]) for k in BLOCKS},
            seeds_anchor=_seeds(ANCHOR_RANGE), q_seed=Q_SEED,
            s_anchor=S_ANCHOR, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=WB,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            dm_pass=DM_PASS, dm_se_factor=DM_SE_FACTOR),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit '
                     'oberhalb der Rampe, R6)', predicate=[dict(
                         metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — ORDNUNGSPARAMETER-BESTAETIGT',
                 text='dmeff >= 5.0 (dmdrop = DM(0) - DM(1) >= 5.0 '
                      'UND >= 3 * SE): DM kollabiert mit dem Schalter '
                      '— Vorhersage-Instrument; naechster Stein: '
                      'DM-Schwellwert-Kippregel pro Seed',
                 predicate=[dict(metric='A0.dmeff', op='ge',
                                 value=5.0)]),
            dict(id='KILL — DM-NEBENPRODUKT',
                 text='dmeff < 5.0 (dmdrop zu klein oder < 3 * SE): '
                      'DM als Diagnostikum gefallen (konsistent mit '
                      'T20-R2 KILL DM-NEBENSPUR); naechster Kandidat: '
                      'Gain-Verteilung',
                 predicate=[dict(metric='A0.dmeff', op='lt',
                                 value=5.0)]),
        ])
