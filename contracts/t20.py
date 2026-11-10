'''T20-Contract — DM-KOVARIATE (registriert XVII Nachtrag 30).
Kausal-Test bei festem H=48: DM variieren via Basis-Konjugation
W' = Q_a W Q_a^T (Spektrum identisch, Diagonalmasse veraendert),
s50 je Stufe messen. Verdikt-Mass ist s50spread = max - min der
Stufen-s50, an jedem Arm als registriertes Mass.
  PASS (DM-KAUSAL):    s50spread >= 0.2
  KILL (DM-NEBENSPUR): s50spread < 0.2
  UNENTSCHEIDEN (Crash): DM-Bereich < 10 Prozent, Stufen-Raster
    verfehlt, Sanity gerissen, ndiv > 6, colnorm-Verletzung.
    EINE Reparatur, dann Linie tot.
Instrument: experiments/t20_dm_kovariate.py; Verdikt generiert vom
W1-Kernel. Rollen: ref (Bericht), anchor (SANITY), sgrid (Stufen-
s50-Kurven).'''

H = 48
S_ANCHOR = 5.5
S_GRID = [1.3, 1.55, 1.83, 2.2]
CONJ_A = [0, 1, 4, 12]
Q_SEED = 2020


def _inv_ref():
    return [dict(metric='colnorm', op='approx', expect=0.5 * H ** 0.5),
            dict(metric='ndiv', op='le', expect=6)]


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='ndiv', op='le', expect=6)]


def contract_t20():
    arms = [dict(name='REF48', H=H, kind='ref', role='ref', s=None,
                 conj_a=0, invariants=_inv_ref()),
            dict(name='HIGH48', H=H, kind='norm', role='anchor', s=S_ANCHOR,
                 conj_a=0, invariants=_inv_norm(S_ANCHOR))]
    for a in CONJ_A:
        for s in S_GRID:
            arms.append(dict(name='Q%d_s%s' % (a, s), H=H, kind='norm',
                             role='sgrid', s=s, conj_a=a,
                             invariants=_inv_norm(s)))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 's50', 's50spread'],
        params=dict(
            seeds=list(range(190, 210)), H_list=[H], s_grid=list(S_GRID),
            conj_a=list(CONJ_A), q_seed=Q_SEED,
            s_anchor=S_ANCHOR, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit oberhalb der Rampe)',
                     predicate=[dict(metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — DM-KAUSAL',
                 text='s50spread >= 0.2: s50 hoert auf DM bei festem H — '
                      'H war nie die kausale Variable',
                 predicate=[dict(metric='Q0_s1.3.s50spread', op='ge', value=0.2)]),
            dict(id='KILL — DM-NEBENSPUR',
                 text='s50spread < 0.2: s50 bleibt bei ~1.83 ueber alle '
                      'DM-Stufen — DM ist Kovariate ohne Kausalitaet',
                 predicate=[dict(metric='Q0_s1.3.s50spread', op='lt', value=0.2)]),
        ])
