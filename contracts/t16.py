'''T16-Contract — reine Daten (registriert XVII Nachtrag 22).
ERSTER KERNEL-GENERIERTER STEIN: Arme, Seeds, s-Raster, Task-
Konstanten, Invarianten, Sanity und Verdikt-Logik liegen HIER und
nirgendwo sonst. Das Instrument (experiments/t16_2d_karte.py) liest
alles aus diesem Contract; Verdikt generiert tools/lab_kernel.py.

THESE (P-RAMPE-SHIFT-SUBH): Die Lernbarkeits-Rampe verschiebt sich mit
H (s50 waechst), ABER sub-proportional zu H: ratio50 in (1, 4).'''

S_GRID = [1.5, 2.0, 2.8, 3.5, 4.4, 5.5]
H_LIST = [24, 96]


def _inv_ref(H):
    return [dict(metric='colnorm', op='approx', expect=0.5 * H ** 0.5),
            dict(metric='dmdev', op='eq', expect=0.0, tol=1e-9),
            dict(metric='ndiv', op='le', expect=3)]


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='dmdev', op='eq', expect=0.0, tol=1e-9),
            dict(metric='ndiv', op='le', expect=3)]


def contract_t16():
    arms = [dict(name='REF24', H=24, kind='ref', s=None,
                 invariants=_inv_ref(24)),
            dict(name='REF96', H=96, kind='ref', s=None,
                 invariants=_inv_ref(96))]
    for h in H_LIST:
        for s in S_GRID:
            arms.append(dict(name='R%d_%s' % (h, s), H=h, kind='norm',
                             s=s, invariants=_inv_norm(s)))
    return dict(
        measures=['pf', 'colnorm', 'dmdev', 'ndiv', 's50', 'ratio50'],
        params=dict(
            seeds=list(range(160, 170)), H_list=list(H_LIST),
            s_grid=list(S_GRID), fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity', text='REF24 und REF96 lernen',
                     predicate=[dict(metric='REF24.pf', op='ge', value=0.5),
                                dict(metric='REF96.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — P-RAMPE-SHIFT-SUBH',
                 text='Rampe verschiebt sich mit H, sub-proportional '
                      '(ratio50 strikt in (1, 4))',
                 predicate=[dict(metric='REF96.ratio50', op='gt', value=1.0),
                            dict(metric='REF96.ratio50', op='lt', value=4.0)]),
            dict(id='KILL-A — KEIN/INVERSER SHIFT',
                 text='ratio50 <= 1: Rampe verschiebt sich nicht (oder '
                      'invers) mit H — T15-Richtung auf Rampen-Ebene falsch',
                 predicate=[dict(metric='REF96.ratio50', op='le', value=1.0)]),
            dict(id='KILL-B — H-PROPORTIONAL',
                 text='ratio50 >= 4: Rampe skaliert mit H — sqrt-artige '
                      'Rezepte unzureichend',
                 predicate=[dict(metric='REF96.ratio50', op='ge', value=4.0)]),
        ])
