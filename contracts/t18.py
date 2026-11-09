'''T18-Contract — reine Daten (registriert XVII Nachtrag 26).
ERSTER VORHERSAGE-STEIN: Das Theory-Core-Modell (s50 ~ 0.546*H^0.408,
Nachtrag 25) sagt s50(12) ~ 1.51 voraus; der Kernel verdiktet.
Verdikt generiert vom W1-Kernel; Instrument liest alles hier.

Rollen: ref (Konventions-Bericht, KEIN Sanity — liegt prognostiziert
auf der Rampe), anchor (SANITY weit oberhalb), sgrid (s50-Kurve).'''

S_GRID = [0.8, 1.1, 1.5, 1.9, 2.4]
H = 12
S_ANCHOR = 3.5


def _inv_ref():
    return [dict(metric='colnorm', op='approx', expect=0.5 * H ** 0.5),
            dict(metric='dmdev', op='eq', expect=0.0, tol=1e-9),
            dict(metric='ndiv', op='le', expect=3)]


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='dmdev', op='eq', expect=0.0, tol=1e-9),
            dict(metric='ndiv', op='le', expect=3)]


def contract_t18():
    arms = [dict(name='REF12', H=H, kind='ref', role='ref', s=None,
                 invariants=_inv_ref()),
            dict(name='HIGH12', H=H, kind='norm', role='anchor', s=S_ANCHOR,
                 invariants=_inv_norm(S_ANCHOR))]
    for s in S_GRID:
        arms.append(dict(name='R12_%s' % s, H=H, kind='norm', role='sgrid',
                         s=s, invariants=_inv_norm(s)))
    return dict(
        measures=['pf', 'colnorm', 'dmdev', 'ndiv', 's50'],
        params=dict(
            seeds=list(range(180, 190)), H_list=[H], s_grid=list(S_GRID),
            s_anchor=S_ANCHOR, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH12 lernt (Anker weit oberhalb der Rampe)',
                     predicate=[dict(metric='HIGH12.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — P-GESCHLOSSENE-VORHERSAGE',
                 text='s50(12) in [1.1, 1.9]: Theory-Core-Vorhersage '
                      'bestaetigt (s50 ~ 0.546*H^0.408)',
                 predicate=[dict(metric='REF12.s50', op='ge', value=1.1),
                            dict(metric='REF12.s50', op='lt', value=1.9)]),
            dict(id='KILL-A — VORHERSAGE ZU HOCH',
                 text='s50(12) < 1.1: Netz lernt frueher als das Gesetz',
                 predicate=[dict(metric='REF12.s50', op='lt', value=1.1)]),
            dict(id='KILL-B — VORHERSAGE ZU NIEDRIG',
                 text='s50(12) >= 1.9: Netz lernt spaeter als das Gesetz',
                 predicate=[dict(metric='REF12.s50', op='ge', value=1.9)]),
        ])
