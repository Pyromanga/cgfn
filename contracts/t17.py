'''T17-Contract — reine Daten (registriert XVII Nachtrag 24).
Zweiter kernel-generierter Stein: s50-Extrapolationstest bei H=192.
Verdikt generiert vom W1-Kernel; Instrument liest alles hier.

THESE (P-SUBSQRT-EXTRAPOLATION): s50(192)/s50(96) strikt in (1.0,
1.4142) — die Sub-Sqrt-Ordnung aus T16 (ratio50=1.75 < sqrt(4))
extrapoliert. Anker: T16 s50(96) = 3.5 (Commit 731933f).'''

S_GRID = [2.8, 3.5, 4.4, 5.5, 7.0]
H = 192
S50_REF96 = 3.5  # T16-Anker (results/t16_measurements.json, Commit 731933f)


def _inv_ref():
    return [dict(metric='colnorm', op='approx', expect=0.5 * H ** 0.5),
            dict(metric='dmdev', op='eq', expect=0.0, tol=1e-9),
            dict(metric='ndiv', op='le', expect=3)]


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='dmdev', op='eq', expect=0.0, tol=1e-9),
            dict(metric='ndiv', op='le', expect=3)]


def contract_t17():
    arms = [dict(name='REF192', H=H, kind='ref', s=None,
                 invariants=_inv_ref())]
    for s in S_GRID:
        arms.append(dict(name='R192_%s' % s, H=H, kind='norm', s=s,
                         invariants=_inv_norm(s)))
    return dict(
        measures=['pf', 'colnorm', 'dmdev', 'ndiv', 's50', 'ratio2'],
        params=dict(
            seeds=list(range(170, 180)), H_list=[H], s_grid=list(S_GRID),
            s50_ref96=S50_REF96, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity', text='REF192 lernt',
                     predicate=[dict(metric='REF192.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — P-SUBSQRT-EXTRAPOLATION',
                 text='ratio2 in (1.0, 1.4142): Sub-Sqrt-Ordnung '
                      'extrapoliert nach H=192',
                 predicate=[dict(metric='REF192.ratio2', op='gt', value=1.0),
                            dict(metric='REF192.ratio2', op='lt', value=1.4142)]),
            dict(id='KILL-A — SQRT-ARTIG ODER STAERKER',
                 text='ratio2 >= 1.4142: H-Skalierung naeher an sqrt(H), '
                      'Sub-Sqrt extrapoliert nicht',
                 predicate=[dict(metric='REF192.ratio2', op='ge', value=1.4142)]),
            dict(id='KILL-B — KEIN WEITERER SHIFT',
                 text='ratio2 <= 1.0: Die Ramme stoppt zwischen H=96 und 192',
                 predicate=[dict(metric='REF192.ratio2', op='le', value=1.0)]),
        ])
