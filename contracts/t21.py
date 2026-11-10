'''T21-Contract — DELLE-STATISTIK (registriert XVII Nachtrag 33).
Letzte offene Messfrage der T-Linie nach T19-R2-KILL-A und
T20-R2-KILL: Ist die s50-Delle bei H=48 REAL oder Seed-Rauschen?
Frischer Seed-Block 210..309 (100, DISJUNKT zu allen frueheren
Bloecken 190..209). H in {24, 48} x s in {1.55, 1.83, 2.2}.
PRIMAER (VERDIKT): d183 = P_lernt(48, 1.83) - P_lernt(24, 1.83),
  registriert an jedem Arm (SE(d183) ~ 0.07, Schwelle = 2 Sigma).
  PASS — DELLE-RECHT: d183 >= +0.15 (H=48 lernt frueher, Delle real)
  KILL — RAUSCHEN:    -0.15 < d183 < +0.15 (Kurven identisch)
  KEHRUNG:            d183 <= -0.15 (H=24 lernt frueher, ueberraschend)
SEKUNDAER (Bericht, kein Crash): s50rep je H (3-Punkt-Kurve,
  -1.0 = ZENSIERT), median acc, volle Kurven.
UNENTSCHEIDEN (Crash): Sanity gerissen, ndiv > 6 von 100,
colnorm-Verletzung. Rollen: anchor (Sanity), sgrid (H x s).'''

H_LIST = [24, 48]
S_ANCHOR = 5.5
S_GRID = [1.55, 1.83, 2.2]


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='ndiv', op='le', expect=6)]


def contract_t21():
    arms = [dict(name='HIGH48', H=48, kind='norm', role='anchor',
                 s=S_ANCHOR, wb=0.6, invariants=_inv_norm(S_ANCHOR))]
    for h in H_LIST:
        for s in S_GRID:
            arms.append(dict(name='H%s_s%s' % (h, s), H=h, kind='norm',
                             role='sgrid', s=s, wb=0.6,
                             invariants=_inv_norm(s)))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'd183', 's50rep'],
        params=dict(
            seeds=list(range(210, 310)), H_list=list(H_LIST),
            s_grid=list(S_GRID), s_anchor=S_ANCHOR,
            fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, a_lo=4, a_hi=8,
            b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit oberhalb der Rampe)',
                     predicate=[dict(metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — DELLE-RECHT',
                 text='d183 >= +0.15: P_lernt(48, 1.83) liegt signifikant '
                      'ueber P_lernt(24, 1.83) — die Delle ist real',
                 predicate=[dict(metric='H48_s1.83.d183', op='ge', value=0.15)]),
            dict(id='KILL — RAUSCHEN',
                 text='-0.15 < d183 < +0.15: beide Kurven identisch '
                      'innerhalb 2 Sigma — die Delle war Seed-Lotto',
                 predicate=[dict(metric='H48_s1.83.d183', op='lt', value=0.15),
                            dict(metric='H48_s1.83.d183', op='ge', value=-0.15)]),
            dict(id='KEHRUNG',
                 text='d183 <= -0.15: H=24 lernt FRUEHER — Vorzeichen '
                      'ueberraschend, Bericht und neue Frage',
                 predicate=[dict(metric='H48_s1.83.d183', op='le', value=-0.15)]),
        ])
