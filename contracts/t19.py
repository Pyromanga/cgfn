'''T19-Contract — R2-REPARATUR (registriert XVII Nachtrag 28).
Einzige Aenderung gegen Nachtrag 27: Das s-Raster liegt NUR UNTEN
dichter und weiter ({0.7 .. 2.2}), weil Lauf 1 zeigte, dass
P_lernt(1.9)=0.55 bereits ueber der 0.5-Linie liegt — das alte
Raster (um die Vorhersage 2.55) war Prediction-Lock-in.
ALLES ANDERE GENAGELT: Seeds 190..209, Vorhersage s50(48) ~ 2.55,
Band [2.05, 3.05), KILL-A/KILL-B wie Nachtrag 27. Verdikt
generiert vom W1-Kernel; Instrument liest alles hier.
Nach UNENTSCHEIDEN R2 ist die Linie TOT (keine zweite Reparatur).

Rollen: ref (Konventions-Bericht, KEIN Sanity), anchor (SANITY
weit oberhalb), sgrid (s50-Kurve, dicht unten).'''

S_GRID = [0.7, 0.9, 1.1, 1.3, 1.55, 1.9, 2.2]
H = 48
S_ANCHOR = 5.5


def _inv_ref():
    return [dict(metric='colnorm', op='approx', expect=0.5 * H ** 0.5),
            dict(metric='dmdev', op='eq', expect=0.0, tol=1e-9),
            dict(metric='ndiv', op='le', expect=6)]


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='dmdev', op='eq', expect=0.0, tol=1e-9),
            dict(metric='ndiv', op='le', expect=6)]


def contract_t19():
    arms = [dict(name='REF48', H=H, kind='ref', role='ref', s=None,
                 invariants=_inv_ref()),
            dict(name='HIGH48', H=H, kind='norm', role='anchor', s=S_ANCHOR,
                 invariants=_inv_norm(S_ANCHOR))]
    for s in S_GRID:
        arms.append(dict(name='R48_%s' % s, H=H, kind='norm', role='sgrid',
                         s=s, invariants=_inv_norm(s)))
    return dict(
        measures=['pf', 'colnorm', 'dmdev', 'ndiv', 's50'],
        params=dict(
            seeds=list(range(190, 210)), H_list=[H], s_grid=list(S_GRID),
            s_anchor=S_ANCHOR, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit oberhalb der Rampe)',
                     predicate=[dict(metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — P-CORE-VORHERSAGE-2',
                 text='s50(48) in [2.05, 3.05): Theory-Core trifft die '
                      'zweite Vorhersage (+-20 Prozent um 2.55)',
                 predicate=[dict(metric='REF48.s50', op='ge', value=2.05),
                            dict(metric='REF48.s50', op='lt', value=3.05)]),
            dict(id='KILL-A — ZU KLEIN',
                 text='s50(48) < 2.05: beide Formen zu optimistisch, '
                      'Kurve biegt staerker (R2: Moegliche '
                      'Nicht-Monotonie gegen s50(24)=2.00)',
                 predicate=[dict(metric='REF48.s50', op='lt', value=2.05)]),
            dict(id='KILL-B — ZU GROSS',
                 text='s50(48) >= 3.05: Kurve steiler als beide Formen',
                 predicate=[dict(metric='REF48.s50', op='ge', value=3.05)]),
        ])
