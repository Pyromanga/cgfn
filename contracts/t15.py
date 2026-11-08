'''T15-Contract — reine Daten, wortgetreu aus der T15-Registrierung
(XVII Nachtrag 15). Lektion im Schluessel-Design: REF-Arme heissen
REF24/REF96 — der historische Run-1-Bug (beide unter 'REF') ist genau
das, was das K-Gate in Test (c) verweigert.'''


def contract_t15():
    return dict(
        measures=['pf'],
        arms=[
            dict(name='REF24', H=24, invariants=[
                dict(metric='colnorm', op='approx', expect=0.5 * 24 ** 0.5),
                dict(metric='dm', op='eq', expect=7.5793, tol=1e-3)]),
            dict(name='REF96', H=96, invariants=[
                dict(metric='colnorm', op='approx', expect=0.5 * 96 ** 0.5),
                dict(metric='dm', op='eq', expect=20.6678, tol=1e-3)]),
        ] + [dict(name='R24_%s' % s, H=24, invariants=[
                dict(metric='colnorm', op='eq', expect=float(s)),
                dict(metric='dm', op='eq', expect=7.5793, tol=1e-3)])
             for s in ('2.0', '3.5', '4.9')]
          + [dict(name='R96_%s' % s, H=96, invariants=[
                dict(metric='colnorm', op='eq', expect=float(s)),
                dict(metric='dm', op='eq', expect=20.6678, tol=1e-3)])
             for s in ('2.0', '3.5', '4.9')],
        sanity=[dict(id='sanity', text='REF24 und REF96 lernen',
                     predicate=[dict(metric='REF24.pf', op='ge', value=0.5),
                                dict(metric='REF96.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — P-SQRT', text='Kante folgt 0.5*sqrt(H)',
                 predicate=[dict(metric='R24_2.0.pf', op='eq', value=0.0),
                            dict(metric='R24_3.5.pf', op='ge', value=0.5),
                            dict(metric='R96_3.5.pf', op='eq', value=0.0),
                            dict(metric='R96_4.9.pf', op='ge', value=0.5)]),
            dict(id='KILL-A — ABSOLUTE KANTE',
                 text='Kante ~3.5 unabhaengig von H',
                 predicate=[dict(metric='R96_3.5.pf', op='ge', value=0.5)]),
            dict(id='KILL-B — H=24-KANTE HOEHER',
                 text='Kante zwischen 3.5 und 4.9 auch bei H=24',
                 predicate=[dict(metric='R24_3.5.pf', op='eq', value=0.0),
                            dict(metric='R24_4.9.pf', op='ge', value=0.5)]),
        ])
