'''T11-Contract — reine Daten, NUR fuer den Audit-Test (b): REF
deklariert UNNORMIERT (bewaehrte Konvention, Spaltennorm ~ 0.5*sqrt(48)
~ 3.46), genau wie in der historischen T11-Registrierung. Der
historische Lauf mass stattdessen 0.5 (stille Normierung — der
Kern-Mea-culpa-Bug).'''


def contract_t11():
    return dict(
        measures=['pf'],
        arms=[
            dict(name='REF', H=48, invariants=[
                dict(metric='colnorm', op='approx', expect=0.5 * 48 ** 0.5)]),
            dict(name='AMPNEG', H=48, invariants=[
                dict(metric='colnorm', op='eq', expect=0.5)]),
        ],
        sanity=[dict(id='sanity', text='REF lernt',
                     predicate=[dict(metric='REF.pf', op='ge', value=0.5)])],
        verdict=[])
