'''Historien-Fixtures (reine Daten): Messwerte und Bug-Rekonstruktionen
aus der Serie — NUR fuer den Known-Answer-Test, kein neues Experiment.'''
from contracts.t15 import contract_t15


def measurements_t15_run2():
    '''Run-2-Zahlen (Commit 4d9bd4b, Ergebnis-Datei): pf je Arm gemessen.
    colnorm/dm: REKONSTRUIERTE Audit-Werte — die exakt normierten Arme
    trugen ihre Norm per Konstruktion (dm wurde gemessen), die REF-Arme
    liegen im unnnormierten Konventionsrahmen. Kuenftige kernel-
    generierte Laeufe muessen colnorm SELBST messen (A-Standard).'''
    return {
        'REF24': dict(pf=0.80, colnorm=2.5129, dm=7.5793),
        'REF96': dict(pf=1.00, colnorm=4.9044, dm=20.6678),
        'R24_2.0': dict(pf=0.70, colnorm=2.0, dm=7.5793),
        'R24_3.5': dict(pf=0.90, colnorm=3.5, dm=7.5793),
        'R24_4.9': dict(pf=1.00, colnorm=4.9, dm=7.5793),
        'R96_2.0': dict(pf=0.10, colnorm=2.0, dm=20.6678),
        'R96_3.5': dict(pf=0.60, colnorm=3.5, dm=20.6678),
        'R96_4.9': dict(pf=0.80, colnorm=4.9, dm=20.6678),
    }


def measurements_t11_historisch():
    '''Historische T11-Messung: REF verhungerte (pf 0.00), Spaltennorm 0.5
    statt ~3.46 (nachtraeglich diagnostiziert, Nachtrag 12).'''
    return {
        'REF': dict(pf=0.00, colnorm=0.5),
        'AMPNEG': dict(pf=0.00, colnorm=0.5),
    }


def contract_t15_run1_bug():
    '''Test (c): die historische T15-Run-1-Struktur — beide REF-Arme
    unter EINEM Key 'REF'. Das K-Gate MUSS verweigern.'''
    c = contract_t15()
    bug_arms = [dict(name='REF', H=24, invariants=[
                    dict(metric='colnorm', op='approx',
                         expect=0.5 * 24 ** 0.5)]),
                dict(name='REF', H=96, invariants=[
                    dict(metric='colnorm', op='approx',
                         expect=0.5 * 96 ** 0.5)])]
    c['arms'] = bug_arms + c['arms'][2:]
    return c


def contract_unregistriertes_mass():
    '''Test (k1): Praedikat auf ein Mass, das der Contract NICHT
    registriert ("schaerfe") — die operationalisierte T15-"scharf"-
    Klasse. Die Registrierung verlangt eine Messung, die kein Gate
    liefern kann -> Pre-Flight muss VERWEIGERN.'''
    c = contract_t15()
    c['verdict'].append(dict(
        id='PASS — SCHARF-TEST', text='nur fuer K-Gate-Test (k1)',
        predicate=[dict(metric='R96_3.5.schaerfe', op='ge', value=0.8)]))
    return c
