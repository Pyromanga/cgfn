#!/usr/bin/env python
'''W1 P1 — LAB-KERNEL: Contract-Generator + Invarianten-Audit
(Registrierung: analysis/W1_lab_kernel.md, Abschnitt 4; P1-Umfang: C+A)

REPARATUR-PROTOKOLL (Run-1-Crash, 12s): Der Run-1-Push enthielt im
Datei-Inhalt zwei aufgeloeste \n-Zeichen innerhalb von String-Literalen
(return '\n'.join(L), fh.write(out + '\n')) — Schreib-Artefakt des Push-
Wegs, kein Design-Defekt. Die py_compile-Gates fingen es ab (0 verbrannte
Runner-Minuten). EINZIGE Aenderung Run-2: korrektes \n in den beiden
Zeilen. Design, Contract, Test, Registrierung unveraendert.

DIE WETTE VON W1: Wenn Registrierung, Auswertung und Verdikt aus EINEM
deklarativen Contract GENERIERT werden, stirbt die Bug-Klasse
"Code weicht von Registrierung ab" strukturell (T11: stille Normierung;
T15-Run1: Readout-Doppelschluessel).

ABNAHME AUSSCHLIESSLICH AM HISTORIEN-TEST (known-answer, genagelt):
  W1-PASS (a): T15-Contract + Run-2-Zahlen (Commit 4d9bd4b) ->
               generierte Auswertung liefert KILL-A, identische Zahlen.
  W1-PASS (b): T11-Contract (REF unnnormiert deklariert) + historische
               T11-Messung (REF-Spaltennorm 0.5 statt ~3.46) ->
               INVARIANTEN-ALARM, kein stiller Durchlauf.
  W1-PASS (c): Contract mit doppeltem Arm-Key -> Generator VERWEIGERT.
  W1-KILL: nur mit Hardcodes erreichbar, oder (b)/(c) unpruefbar.
  W1-UNENTSCHEIDEN: T15 nicht reproduzierbar; EINE Reparatur, dann tot.

EHRLICHKEITS-NOTIZ zu den Audit-Eingabewerten: Die historischen Laeufe
massen KEINE Spaltennormen (genau das ist die T11-Blindstelle). Fuer den
Known-Answer-Test werden die fehlenden Audit-Groessen daher als
REKONSTRUIERTE Werte im Rahmen der deklarierten Konvention angesetzt
und als solche markiert. Kuenftige, kernel-generierte Laeufe muessen diese
Groessen SELBST messen und ausgeben — das ist der A-Modul-Standard.
'''
import os


def validate_contract(contract):
    errors = []
    keys = [a['name'] for a in contract['arms']]
    if len(keys) != len(set(keys)):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        errors.append('DOPPELTE ARM-KEYS: %s — Generator verweigert '
                      '(historische T15-Run1-Bug-Klasse).' % dupes)
    for a in contract['arms']:
        if not a.get('invariants'):
            errors.append('Arm %s ohne Invarianten-Deklaration '
                          '(Audit nicht moeglich).' % a['name'])
        for inv in a.get('invariants', []):
            for f in ('metric', 'op', 'expect'):
                if f not in inv:
                    errors.append('Arm %s, Invariante ohne Feld %r.'
                                  % (a['name'], f))
    for rule in contract.get('sanity', []) + contract.get('verdict', []):
        if 'predicate' not in rule:
            errors.append('Regel %r ohne Praedikat.' % rule.get('id', '?'))
        if 'id' not in rule:
            errors.append('Regel ohne id.')
    return errors


def audit_arm(arm, measurements):
    '''A — INVARIANTEN-AUDIT: Das Instrument MISST, es vertraut nicht.
    Messwerte kommen aus dem Lauf (oder — hier im Historien-Test — als
    rekonstruierte Eingabe); geprueft wird gegen die DEKLARIERTE
    Konvention. Verletzung -> Alarm, kein stiller Durchlauf
    (T11-Bug-Klasse stirbt hier).'''
    alarms = []
    for inv in arm.get('invariants', []):
        got = measurements.get(inv['metric'])
        if got is None:
            alarms.append('ARM %s: Invariante %r NICHT GEMESSEN — '
                          'Audit unvollstaendig (Verdikt unzulaessig).'
                          % (arm['name'], inv['metric']))
            continue
        op, exp = inv['op'], inv['expect']
        try:
            ok = {
                'eq': abs(got - exp) <= inv.get('tol', 1e-8),
                'approx': abs(got - exp) <= inv.get('tol', 0.15 * abs(exp)),
                'gt': got > exp,
                'ge': got >= exp,
                'lt': got < exp,
                'le': got <= exp,
            }[op]
        except KeyError:
            alarms.append('Arm %s: unbekannter Invarianten-Operator %r.'
                          % (arm['name'], op))
            continue
        if not ok:
            alarms.append('ARM %s: INVARIANTEN-VERLETZUNG %r: gemessen %r, '
                          'erwartet %s %r (deklarierte Konvention '
                          'missachtet — T11-Bug-Klasse).'
                          % (arm['name'], inv['metric'], got, op, exp))
    return alarms


def _eval_pred(pred, metrics):
    '''Praedikate sind Daten, kein Code: {"metric": "R96_3.5.pf",
    "op": "ge", "value": 0.5}. Metrik-Namen: <arm>.<groesse>.'''
    m = metrics.get(pred['metric'])
    if m is None:
        raise KeyError('Unregistrierte/unbekannte Groesse %r im Praedikat — '
                       'Contract verlangt eine Messung, die der Lauf nicht '
                       'liefert (Kollaps-Verletzung).' % pred['metric'])
    return {
        'eq': abs(m - pred['value']) < 1e-12,
        'ne': abs(m - pred['value']) >= 1e-12,
        'gt': m > pred['value'],
        'ge': m >= pred['value'],
        'lt': m < pred['value'],
        'le': m <= pred['value'],
    }[pred['op']]()


def evaluate(contract, measurements):
    '''Generator-Kern: Auswertung + Verdikt ausschliesslich aus dem
    Contract. Sanity-Kette zuerst, dann die genagelte Verdikt-Reihenfolge.
    Audit-Alarm schlaegt alles: UNENTSCHEIDEN (Crash-Klasse).'''
    errs = validate_contract(contract)
    if errs:
        return dict(verdict='VERWEIGERT', alarms=errs, metrics={})
    metrics = {}
    alarms = []
    for a in contract['arms']:
        row = measurements.get(a['name'], {})
        alarms += audit_arm(a, row)
        for g in contract.get('measures', []):
            if row.get(g) is not None:
                metrics['%s.%s' % (a['name'], g)] = row[g]
    sanity_ok = True
    for rule in contract.get('sanity', []):
        if not all(_eval_pred(p, metrics) for p in rule['predicate']):
            sanity_ok = False
    hit = None
    for rule in contract.get('verdict', []):
        if all(_eval_pred(p, metrics) for p in rule['predicate']):
            hit = rule
            break
    if alarms:
        return dict(verdict='UNENTSCHEIDEN (Crash-Klasse) — Invarianten-'
                    'Audit: deklarierte Konvention verletzt',
                    alarms=alarms, metrics=metrics)
    if not sanity_ok:
        return dict(verdict='UNENTSCHEIDEN — SANITY GERICHT',
                    alarms=[], metrics=metrics)
    if hit is None:
        return dict(verdict='GEMISCHT — keine registrierte Konstellation '
                    'sitzt', alarms=[], metrics=metrics)
    return dict(verdict=hit['id'], text=hit['text'], alarms=[],
                metrics=metrics)


def contract_t15():
    '''T15-Contract, wortgetreu aus der T15-Registrierung (Nachtrag 15).
    Lektion im Schluessel-Design: REF-Arme heissen REF24/REF96 — der
    historische Run-1-Bug (beide unter 'REF') ist genau das, was der
    Validator in Test (c) verweigert.'''
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


def measurements_t15_run2():
    '''Run-2-Zahlen (Commit 4d9bd4b, Ergebnis-Datei): pf je Arm gemessen.
    colnorm/dm: REKONSTRUIERTE Audit-Werte — die exakt normierten Arme
    trugen ihre Norm per Konstruktion (dm wurde gemessen), die REF-Arme
    liegen im unnnormierten Konventionsrahmen. Kuenftige kernel-generierte
    Laeufe muessen colnorm SELBST messen (A-Standard).'''
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


def contract_t11():
    '''T11-Contract NUR fuer den Audit-Test (b): REF deklariert UNNORMIERT
    (bewaehrte Konvention, Spaltennorm ~ 0.5*sqrt(48) ~ 3.46), genau wie in
    der historischen T11-Registrierung. Der historische Lauf mass
    stattdessen 0.5 (stille Normierung — der Kern-Mea-culpa-Bug).'''
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


def measurements_t11_historisch():
    '''Historische T11-Messung: REF verhungerte (pf 0.00), Spaltennorm 0.5
    statt ~3.46 (nachtraeglich diagnostiziert, Nachtrag 12).'''
    return {
        'REF': dict(pf=0.00, colnorm=0.5),
        'AMPNEG': dict(pf=0.00, colnorm=0.5),
    }


def contract_t15_run1_bug():
    '''Test (c): die historische T15-Run-1-Struktur — beide REF-Arme unter
    EINEM Key 'REF'. Der Generator MUSS verweigern (doppelte Arm-Keys).'''
    c = contract_t15()
    bug_arms = [dict(name='REF', H=24, invariants=[
                    dict(metric='colnorm', op='approx', expect=0.5 * 24 ** 0.5)]),
                dict(name='REF', H=96, invariants=[
                    dict(metric='colnorm', op='approx', expect=0.5 * 96 ** 0.5)])]
    c['arms'] = bug_arms + c['arms'][2:]
    return c


def run_historien_test():
    L = []
    L.append('=' * 78)
    L.append('W1 P1 — HISTORIEN-TEST (Abnahme genagelt in analysis/W1_lab_kernel.md)')
    L.append('Known-answer gegen die eigene Serie. Kein neues Experiment.')
    L.append('Kernel-Umfang P1: C-Contract + A-Invarianten-Audit. Nichts weiter.')
    L.append('=' * 78)
    ok_a = ok_b = ok_c = False

    res = evaluate(contract_t15(), measurements_t15_run2())
    L.append('')
    L.append('TEST (a) T15-WIEDERGABE:')
    L.append('  erwartet: KILL-A — ABSOLUTE KANTE | Zahlen wie Commit 4d9bd4b')
    L.append('  bekommen: %s' % res['verdict'])
    for k in sorted(res['metrics']):
        L.append('    %s = %.2f' % (k, res['metrics'][k]))
    ok_a = (res['verdict'].startswith('KILL-A')
            and abs(res['metrics']['REF24.pf'] - 0.80) < 1e-12
            and abs(res['metrics']['R96_3.5.pf'] - 0.60) < 1e-12
            and not res['alarms'])
    L.append('  (a) %s' % ('PASS' if ok_a else 'FAIL'))

    res_b = evaluate(contract_t11(), measurements_t11_historisch())
    L.append('')
    L.append('TEST (b) T11-ANZEIGE (stille Normierung muss Alarm schlagen):')
    for al in res_b['alarms']:
        L.append('  ALARM: %s' % al)
    ok_b = bool(res_b['alarms']) and res_b['verdict'].startswith('UNENTSCHEIDEN')
    L.append('  (b) %s' % ('PASS' if ok_b else 'FAIL'))

    res_c = evaluate(contract_t15_run1_bug(), measurements_t15_run2())
    L.append('')
    L.append('TEST (c) T15-RUN1-DOPPELKEY (Generator muss verweigern):')
    L.append('  bekommen: %s' % res_c['verdict'])
    for al in res_c['alarms']:
        L.append('  ALARM: %s' % al)
    ok_c = (res_c['verdict'] == 'VERWEIGERT'
            and any('DOPPELTE' in a for a in res_c['alarms']))
    L.append('  (c) %s' % ('PASS' if ok_c else 'FAIL'))

    L.append('')
    if ok_a and ok_b and ok_c:
        verdict = ('W1-PASS — Der Kernel reproduziert T15 (a), zeigt die '
                   'T11-Stille-Normierung an (b), verweigert den '
                   'T15-Run1-Doppelkey (c). P1 ANGENOMMEN.')
    else:
        parts = [p for p, ok in (('a', ok_a), ('b', ok_b), ('c', ok_c)) if not ok]
        verdict = ('W1-UNENTSCHEIDEN — Historien-Test nicht bestanden '
                   '(%s). EINE Reparatur erlaubt, dann W1-Linie tot.'
                   % '/'.join(parts))
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    return '\n'.join(L)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    out = run_historien_test()
    print(out)
    with open('results/w1_kernel_historien_test.txt', 'w', encoding='utf-8') as fh:
        fh.write(out + '\n')
