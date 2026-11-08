#!/usr/bin/env python
'''W1 P2 — LAB-KERNEL ENGINE: C-Contract-Generator, A-Invarianten-Audit,
K-Kollaps-Gate, D-Determinismus-Diff, N-Auto-Nachtrag.
(Registrierung: analysis/W1_lab_kernel.md Abschnitt 2/3; P2-Abnahme
genagelt in XVII Nachtrag 20. Architektur-Split aus Nachtrag 19 hier
eingeloest: Engine in tools/, Contracts in contracts/ (reine Daten),
Historien-Fixtures + Runner in tests/.)

DIE WETTE VON W1 (unveraendert seit P0): Wenn Registrierung, Auswertung
und Verdikt aus EINEM deklarativen Contract GENERIERT werden, stirbt die
Bug-Klasse "Code weicht von Registrierung ab" strukturell (T11: stille
Normierung; T15-Run1: Readout-Doppelschluessel).

P2-NEU gegenueber P1:
  K — preflight_gate: verweigert Contracts mit doppelten Arm-Keys,
      Praedikaten auf unbekannte Arme oder UNREGISTRIERTE Masse
      (operationalisierte "scharf"-Klasse), Regeln ohne id/Praedikat,
      Armen ohne Invarianten-Deklaration.
      EHRLICHE GRENZE: Freitext-Begriffe ohne messbare Operationalisierung
      prueft keine Maschine; K prueft die datenfassbare Variante, der
      Rest bleibt Menschenpflicht.
  K-Laufzeit — evaluate schlaegt NICHT-GEMESSEN-Alarm, wenn eine
      registrierte Groesse im Mess-Input fehlt (kein stilles Loch).
  D — determinismus_diff: Rerun gegen letzten Result-Commit; leer bei
      identisch, je Abweichung ein Flag (Umwelt-Bug-Verdacht).
  N — auto_nachtrag: Zahlen-Skelett mit Verdikt und LUECKENLISTE.

PROTOKOLL-HINWEIS P1 (Nachtrag 19): Historien-Test nach zwei gefangenen
Eigene-Bugs (Schreib-Artefakt, bool-Aufruf) im dritten Versuch PASS.
'''


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


def preflight_gate(contract):
    '''K — KOLLAPS-GATE (Pre-Flight beim Feuern): verweigert den Start,
    wenn der Contract selbst schon unmoeglich ist. Getestet wird alles,
    was Daten fassen koennen: doppelte Arm-Keys, Praedikate auf
    unbekannte Arme, Praedikate auf unregistrierte Masse (die
    operationalisierte "scharf"-Klasse: Registrierung verlangt eine
    Messung, die kein Gate liefern kann).'''
    errors = validate_contract(contract)
    arm_names = {a['name'] for a in contract['arms']}
    measures = set(contract.get('measures', []))
    if not measures:
        errors.append('Contract ohne registrierte Masse — nichts messbar.')
    for rule in contract.get('sanity', []) + contract.get('verdict', []):
        for p in rule.get('predicate', []):
            m = p['metric']
            if '.' not in m:
                errors.append('Praedikat-Metrik %r ohne Arm-Trenner '
                              '(Format <arm>.<groesse>).' % m)
                continue
            arm, g = m.rsplit('.', 1)
            if arm not in arm_names:
                errors.append('Praedikat %r referenziert UNBEKANNTEN Arm %r.'
                              % (m, arm))
            if g not in measures:
                errors.append('Praedikat %r referenziert UNREGISTRIERTES '
                              'Mass %r — Registrierung verlangt eine '
                              'Messung, die kein Gate liefern kann '
                              '(Kollaps-Verletzung).' % (m, g))
    return errors


def audit_arm(arm, measurements):
    '''A — INVARIANTEN-AUDIT: Das Instrument MISST, es vertraut nicht.
    Messwerte kommen aus dem Lauf (oder — im Historien-Test — als
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
    }[pred['op']]


def evaluate(contract, measurements):
    '''Generator-Kern: Auswertung + Verdikt ausschliesslich aus dem
    Contract. K-Pre-Flight zuerst, dann Invarianten-Audit und
    Mess-Abdeckung (K-Laufzeit), Sanity-Kette, genagelte Verdikt-
    Reihenfolge. Jede Verletzung schlaegt Alarm — nichts laeuft still.'''
    errs = preflight_gate(contract)
    if errs:
        return dict(verdict='VERWEIGERT', alarms=errs, metrics={})
    metrics = {}
    alarms = []
    for a in contract['arms']:
        row = measurements.get(a['name'], {})
        if not row:
            alarms.append('ARM %s: KEINE Messwerte — Lauf unvollstaendig '
                          '(Kollaps-Verletzung).' % a['name'])
        alarms += audit_arm(a, row)
        for g in contract['measures']:
            if row.get(g) is None:
                alarms.append('ARM %s: registrierte Groesse %r NICHT '
                              'GEMESSEN (Kollaps-Verletzung).'
                              % (a['name'], g))
            else:
                metrics['%s.%s' % (a['name'], g)] = row[g]

    def rule_ok(rule):
        ok = True
        for p in rule['predicate']:
            try:
                if not _eval_pred(p, metrics):
                    ok = False
            except KeyError as e:
                alarms.append('PRAEDIKAT %r: %s' % (p['metric'], e))
                ok = False
        return ok

    sanity_ok = all(rule_ok(r) for r in contract.get('sanity', []))
    hit = None
    for rule in contract.get('verdict', []):
        if rule_ok(rule):
            hit = rule
            break
    if alarms:
        return dict(verdict='UNENTSCHEIDEN (Crash-Klasse) — Invarianten-'
                    'Audit/Kollaps-Gate: deklarierte Konvention verletzt',
                    alarms=alarms, metrics=metrics)
    if not sanity_ok:
        return dict(verdict='UNENTSCHEIDEN — SANITY GERICHT',
                    alarms=[], metrics=metrics)
    if hit is None:
        return dict(verdict='GEMISCHT — keine registrierte Konstellation '
                    'sitzt', alarms=[], metrics=metrics)
    return dict(verdict=hit['id'], text=hit['text'], alarms=[],
                metrics=metrics)


def determinismus_diff(current, previous):
    '''D — DETERMINISMUS-DIFF: Rerun gegen den letzten Result-Commit.
    Leer = deterministisch reproduziert. JEDE Abweichung wird geflaggt
    (Umwelt-Bug-Verdacht: Runner-Image, torch-Version) — geflaggt, nicht
    verwurstet. Macht die Determinismus-Annahme zur messbaren Groesse.'''
    diffs = []
    for arm in sorted(set(current) | set(previous)):
        c = current.get(arm) or {}
        p = previous.get(arm) or {}
        for g in sorted(set(c) | set(p)):
            cv, pv = c.get(g), p.get(g)
            if cv is None or pv is None or abs(cv - pv) > 1e-9:
                diffs.append('ARM %s, %s: Rerun %r vs Commit %r — '
                             'Abweichung ohne Crash = Umwelt-Bug-Verdacht.'
                             % (arm, g, cv, pv))
    return diffs


def auto_nachtrag(contract, result, nummer, datum):
    '''N — AUTO-NACHTRAG: Skelett aus Contract + Result. Zahlen und
    Verdikt maschinell; die Lesart (XVII-Urteil) bleibt handgeschrieben.
    LUECKENLISTE: registrierte Groessen, die im Result fehlen.'''
    L = []
    L.append('# Nachtrag %s (%s) — AUTO-SKELETT (N-Modul; Lesart '
             'handzuschreiben)' % (nummer, datum))
    L.append('')
    L.append('VERDICT: %s' % result['verdict'])
    for al in result.get('alarms', []):
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('## Arm-Zahlen (aus dem Contract generiert)')
    fehlen = []
    for a in contract['arms']:
        for g in contract['measures']:
            key = '%s.%s' % (a['name'], g)
            v = result['metrics'].get(key)
            if v is None:
                L.append('  %s = FEHLT' % key)
                fehlen.append(key)
            else:
                L.append('  %s = %s'
                         % (key, '%g' % v if isinstance(v, float) else v))
    L.append('')
    if fehlen:
        L.append('## LUECKENLISTE: %s' % ', '.join(fehlen))
    else:
        L.append('## LUECKENLISTE: KEINE — alle registrierten Groessen '
                 'gemessen.')
    return '\n'.join(L)
