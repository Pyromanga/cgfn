#!/usr/bin/env python
'''W1 P2 — HISTORIEN-TEST-RUNNER. Abnahme genagelt in XVII Nachtrag 20
und analysis/W1_lab_kernel.md Abschnitt 4: Regression a/b/c (P1) plus
P2-Module D (Determinismus-Diff), K (Kollaps-Gate: Pre-Flight und
Laufzeit), N (Auto-Nachtrag-Skelett). Alles known-answer gegen die
eigene Serie — kein neues Experiment, 0 Runner-Minuten netto.'''
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import lab_kernel
from contracts.t15 import contract_t15
from contracts.t11 import contract_t11
from tests.fixtures import (
    measurements_t15_run2,
    measurements_t11_historisch,
    contract_t15_run1_bug,
    contract_unregistriertes_mass,
)


def run_historien_test():
    L = []
    L.append('=' * 78)
    L.append('W1 P2 — HISTORIEN-TEST (Regression a/b/c + Module D/K/N)')
    L.append('Known-answer gegen die eigene Serie. Kein neues Experiment.')
    L.append('Architektur-Split (Nachtrag 19/20): Engine tools/, Contracts')
    L.append('contracts/, Fixtures + Runner tests/ — Schuld eingeloest.')
    L.append('=' * 78)
    ok = {}

    res = lab_kernel.evaluate(contract_t15(), measurements_t15_run2())
    L.append('')
    L.append('TEST (a) T15-WIEDERGABE (Regression P1):')
    L.append('  erwartet: KILL-A — ABSOLUTE KANTE | Zahlen wie Commit 4d9bd4b')
    L.append('  bekommen: %s' % res['verdict'])
    for k in sorted(res['metrics']):
        L.append('    %s = %.2f' % (k, res['metrics'][k]))
    ok['a'] = (res['verdict'].startswith('KILL-A')
               and abs(res['metrics']['REF24.pf'] - 0.80) < 1e-12
               and abs(res['metrics']['R96_3.5.pf'] - 0.60) < 1e-12
               and not res['alarms'])
    L.append('  (a) %s' % ('PASS' if ok['a'] else 'FAIL'))

    res_b = lab_kernel.evaluate(contract_t11(), measurements_t11_historisch())
    L.append('')
    L.append('TEST (b) T11-ANZEIGE (Regression P1; stille Normierung -> Alarm):')
    for al in res_b['alarms']:
        L.append('  ALARM: %s' % al)
    ok['b'] = bool(res_b['alarms']) and res_b['verdict'].startswith('UNENTSCHEIDEN')
    L.append('  (b) %s' % ('PASS' if ok['b'] else 'FAIL'))

    res_c = lab_kernel.evaluate(contract_t15_run1_bug(),
                                measurements_t15_run2())
    L.append('')
    L.append('TEST (c) T15-RUN1-DOPPELKEY (Regression P1; Generator verweigert):')
    L.append('  bekommen: %s' % res_c['verdict'])
    for al in res_c['alarms']:
        L.append('  ALARM: %s' % al)
    ok['c'] = (res_c['verdict'] == 'VERWEIGERT'
               and any('DOPPELTE' in a for a in res_c['alarms']))
    L.append('  (c) %s' % ('PASS' if ok['c'] else 'FAIL'))

    m2 = measurements_t15_run2()
    diffs_leer = lab_kernel.determinismus_diff(m2, measurements_t15_run2())
    mut = measurements_t15_run2()
    mut['REF24'] = dict(mut['REF24'])
    mut['REF24']['pf'] = 0.70
    diffs_mut = lab_kernel.determinismus_diff(mut, measurements_t15_run2())
    L.append('')
    L.append('TEST (d) D-DETERMINISMUS-DIFF:')
    L.append('  identisch -> %d Flags (erwartet 0)' % len(diffs_leer))
    L.append('  mutiert (REF24.pf 0.70) -> %d Flag(s): %s'
             % (len(diffs_mut), diffs_mut[0] if diffs_mut else 'KEIN'))
    ok['d'] = (not diffs_leer and len(diffs_mut) == 1
               and 'REF24' in diffs_mut[0] and 'pf' in diffs_mut[0])
    L.append('  (d) %s' % ('PASS' if ok['d'] else 'FAIL'))

    res_k1 = lab_kernel.evaluate(contract_unregistriertes_mass(), m2)
    L.append('')
    L.append('TEST (k1) K-PRE-FLIGHT (Praedikat auf unregistriertes Mass):')
    L.append('  bekommen: %s' % res_k1['verdict'])
    for al in res_k1['alarms']:
        L.append('  ALARM: %s' % al)
    ok['k1'] = (res_k1['verdict'] == 'VERWEIGERT'
                and any('UNREGISTRIERTES' in a for a in res_k1['alarms']))
    L.append('  (k1) %s' % ('PASS' if ok['k1'] else 'FAIL'))

    m_kaputt = {k: dict(v) for k, v in m2.items() if k != 'R96_4.9'}
    res_k2 = lab_kernel.evaluate(contract_t15(), m_kaputt)
    L.append('')
    L.append('TEST (k2) K-LAUFZEIT (fehlende Messung -> Alarm, kein Loch):')
    L.append('  bekommen: %s' % res_k2['verdict'])
    for al in res_k2['alarms']:
        L.append('  ALARM: %s' % al)
    ok['k2'] = (res_k2['verdict'].startswith('UNENTSCHEIDEN')
                and any('NICHT GEMESSEN' in a for a in res_k2['alarms']))
    L.append('  (k2) %s' % ('PASS' if ok['k2'] else 'FAIL'))

    res_n = lab_kernel.evaluate(contract_t15(), m2)
    skelett = lab_kernel.auto_nachtrag(contract_t15(), res_n, 'XX',
                                       '2026-11-08')
    skelett_kaputt = lab_kernel.auto_nachtrag(contract_t15(), res_k2, 'XX',
                                              '2026-11-08')
    L.append('')
    L.append('TEST (n) N-AUTO-NACHTRAG:')
    L.append('  Skelett (vollstaendig): %d Zeilen, LUECKENLISTE: %s'
             % (len(skelett.split('\n')),
                'KEINE' if 'LUECKENLISTE: KEINE' in skelett else 'FEHLER'))
    L.append('  Skelett (R96_4.9 fehlt): LUECKENLISTE nennt R96_4.9.pf: %s'
             % ('JA' if 'R96_4.9.pf' in skelett_kaputt.split('LUECKENLISTE')[1]
                else 'NEIN'))
    ok['n'] = ('KILL-A' in skelett
               and 'R96_3.5.pf = 0.6' in skelett
               and 'LUECKENLISTE: KEINE' in skelett
               and 'R96_4.9.pf' in skelett_kaputt.split('LUECKENLISTE')[1])
    L.append('  (n) %s' % ('PASS' if ok['n'] else 'FAIL'))

    L.append('')
    order = ['a', 'b', 'c', 'd', 'k1', 'k2', 'n']
    if all(ok[p] for p in order):
        verdict = ('W1-P2-PASS — Regression a/b/c bestanden; D-Diff leer/'
                   'flaggt korrekt, K-Gate verweigert/alarmt, N-Skelett '
                   'komplett/LUECKENLISTE korrekt. P2 ANGENOMMEN.')
    else:
        teile = [p for p in order if not ok[p]]
        verdict = ('W1-P2-UNENTSCHEIDEN — nicht bestanden (%s). '
                   'EINE Reparatur erlaubt, dann P2-Linie tot.'
                   % '/'.join(teile))
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    return '\n'.join(L)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    out = run_historien_test()
    print(out)
    with open('results/w1_kernel_p2_test.txt', 'w', encoding='utf-8') as fh:
        fh.write(out + '\n')
