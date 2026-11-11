#!/usr/bin/env python
'''T21-REPARATUR (registriert XVII Nachtrag 35, VOR dem Lauf).
KEINE neue Physik: Das Experiment ist deterministisch; ein Rerun
wuerde bit-identische Zahlen liefern. Diese Reparatur laedt die
GEFRORENEN Run-2-Messungen (results/t21_measurements.json), setzt
den Readout-Fix (Anchor-Einpunkt-Kurve -> s50rep = -1 ZENSIERT,
exakt die sgrid-Regel aus Nachtrag 33) und ruft lab_kernel.evaluate
auf den unveraenderten Daten. Schwellen UNVERAENDERT (±0.15).
Ausgang: formales Verdikt aus Run-2-Daten, Lesart als Nachtrag 36.'''
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import lab_kernel
from contracts.t21 import contract_t21

C = contract_t21()

MS_PATH = 'results/t21_measurements.json'
OUT_PATH = 'results/t21_repair_verdict.txt'

raw = json.loads(Path(MS_PATH).read_text(encoding='utf-8'))
measurements = raw['measurements']

L = []
L.append('=' * 118)
L.append('T21-REPARATUR: NEU-AUSWERTUNG DER GEFRORENEN RUN-2-DATEN')
L.append('Readout-Fix (Nachtrag 35): Anchor-Einpunkt-Kurve -> s50rep = -1 ZENSIERT.')
L.append('KEINE neue Physik, Schwellen unveraendert (±0.15). Daten: Bot-Commit be35ddf.')
L.append('=' * 118)

# Integritaets-Proben gegen die Run-2-Auswertung
d183_frozen = raw['d183']
if measurements['H48_s1.83']['d183'] is None:
    L.append('ALARM: gefrorene d183 fehlt — Abbruch ohne Verdikt.')
    Path(OUT_PATH).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    sys.exit(1)
L.append('Gefroren: d183 = %.4f | H24-Kurve 0.49/0.67/0.78 | '
         'H48-Kurve 0.13/0.29/0.43' % measurements['H48_s1.83']['d183'])
if abs(measurements['H48_s1.83']['d183'] - d183_frozen) > 1e-12:
    L.append('ALARM: d183-Inkonsistenz zwischen Toplevel und Arm-Daten.')
    Path(OUT_PATH).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    sys.exit(1)

# READOUT-FIX: Anchor-Einpunkt-Kurve ist nicht interpolierbar -> ZENSIERT
if 's50rep' not in measurements['HIGH48'] or measurements['HIGH48']['s50rep'] is None:
    measurements['HIGH48']['s50rep'] = -1.0
    L.append('Readout-Fix: HIGH48.s50rep = -1.0 (ZENSIERT, Einpunkt-Kurve).')

res = lab_kernel.evaluate(C, measurements)
for al in res['alarms']:
    L.append('ALARM: %s' % al)
L.append('')
L.append('VERDICT (aus gefrorenen Run-2-Daten, Readout repariert): %s' % res['verdict'])
skelett = lab_kernel.auto_nachtrag(C, res, '36', '2026-11-11')
L.append('')
L.append('--- N-SKELETT (N-Modul, Zahlen maschinell; Lesart handzuschreiben) ---')
L.append(skelett)
L.append('=== FERTIG ===')
out = '\n'.join(L) + '\n'
Path(OUT_PATH).write_text(out, encoding='utf-8')
print(out)


if __name__ == '__main__':
    pass
