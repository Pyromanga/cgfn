#!/usr/bin/env python
'''T23 READOUT-REPARATUR (R5: die EINE erlaubte Reparatur, registriert
XVII Nachtrag 41, Pfad (b), Nutzer-Entscheidung).
Laedt results/t23_measurements.json (Run 1, Bot-Commit 184c7c1,
GEFROREN — kein Rerun, keine Seeds, kein Torch, deterministisch).
Eingriff 1 (READOUT, T21-Praezedenzklasse): HIGH48.pstage = HIGH48.pf
(Ein-Arm-Stufenmittel des Anchor, Definition in Nachtrag 41).
Eingriff 2 (DOKUMENTATION, keine Daten-Aenderung): std-Klausel
zurueckgefuehrt auf Ursprungszweck (Nenner-Schutz, T22 zalt);
verdikts-entscheidende Stufe a=1 hat std=0.025 >= 0.02 (clean);
a=24 (std=0.0, Boden-Stabilitaet 0.15/0.15) als Form-Befund.
Verdikt-Mass drop1 und Schwelle 0.30 UNBERUEHRT.'''
import json
import sys
import os
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import lab_kernel
from contracts.t23 import contract_t23

C = contract_t23()
MS_PATH = 'results/t23_measurements.json'
OUT_PATH = 'results/t23_readout_repair_results.txt'


def main():
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('T23 READOUT-REPARATUR (R5, die EINE erlaubte; Nachtrag 41, Pfad (b))')
    L.append('Daten: %s (Run 1, 184c7c1, GEFROREN — kein Rerun, kein Torch)' % MS_PATH)
    L.append('Eingriff 1: HIGH48.pstage = HIGH48.pf (Ein-Arm-Stufenmittel, n41-Definition)')
    L.append('Eingriff 2: std-Klausel-Dokumentation (Nenner-Schutz-Ursprung, T22;')
    L.append('  a=1 std=0.025 >= 0.02 clean; a=24 std=0.0 Boden-Stabilitaet, Form-Befund)')
    L.append('Verdikt-Mass drop1 + Schwelle 0.30 UNBERUEHRT. Deterministisch.')
    L.append('=' * 118)

    raw = json.loads(Path(MS_PATH).read_text(encoding='utf-8'))
    ms = raw['measurements']
    stage_std = raw.get('stage_std', {})
    L.append('')
    L.append('KONTROLLE: gefrorene Werte unveraendert: drop1=%s | drop124=%s'
             % (raw.get('drop1'), raw.get('drop124')))
    L.append('std(P_Block) je Stufe: %s' % stage_std)
    if raw.get('drop1') is None:
        L.append('ALARM: drop1 im gefrorenen Datensatz fehlend — Abbruch.')
        Path(OUT_PATH).write_text('\n'.join(L) + '\n', encoding='utf-8')
        sys.exit(1)

    # Eingriff 1: Readout-Reparatur (nur fehlendes Anchor-Mass setzen)
    fixed = 0
    for name in ms:
        if name == 'HIGH48' and ms[name].get('pstage') is None:
            ms[name]['pstage'] = ms[name]['pf']
            fixed += 1
            L.append('READOUT-FIX: HIGH48.pstage = %s (war NICHT GEMESSEN)'
                     % ms[name]['pstage'])
    L.append('Fehlende Masse gesetzt: %d (nur HIGH48.pstage; kein Messwert'
             ' veraendert)' % fixed)

    res = lab_kernel.evaluate(C, ms)
    for al in res['alarms']:
        L.append('ALARM: %s' % al)
    L.append('')
    L.append('VERDICT: %s' % res['verdict'])
    L.append('')
    L.append('KLAUSEL-PROTOKOLL (Nachtrag 41):')
    L.append('  - std(a=1) = %s >= 0.02: verdikts-entscheidende Stufe CLEAN'
             % stage_std.get('1'))
    L.append('  - std(a=24) = %s < 0.02: Boden-Stabilitaet (0.15/0.15);'
             ' Form-Befund, kein Nenner im Primaer-Mass' % stage_std.get('24'))
    L.append('  - Ursprung der Klausel: T22 zalt-Nenner-Schutz (Nachtrag 38)')
    skelett = lab_kernel.auto_nachtrag(C, res, '42', '2026-11-11')
    L.append('')
    L.append('--- N-SKELETT (N-Modul; Lesart handzuschreiben) ---')
    L.append(skelett)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(OUT_PATH).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f s (deterministisch, kein Rerun)' % (time.time() - t0))


if __name__ == '__main__':
    main()
