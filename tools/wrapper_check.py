#!/usr/bin/env python
'''MSL-Wrapper-Guard (2026-11-09, geboren aus dem T19-R2-Zwischenfall).

Hintergrund: get_file_contents liefert einen Statuskopf der Form
"successfully downloaded text file (SHA: ...)" VOR dem eigentlichen
Dateiinhalt. Wer diese Ausgabe unverarbeitet weitercommittet, backt
den Wrapper in Zeile 1 und toet jeden Syntax-Gate mit einer
kryptischen Fehlermeldung. Passiert: Commit ae3efbf (make_Uw-Fix),
zwei tote Laeufe, Diagnose-Lauf noetig.

Dieser Guard erkennt den Marker am Dateikopf VOR py_compile und
sagt es in Klartext. Aufgerufen aus dem Workflow-Gate JEDES
kernel-generierten Steins (1 Runner-Sekunde, 0 Risiko).'''
import sys
from pathlib import Path

MARKERS = [
    'successfully downloaded text file',
    'Failed to get file contents',
]

targets = sys.argv[1:]
if not targets:
    targets = [str(p) for p in sorted(Path('.').rglob('*.py'))
               if '.git' not in p.parts and 'venv' not in p.parts]
bad = []
for t in targets:
    p = Path(t)
    if not p.exists():
        continue
    head = p.read_bytes()[:400].decode('utf-8', 'replace')
    for m in MARKERS:
        if m in head:
            bad.append('%s: MSL-Fetch-Wrapper ("%s...") am Dateikopf' % (t, m[:30]))
if bad:
    print('[e] MSL-WRAPPER-GUARD: get_file_contents-Ausgabe wurde unverarbeitet committet:')
    for b in bad:
        print('    ' + b)
    print('[e] Fix: alles vor dem Shebang der Quelldatei strippen, neu committen.')
    sys.exit(1)
print('[i] wrapper-guard ok (%d Dateien geprueft)' % len(targets))
