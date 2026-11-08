

---

# Nachtrag 21 (2026-11-08): W1-P2 — PASS (im ERSTEN Versuch). Vollausbau D/K/N abgenommen, Architektur-Schuld eingeloest.

## W1-P2 (Lauf 22:44-22:45 UTC, Bot-Commit 7af90d2, reines Python, Sekunden)
VERDICT: W1-P2-PASS — alle sieben Tests bestanden:
  REGRESSION (P1, gegen die getrennten Bausteine):
  (a) T15-Wiedergabe: KILL-A — ABSOLUTE KANTE, alle acht Arm-Zahlen
      identisch mit Commit 4d9bd4b.
  (b) T11-stille-Normierung: INVARIANTEN-VERLETZUNG-Alarm
      (gemessen 0.5, deklariert ~3.46).
  (c) T15-Run1-Doppelkey: VERWEIGERT (plus zwei zusaetzliche
      UNBEKANNTER-ARM-Alarme durch das schaerfere K-Gate).
  NEU (P2):
  (d) D-DIFF: identisch -> 0 Flags; mutiert -> genau das geflaggte
      REF24.pf (Rerun 0.7 vs Commit 0.8).
  (k1) K-PRE-FLIGHT: Praedikat auf unregistriertes Mass ("schaerfe")
      -> VERWEIGERT (die operationalisierte T15-"scharf"-Klasse —
      genau dieser Bug-Typ machte T15-PASS unmoeglich).
  (k2) K-LAUFZEIT: fehlender Arm -> fuenf Alarme, Verdikt
      UNENTSCHEIDEN (Crash-Klasse), kein stilles Loch.
  (n) N-SKELETT: KILL-A + alle acht Arm-Zahlen + LUECKENLISTE: KEINE;
      bei fehlendem Arm nennt die LUECKENLISTE die fehlende Groesse.

## Protokoll (ehrlich)
Erster Versuch, null Reparaturen, null verbrannte Runner-Minuten.
Der Vor-Push-Selbst-Audit (Umlaut-Probe + \n-Literal-Probe aus der
W1-Run-1-Lehre) lief als harter Gate vor dem Push — und fing beim
Nachtrag-21-Push zweimal einen Umlaut im eigenen Protokoll-Text,
bevor er das Repo erreichte (Null-Kosten, Korrektur im Sandbox-
Auditorium). Die Lehre aus dem eigenen Bug-Protokoll zahlt sich aus.
Die Architektur-Schuld aus Nachtrag 19 ist EINGELOEST, bevor der
Kernel erstmals scharf schiesst: Engine (tools/lab_kernel.py),
Contracts (contracts/, reine Daten), Fixtures + Runner (tests/).

## Stand der W-Linie
W1 ist damit VOLLSTAENDIG (P0 Design, P1 C+A, P2 D/K/N + Split).
Alle fuenf Module des registrierten Designs sind gebaut und am
Historien-Test abgenommen. Der Kernel ist SCHARF: Die naechste
registrierte These (z.B. T16, 2D-Karte H x s) laeuft erstmals durch
den Contract-Generator — Verdikt generiert, Audit misst, K-Gate
verweigert vor dem Start, D-Diff gegen den letzten Result-Commit,
N schreibt das Zahlen-Skelett. Danach P3 (XVII als abfragbare
Wissensstruktur) und P4 (These-Generator).
