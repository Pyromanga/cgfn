
---

# Nachtrag 20 (2026-11-08): W1-P2-REGISTRIERUNG — Vollausbau D/K/N, Architektur-Schuld einzuloesen VOR jeder Kernel-Anwendung (genagelt VOR dem Lauf)

## P2-Umfang (registriert, KEINE NACHVERHANDLUNG)
1. ARCHITEKTUR-SPLIT (die Schuld aus Nachtrag 19): Engine ->
   tools/lab_kernel.py; Contracts -> contracts/ (t15.py, t11.py, reine
   Daten); Historien-Fixtures + Runner -> tests/ (fixtures.py,
   run_historien_test.py). Die P1-Regression a/b/c bleibt Teil der
   Abnahme — jetzt gegen die getrennten Bausteine.
2. D — DETERMINISMUS-DIFF: determinismus_diff(rerun, commit). Leer bei
   identischem Rerun; JEDE Abweichung ein Flag (Umwelt-Bug-Verdacht:
   Runner-Image, torch-Version). Geflaggt wird, nicht verwurstet.
3. K — KOLLAPS-GATE, zwei Ebenen:
   preflight_gate(contract) verweigert den Start bei (i) doppelten
   Arm-Keys (T15-Run1-Klasse), (ii) Praedikaten auf unbekannte Arme,
   (iii) Praedikaten auf UNREGISTRIERTE Masse — die operationalisierte
   T15-"scharf"-Klasse: Eine Registrierung, die eine Messung verlangt,
   die kein Gate liefern kann, startet nicht.
   Laufzeit: evaluate schlaegt NICHT-GEMESSEN-Alarm, wenn eine
   registrierte Groesse im Mess-Input fehlt — kein stilles Loch.
   EHRLICHE GRENZE (registriert): Freitext-Begriffe ohne messbare
   Operationalisierung prueft keine Maschine. K prueft die
   datenfassbare Variante; der Rest bleibt Menschenpflicht.
4. N — AUTO-NACHTRAG: auto_nachtrag(contract, result) — Zahlen-Skelett
   mit Verdikt und LUECKENLISTE (welche registrierte Groesse im Output
   fehlt). XVII-Urteile (Lesart, Selbsteinordnung) bleiben
   handgeschrieben; die Zahlen nicht.

## P2-Abnahme (genagelt, alles Historien-/Known-Answer-Test, 0 neue Physik)
  P2-PASS: Regression (a) T15-Wiedergabe KILL-A mit identischen Zahlen,
  (b) T11-stille-Normierung -> Audit-Alarm, (c) T15-Run1-Doppelkey ->
  VERWEIGERT — UND neu:
  (d) D-DIFF: diff(Run2, Run2) LEER; diff(Run2, Run2-mutiert) flaggt
      GENAU die mutierte Arm-Groesse (REF24.pf).
  (k1) K-PRE-FLIGHT: Contract mit Praedikat auf unregistriertes Mass
      ("schaerfe") -> VERWEIGERT mit UNREGISTRIERT-Alarm.
  (k2) K-LAUFZEIT: Mess-Input ohne einen Arm -> NICHT-GEMESSEN-Alarm,
      Verdikt UNENTSCHEIDEN (Crash-Klasse), kein stiller Durchlauf.
  (n) N-SKELETT: T15-Result -> Skelett mit KILL-A, allen acht Arm-Zahlen
      (z.B. R96_3.5.pf = 0.6) und LUECKENLISTE: KEINE; unvollstaendiger
      Result -> LUECKENLISTE nennt die fehlende Groesse (R96_4.9.pf).
  P2-KILL: Ein Modul nur mit Hardcodes/test-spezifischen Sonderfaellen
      erreichbar — dann ist der Contract-Schaum-Vorwurf zurueck.
  P2-UNENTSCHEIDEN: Ein Test faellt; EINE Reparatur erlaubt, dann
      P2-Linie tot (wie P1).
  Bewahrung: kein Rerun ohne Crash; Kostenziel 0 Runner-Minuten netto
  (reines Python, Sekunden, Kontingent unberuehrt).
