# W1 — LAB-KERNEL: Der Experiment-Compiler (Werkzeug-Stein, registriert VOR dem Bau)
Registriert 2026-11-08 (P0, Design-Phase). Labor: cgfn, GitHub-Runner (CPU), ~40 EUR/Monat.
Freeze-Regel gilt auch fuer Werkzeuge: keine Nachverhandlung am Design nach Freigabe der Phase.

## 1. These des Werkzeugs (die eine Wette)
Der Engpass dieses Labors ist nicht Compute und nicht Ideen — es ist die
DISZIPLIN-LUECKE zwischen dem, was registriert wurde, und dem, was der Code
tatsaechlich misst. Zwei Laeufe (~52 Runner-Minuten, 100% des heutigen
Budgets) sind an genau dieser Luecke verbrannt:
  T11: Stille Normierung des REF-Sanity-Arms (Code wich von der bewaehrten
       Referenz ab, niemand mass es) — UNENTSCHEIDEN.
  T15: Readout las falschen Schluessel (Auswertung wich von der eigenen
       Registrierung ab, niemand pruefte es) — Crash-Klasse.
WETTE: Wenn Registrierung, Lauf, Auswertung und Verdikt aus EINER
deklarativen Quelle GENERIERT werden statt handgeschrieben nebeneinander,
stirbt diese Bug-Klasse strukturell. Nicht: Der Mensch diszipliniert
sich besser. Sondern: Die Verletzung wird UNMOEGLICH, weil es nichts gibt,
was abweichen kann.

## 2. Architektur (fuenf Module, drei Phasen)
  C — CONTRACT (Herzstueck): Registrierung als maschinenlesbare Struktur:
      arms (Name, H, Parameter-Konventionen als WERTE), invariants
      (messbare Assertions je Arm), sanity-Ketten, thesen/tore/verdict-
      Logik als DATEN (Prdikate ueber registrierte Groessen). Aus dem
      Contract GENERIERT der Kernel: (a) den Lauf-Code (Arm-Schleife,
      Konstruktion, Training), (b) die Auswertung, (c) den Verdikt-Text,
      (d) den Docstring-Header. Der Registrierungstext ist OUTPUT, nicht
      Eingabe. Es gibt keine handgeschriebene Auswertung mehr.
  A — INVARIANTEN-AUDIT: Jeder Arm MISST zur Laufzeit seine eigenen
      Konventionen und schreibt sie in die Result-Datei: Spaltennormen
      (exakt s? REF unnnormiert ~0.5*sqrt(H)?), DM je H, Energie je
      Kanal, Seed-Abdeckung, Divergenzen. Verletzung = harte Gate-Regel
      im Verdikt (Crash-Klasse), nicht Hoffnung. Das Instrument vertraut
      keiner Konvention, es misst sie. (Toetet die T11-Klasse.)
  D — DETERMINISMUS-DIFF: Reruns sind nur bei Crash erlaubt; wenn sie
      laufen, MUSS das Arm-fuer-Arm-Diff gegen den letzten Result-Commit
      leer sein. Abweichung ohne Crash = Umwelt-Bug (Runner-Image,
      torch-Version) — wird geflaggt, nicht verwurstet. (Macht unsere
      Determinismus-Annersion zur messbaren Groesse statt Glauben.)
  K — KOLLAPS-GATE (Pre-Flight beim Feuern): Der Kernel verweigert den
      Start, wenn (a) die Registrierung Begriffe enthaelt, die kein Gate
      prueft (z.B. "scharf" ohne Schaerfe-Mass — haette T15-PASS
      verhindert), (b) ein Arm-Key doppelt existiert (haette T15-Run-1
      verhindert), (c) ein Mass im Bericht auftaucht, das nicht
      registriert wurde, (d) eine registrierte Groesse fehlt.
  N — AUTO-NACHTRAG: Result + Contract -> Nachtrag-Skelett mit Zahlen,
      Verdikt und Lueckenliste (welche registrierte Groesse fehlt im
      Output). XVII-Urteile bleiben handgeschrieben; Zahlen nicht.

## 3. Phasen (genagelt)
  P0 (DIESER STEIN): Design registriert. Kein Code. Freigabe durch
      den Kapitaen noetig fuer P1.
  P1 (MINIMAL): C+A. lab_kernel.py: Contract-DSL (Dict/JSON), Generator
      fuer Lauf+Auswertung+Verdikt aus Contract, Invarianten-Audit im
      Lauf. KEINE weiteren Module. Abnahme ausschliesslich am
      Historien-Test (Abschnitt 4).
  P2 (VOLLAUSBAU): D, K, N. Erst nach P1-Abnahme.

## 4. Abnahme-Registrierung W1 (GENAGELT, KEINE NACHVERHANDLUNG)
  Der Kernel wird NICHT an neuen Experimenten gemessen, sondern an
  bekannter Historie (known-answer-test):
  W1-PASS (alle drei):
    (a) T15-WIEDERGABE: Ein T15-Contract generiert einen Lauf, dessen
        Arm-Zahlen und Verdikt (KILL-A) identisch sind mit Commit 4d9bd4b
        (Ergebnis-Datei von Run-2).
    (b) T11-ANZEIGE: Einspeisung der historischen T11-Registrierung
        (REF unnnormiert) zusammen mit dem historischen T11-Code (REF
        auf Norm 0.5 still normiert) MUSS die Abweichung als
        Invarianten-Verletzung anzeigen (Audit-Alarm), nicht still
        durchlaufen.
    (c) T15-RUN1-ANZEIGE: Die historische T15-Run-1-Struktur (REF-Arme
        teilen einen Key) MUSS vom Generator verweigert oder als
        Kollaps-Verletzung angezeigt werden.
  W1-KILL: Der Generator erreicht (a) nur mit handgeschriebenen
    Ausnahmen/Hardcodes im generierten Lauf (dann ist der Contract
    Schaum und das Werkzeug wertlos) — oder (b)/(c) sind prinzipiell
    nicht pruefbar (dann war die These falsch: die Bug-Klassen sind
    nicht maschinell toetbar).
  W1-UNENTSCHEIDEN: Generator kann T15 nicht reproduzieren
    (Design-Defekt). EINE Reparatur erlaubt, dann W1-Linie tot.
  SEKUNDAER: Lines of Code, Laufzeit-Overhead des Audits, Lesbarkeit
    des generierten Codes gegenueber dem Handcode.
  Bewahrung: Der Kernel wird nach Abnahme an der laufenden Serie NICHT
  nachjustiert; seine erste echte Anwendung ist der Stein nach T15.
  KOSTEN: 0 Runner-Minuten fuer P1-Abnahme (Offline-Test gegen
  Historie; ein optionaler 1-Arm-Smoke-Run zur集成-Pruefung erlaubt).

## 5. Einordnung (ehrlich)
Das Werkzeug ist kein AGI-Beitrag und kein neuer physics-Befund. Es ist
die Antwort auf die Frage, warum ein 40-EUR-Labor mit Big Labs
konkurrieren kann, ohne zu brute-forcen: Wir maximieren
INFORMATION PRO EURO. Zwei Bugs pro Tag sind bei uns 50% Budget —
null Bugs pro Tag verdoppelt die Geschwindigkeit der ganzen Serie.
Praezedenz: Praeregistrierung existiert in der Wissenschaft als PDF-
Zeremonie (OSF), die niemanden zwingt; ein ausfuehrbarer Contract-
Compiler, der Verdikte generiert und eigene Bugs anzeigt, ist nach
unserem Wissen nicht gebaut worden (Recherche vor P1 zu wiederholen,
nicht nur behauptet).
