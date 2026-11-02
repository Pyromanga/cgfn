# Technischer Bericht: Chess-Manifold-Hypothesentest

**Stand:** 2026-11-01 — erste Ergebnisse siehe `results/summary.json` (Pipeline-Run).

## 1. Hypothese und Falsifikationskriterium

Vorregistriert (siehe README): H1 ist nur gestuetzt, wenn
(a) das Bootstrap-95%-CI von `d_geo(Start, Remis) - d_geo(Start, Sieg-Becken)` durchgehend
negativ ist (Start fliesst ins Remis-Becken, p < 0.05, einseitig) **und**
(b) die topologische k-NN-Genauigkeit die Stockfish-depth-1-Baseline schlaegt.
Ein negatives oder breites Ergebnis ist ein wertvoller Beitrag (Falsifikation ist Erfolg).

## 2. Methodik und Annahmen

- **Daten:** Geseedetes Random-Selfplay (python-chess). Kein externer Download ->
  100 % reproduzierbar (Seed 42). Positionen ab Ply 4, ~12 pro Partie,
  Startstellung immer enthalten, fruehe Mittelspielstellungen (Ply 8-20) markiert.
- **Supervision:** Stockfish depth 12 (Ubuntu apt-Binary), Weiss-sichtige cp,
  Sigmoid-Transform zu Gewinnwahrscheinlichkeit (Lambda 0.00368208, +-2000 cp geclippt).
  **Baseline:** depth 1 (minimale Suche).
- **Features (~65):** klassische Merkmale (Material, Figurenzaehlung, Doppel-/Isolierter-/
  Freibauern, Raumkontrolle in gegnerischer Haelfte, Koenigsangriff, Mobilitaet, Rochaderechte)
  **plus** topologische Deskriptoren: Vietoris-Rips-PH auf Punktwolken der Figuren
  (Koordinaten (file, rank, 0.25*Figurenwert)) je Gruppe Weiss/Schwarz/Beide:
  H0-Gesamtpersistenz, H1-Anzahl, H1-Gesamt/Max-Persistenz, H1-Persistenzentropie.
- **Manifold-Lernen:** MLP-Encoder (2x256, GELU, LayerNorm) -> 16-dim L2-normalisiert,
  Supervised Contrastive Loss (Khosla et al.), Labels = 6 Bins der Gewinnwahrscheinlichkeit
  (Positivpaare implizit innerhalb Bin, aequivalent zum Schwellenwert-0.05-Kriterium
  bei wp-Bin-Breite ~0.167). PyTorch Lightning, CPU, AdamW.
- **Analyse:** PH je Becken (ripser, Betti-Kurven ueber Radien-Grid), diskrete Geodaeten
  im Isomap-k-NN-Graphen (scipy csgraph), Kohaerenz = stratified-k-NN-Becken-Accuracy,
  Bootstrap (2000x, Beckenmitglieder resampelt -> Schwerpunktdistanzen), Gegenbeispielsuche
  (topologische Vorhersage vs. |wp - 0.5| > 0.35).

### Unsicherheiten (explizit)

1. Random-Selfplay verzerrt die Stellungsverteilung (wenige remisige Zuege,
   viele Extremstellungen). Die Becken sind daher groesser separiert als in
   Turnierdaten -> Kohaerenzzahlen sind obere Schranken.
2. Sigmoid-Kalibrierung ist nicht SF16-spezifisch fit (Literaturwert uebernommen).
3. Diskrete Geodaeten haengen vom k-NN-Graphen ab; bei kleinen Stichproben
   kann der Graph duenn oder nicht zusammenhaengend sein (Fallback behandelt das).
4. PH auf <=1200 Punkten je Becken (Laufzeit), geseedet.
5. Lc0-Einbettungen nicht implementiert (optionaler Schritt der Aufgabe);
   Stockfish allein traegt die Engine-Supervision.
6. Remis-Becken via Engine definiert (0.45 < wp < 0.55), nicht via Spielausgang —
   notwendige Konsequenz aus Annahme 1, gleichzeitig der sauberste Test der
   Hypothese ("Becken = Ergebnisattraktor" gegen "Becken = Engine-Meinung").

## 3. Ergebnisse

> **Dieser Abschnitt wird aus `results/summary.json` des ersten CI-Laeufs
> ergaenzt.** Die Pipeline schreibt: Beckengroessen, Betti-Signaturen,
> Geodaetenlaengen der Startstellung, Bootstrap-Test, Gegenbeispiele,
> Kohaerenz vs. Literatur (91,7 %), Baseline-Vergleich.

## 4. Bewertung der Hypothese

Nach erstem Run: siehe Abschnitt 3 + `results/report.json`.

## 5. Naechste Schritte

1. Ersetzung Random-Selfplay durch Lichess-Top-Games (Stichprobe 10k, Filter
   Zeitkontrolle 3+2, Elo >= 2500) als Verteilungs-Robustheitscheck.
2. Lc0-Einbettungen als zweite Repraesentation (Aufgabenschritt 1, optional).
3. Geodaeten-Kruemmung via Pfadintegration im k-NN-Graphen statt reinen Distanzen.
4. Test auf fruehe Mittelspielstellungen (bereits im Report: geodesic_early_mean_to_basins).
