# Technischer Bericht: Chess-Manifold-Hypothesentest

**Stand:** 2026-11-02 — Smoke-Run #6 (CI, gruen, 301 Positionen, Seed 42, 100 % reproduzierbar). Alle Zahlen: `results/report.json` (`results/summary.json`, PH-Diagramme als Artefakt `ph_diagrams`).

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
7. **Neu (post-hoc, wichtig):** Die depth-1-Baseline erreicht 100 % Becken-Accuracy.
   Das heisst: In dieser Datenverteilung ist die Becken-Klassifikation fuer eine
   minimale Suche trivial loesbar — Kriterium (b) ist in diesem Sample kaum
   ueberhaupt falsifizierbar. Das ist selbst ein Befund (siehe 4).

## 3. Ergebnisse (Smoke-Run #6, n = 301 Positionen)

**Beckengroessen** (Engine-wp-Thresholds 0.55/0.45): Weiss 211, Remis 37, Schwarz 53.
Die starke Weiss-Dominanz spiegelt das Random-Selfplay wider (Weiss am Zug in der
Stichprobe + depth-12-Eval); sie ist dokumentierte Verzerrung, kein Schwaechen-Beleg.

**Topologische Signaturen (PH auf dem 16-d-Einbettungsraum, ripser):**

| Becken | H0-Persistenz (total) | H1-Loecher | H1-Persistenz (total / max) | Betti0 @ r=0.25 |
|---|---|---|---|---|
| Weiss  | 35.7  | 129 | 1.78 / 0.049 | 11 |
| Remis  | 5.5   | 12  | 0.16 / 0.047 | 4 |
| Schwarz| 11.1  | 17  | 0.39 / 0.063 | 17 |

Alle drei Becken sind bei Radius >= 0.5 topologisch "einfach" (Betti0=1, Betti1=0),
unterscheiden sich aber in der Feinstruktur: Das Remis-Becken hat die **geringste**
H1-Gesamtpersistenz (0.16 vs. 1.78 Weiss) — es ist das kompakteste, "glatteste" Becken;
Schwarz ist am fragmentiertesten (Betti0=17 bei r=0.25). Das ist vereinbar mit der
Interpretation der Remis-Region als schmalen, kohaesiven Attraktor.

**Geodaeten von der Startstellung (diskret, Isomap-k-NN-Graph, k=10):**

- d(Start, Remis-Schwerpunkt) = **0.413**
- d(Start, Weiss-Schwerpunkt) = 0.991
- d(Start, Schwarz-Schwerpunkt) = 0.965

Fruehe Mittelspielstellungen (Ply 8-20) bestaetigen das Muster (Remis 0.60 < Weiss 0.78 < Schwarz 0.89).

**Bootstrap-Test (2000 Resamples, einseitig):**

- `d(Start,Remis) - d(Start,Weiss)`: mean = **-0.577**, 95%-CI **[-0.677, -0.471]**, p = 0.000
- `d(Start,Remis) - d(Start,Schwarz)`: mean = **-0.551**, 95%-CI **[-0.707, -0.398]**, p = 0.000

**Kriterium (a) ERFUELLT:** Beide CIs sind durchgehend negativ, p < 0.05 — die Startstellung
liegt im gelernten Manifold signifikant naeher am Remis-Becken als an beiden Sieg-Becken.

**Kohaerenzmetrik:** k-NN-Becken-Accuracy (stratified 5-fold) = **80.7 %** — unter dem
Literaturwert 91.7 % (Endspiel-Klassifikation), was den Ansatz-transfer vom Endspiel zum
fruehen Spiel als *schwieriger* ausweist, aber nicht als unmoeglich.

**Vorhersagekraft vs. Baseline (Kriterium b):**

- Topologische k-NN-Genauigkeit: **80.7 %**
- Stockfish-depth-1-Baseline: **100.0 %** (all und masked)

**Kriterium (b) NICHT ERFUELLT** — klar. Die depth-1-Baseline klassifiziert die
depth-12-Beckenlabels perfekt; die topologische Methode bleibt deutlich dahinter.

**Gegenbeispiele:** 72 Widersprueche im auswertbaren Subsample. Prominente Faelle:
`rnqk3r/pp5p/3n4/2pP2p1/1P2p3/3BP1pN/2b2P1P/1R1QK2R w K - 0 17` (topo: "Weiss", Engine:
wp = 0.21 -> Schwarz klar am Gewinnen) und `r3kb2/pbp1p2p/3q2rn/1p1P1pp1/1P1n2PP/P3P3/R1QK1P1N/1NB2B1R w q - 0 19`
(topo: "Weiss", wp = 0.28). Beispiele in `results/report.json` (contradiction_examples).

## 4. Bewertung der Hypothese

**Nach der vorregistrierten Entscheidungsregel: NICHT GESTUETZT (H1 verworfen).**

- Kriterium (a) — topologische Naehe der Startstellung zum Remis-Becken — ist **stark
  erfuellt** (p < 0.001, robuste CIs, konsistent auch im fruehen Mittelspiel). Das in der
  Hypothese beschriebene "Flieen ins Remis-Becken" ist in der gelernten Geometrie sichtbar.
- Kriterium (b) — die topologische Methode schlaegt die triviale Baseline — ist **klar
  gescheitert** (80.7 % vs. 100 %). Ohne Mehrwert gegenueber einer Ein-Ply-Suche hat das
  Verfahren keinen praediktiven Eigenwert ueber die Engine-Information hinaus.

**Einordnung und Caveats (ehrlich):**

1. Die Baseline ist *definitionsgemae ss* stark beguenstigt: Die Beckenlabels stammen aus
   depth-12-Evals desselben Engines, dessen depth-1-Ansicht stark korreliert. In einem
   Random-Selfplay-Sample mit vielen Extremstellungen ist 100 % erreichbar. Kriterium (b)
   ist damit in dieser Datenverteilung kaum zu gewinnen — der Test war unfair, aber
   vorregistriert und damit bindend.
2. Trotzdem bleibt der Kernbefund bemerkenswert: Ein reiner Feature-Encoder OHNE jegliche
   Suchinformation erreicht 80.7 % Beckenkohae renz und ordnet die Startstellung
   signifikant dem Remis-Becken zu — die Drei-Becken-Geometrie ist im latenten Raum
   nachweisbar vorhanden (unterschiedliche Betti-/Persistenz-Signaturen je Becken).
3. Smoke-n = 301: alle Zahlen haben breite Unsicherheit; der Full-Run (10k Positionen,
   Profil `full`) ist der naechste Schritt.

**Fazit:** Hypothese **widerlegt im praediktiven Anspruch (b), gestuetzt im strukturellen
Teil (a)**. Naechste Experimente muessen die Datenverteilung aendern (Turnierdaten statt
Random-Selfplay), damit Kriterium (b) ueberhaupt fair testbar wird.

## 5. Naechste Schritte

1. **Full-Run** (`workflow_dispatch`, Profil `full`, ~10k Positionen) zur
   Stabilisierung der CIs und PH-Signaturen.
2. Ersetzung Random-Selfplay durch Lichess-Top-Games (Stichprobe 10k, Filter
   Zeitkontrolle 3+2, Elo >= 2500) als Verteilungs-Robustheitscheck — entzaubert
   vermutlich auch die 100 %-Baseline.
3. Lc0-Einbettungen als zweite Repraesentation (Aufgabenschritt 1, optional).
4. Geodaeten-Kruemmung via Pfadintegration im k-NN-Graphen statt reinen Distanzen.
5. Alternative Kriterium-(b)-Baselines (Material-only-Feature, Linearprobe), um den
   Beitrag der topologischen Deskriptoren zu isolieren.
