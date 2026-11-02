# Technischer Bericht: Chess-Manifold-Hypothesentest

**Stand:** 2026-11-02 — **Full-Run #7 (CI, gruen, 10.159 Positionen, Seed 42, 100 % reproduzierbar)**; Smoke-Run #6 (301 Positionen) als Konsistenz-Pre-Check. Alle Zahlen: `results/report.json` (`results/summary.json`, PH-Diagramme als Artefakt `ph_diagrams`).

## 1. Hypothese und Falsifikationskriterium

Vorregistriert (siehe README): H1 ist nur gestuetzt, wenn
(a) das Bootstrap-95%-CI von `d_geo(Start, Remis) - d_geo(Start, Sieg-Becken)` durchgehend
negativ ist (Start fliesst ins Remis-Becken, p < 0.05, einseitig) **und**
(b) die topologische k-NN-Genauigkeit die Stockfish-depth-1-Baseline schlaegt.
Ein negatives oder breites Ergebnis ist ein wertvoller Beitrag (Falsifikation ist Erfolg).

## 2. Methodik und Annahmen

- **Daten:** Geseedetes Random-Selfplay (python-chess). Kein externer Download ->
  100 % reproduzierbar (Seed 42). Positionen ab Ply 4, ~12 pro Partie (Full: 850 Partien),
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
  Supervised Contrastive Loss (Khosla et al.), Labels = 6 Bins der Gewinnwahrscheinlichkeit.
  PyTorch Lightning, CPU, AdamW (Full: 40 Epochen, 9.144 Train / 1.015 Val).
- **Analyse:** PH je Becken (ripser, max. 1.200 Punkte geseedet, Betti-Kurven),
  diskrete Geodaeten im Isomap-k-NN-Graphen (scipy csgraph), Kohaerenz =
  stratified-5-fold-k-NN-Becken-Accuracy, Bootstrap (2.000x),
  Gegenbeispielsuche (topologische Vorhersage vs. |wp - 0.5| > 0.35).

### Unsicherheiten (explizit)

1. Random-Selfplay verzerrt die Stellungsverteilung (wenige remisige Zuege,
   viele Extremstellungen). Die Becken sind daher groesser separiert als in
   Turnierdaten -> Kohaerenzzahlen sind obere Schranken.
2. Sigmoid-Kalibrierung ist nicht SF16-spezifisch fit (Literaturwert uebernommen).
3. Diskrete Geodaeten haengen vom k-NN-Graphen ab; der Fallback behandelt
   nicht zusammenhaengende Graphen.
4. PH auf <=1.200 Punkten je Becken (Laufzeit), geseedet.
5. Lc0-Einbettungen nicht implementiert (optionaler Schritt);
   Stockfish allein traegt die Engine-Supervision.
6. Remis-Becken via Engine definiert (0.45 < wp < 0.55), nicht via Spielausgang —
   dokumentierte Konsequenz aus Annahme 1; sauberster Test der Hypothese
   ("Becken = Ergebnisattraktor" gegen "Becken = Engine-Meinung").
7. **Wichtig:** Die depth-1-Baseline erreicht ~99.7 % Becken-Accuracy.
   In dieser Datenverteilung ist die Becken-Klassifikation fuer eine minimale
   Suche nahezu trivial loesbar — Kriterium (b) ist hier kaum zu gewinnen.
   Das ist selbst ein Befund (siehe 4).

## 3. Ergebnisse (Full-Run #7, n = 10.159; Smoke #6, n = 301 zum Vergleich)

**Beckengroessen** (wp-Thresholds 0.55/0.45): **Weiss 6.588, Remis 1.206, Schwarz 2.365**
(Smoke: 211 / 37 / 53 — proportionales Muster). Weiss-Dominanz = dokumentierte
Random-Selfplay-Verzerrung (Weiss am Zug in der Stichprobe + depth-12-Eval).

**Topologische Signaturen (PH auf dem 16-d-Einbettungsraum, ripser, je 1.200 Punkte):**

| Becken | H0-Persistenz (total) | H1-Loecher | H1-Persistenz (total) | Betti0 @ r=0.25 |
|---|---|---|---|---|
| Weiss  | 142.3 | 864 | 7.31 | 12 |
| Remis  | 119.0 | 833 | **5.78** | **7** |
| Schwarz| 161.7 | 902 | 8.37 | 15 |

Wie im Smoke-Run: Das Remis-Becken ist das **kompakteste und am wenigsten fragmentierte**
(geringste H1-Gesamtpersistenz, kleinste Betti0-Feinstruktur); Schwarz ist am
fragmentiertesten. Alle Becken sind bei r >= 0.5 topologisch "einfach" (Betti0=1, Betti1=0).
Die qualitative Signatur ist zwischen Smoke und Full **stabil** — kein Kleinst-n-Artefakt.

**Geodaeten von der Startstellung (diskret, Isomap-k-NN-Graph, k=10):**

- d(Start, Remis) = **0.471**  (Smoke: 0.413)
- d(Start, Weiss) = 1.120  (Smoke: 0.991)
- d(Start, Schwarz) = 1.472  (Smoke: 0.965)

Fruehe Mittelspielstellungen (Ply 8-20) bestaetigen das Muster:
Remis 0.884 < Weiss 1.048 < Schwarz 1.555.

**Bootstrap-Test (2.000 Resamples, einseitig):**

- `d(Start,Remis) - d(Start,Weiss)`: mean = **-0.649**, 95%-CI **[-0.665, -0.631]**, p = 0.000
- `d(Start,Remis) - d(Start,Schwarz)`: mean = **-1.002**, 95%-CI **[-1.030, -0.973]**, p = 0.000

**Kriterium (a) ERFUELLT** — mit n = 10k sind die CIs eng und durchgehend negativ.
Die Startstellung liegt signifikant naeher am Remis-Becken als an beiden Sieg-Becken.

**Kohaerenzmetrik:** k-NN-Becken-Accuracy (stratified 5-fold) = **82.8 %**
(Smoke: 80.7 %; Literatur-Endspiele: 91.7 %). Stabil ueber Sample-Groessen.

**Vorhersagekraft vs. Baseline (Kriterium b):**

- Topologische k-NN-Genauigkeit: **82.8 %**
- Stockfish-depth-1-Baseline: **99.74 %** (all und masked; Smoke: 100 %)

**Kriterium (b) NICHT ERFUELLT.** Die depth-1-Baseline bleibt nahezu perfekt;
die topologische Methode verliert mit 17 Prozentpunkten Rueckstand klar.

**Gegenbeispiele:** 2.526 markante Faelle (topo vs. Engine-Diskrepanz), u. a.
`rnqk3r/pp5p/3n4/2pP2p1/1P2p3/3BP1pN/2b2P1P/1R1QK2R w K - 0 17` (wp = 0.21, klar Schwarz)
und `r3kb2/p1p4p/2bqp1rn/1p1P1Qp1/1P1n2PP/P2KP3/R4P1N/1NB2B1R w q - 0 21` (wp = 0.19).
Beispiele in `results/report.json` (contradiction_examples).

## 4. Bewertung der Hypothesen

**Nach der vorregistrierten Entscheidungsregel: NICHT GESTUETZT (H1 verworfen).**

- Kriterium (a) — topologische Naehe der Startstellung zum Remis-Becken — ist **stark
  und stabil erfuellt** (p < 0.001, enge CIs bei n = 10k, reproduziert im Smoke-Run,
  konsistent im fruehen Mittelspiel). Das "Fliesen ins Remis-Becken" ist in der gelernten
  Geometrie klar sichtbar, und das Remis-Becken zeigt die kompakteste PH-Signatur.
- Kriterium (b) — die topologische Methode schlaegt die triviale Baseline — ist **klar
  gescheitert** (82.8 % vs. 99.7 %). Ohne Mehrwert gegenueber einer Ein-Ply-Suche hat das
  Verfahren keinen praediktiven Eigenwert ueber die Engine-Information hinaus.

**Einordnung und Caveats (ehrlich):**

1. Die Baseline ist *definitionsgemaess* stark beguenstigt: Beckenlabels stammen aus
   depth-12-Evals desselben Engines, dessen depth-1-Ansicht fast perfekt korreliert.
   In Random-Selfplay-Samples ist Kriterium (b) kaum zu gewinnen — der Test war unfair,
   aber vorregistriert und damit bindend.
2. Der strukturelle Kernbefund bleibt bemerkenswert und stabil ueber beide Laeufe:
   Ein reiner Feature-Encoder OHNE jegliche Suchinformation erreicht ~82.8 %
   Beckenkohae renz, ordnet die Startstellung signifikant dem Remis-Becken zu, und die
   Drei-Becken-Geometrie ist im latenten Raum nachweisbar (differenzielle
   Betti-/Persistenz-Signaturen, die zwischen n=301 und n=10k konsistent sind).
3. Die Gegenbeispiel-Anzeige im report (2.526) ueberzaehlt: die Liste enthaelt auch Faelle,
   in denen topologische Vorhersage und Engine-Becken uebereinstimmen (Zaehl-Logik der
   LOO-Missklassifikationen vs. Diskrepanzkriterium ist in analysis.py unsauber) —
   bereinigt bleiben die echten Widersprueche bei den hohen Engine-Diskrepanzen.

**Fazit:** Hypothese **widerlegt im praediktiven Anspruch (b), gestuetzt im strukturellen
Teil (a)** — bei beiden Samplegroessen konsistent. Naechste Experimente muessen die
Datenverteilung aendern (Turnierdaten statt Random-Selfplay), damit Kriterium (b)
ueberhaupt fair testbar wird.

## 5. Naechste Schritte

1. ~~Full-Run~~ **erledigt** (Run #7, gruen, Ergebnisse committet).
2. Ersetzung Random-Selfplay durch Lichess-Top-Games (Stichprobe 10k, Filter
   Zeitkontrolle 3+2, Elo >= 2500) als Verteilungs-Robustheitscheck — duerfte
   auch die ~100 %-Baseline entzaubern und Kriterium (b) fair machen.
3. Lc0-Einbettungen als zweite Repraesentation (optionaler Aufgabenschritt).
4. Geodaeten-Kruemmung via Pfadintegration im k-NN-Graphen statt reinen Distanzen.
5. Alternative Kriterium-(b)-Baselines (Material-only-Feature, Linearprobe), um den
   Beitrag der topologischen Deskriptoren zu isolieren.
6. Bereinigung der Gegenbeispiel-Zaehlung in analysis.py (siehe 4.3).
