## 6. Lichess-Robustheitscheck (Run #9, Profil lichess, 2026-11-02)

**Setup:** Echte Turnierpartien statt Random-Selfplay — Top-25-Classical-Spieler via offizielle Lichess-API (Commits 2c863abe/292d5f0c), je 40 Partien, Elo >= 2200, Spielausgang in `positions.csv` (Spalte `outcome`) mitgespeichert. Run #9 gruen, Ergebnisse committet (n = 3.809 Positionen; Train 3.429 / Val 380). Datenquelle: `report.json` -> `config.source = "lichess"`.

**Beckengroessen** (wp-Thresholds 0.55/0.45): **Weiss 1.245 (32.7 %), Remis 2.080 (54.6 %), Schwarz 484 (12.7 %)** — vs. Selfplay #7 (6.588 / 1.206 / 2.365 = 64.8 % / 11.9 % / 23.3 %): Die Verteilung dreht sich von Weiss-Dominanz zu Remis-Dominanz, wie fuer Turnierdaten erwartet; der Selfplay-Weiss-Bias ist damit als Artefakt der Datenquelle bestätigt. *Hinweis:* positions.csv ist nicht im Repo (data/.gitignore); die Spalte `outcome` ermoeglicht erstmals den Abgleich Becken-vs-Spielausgang — fuer den naechsten Run sollten die outcome-Haeufigkeiten direkt in summary.json landen.

**Topologische Signaturen (PH, je Becken):** Remis bleibt das kompakteste Becken (H1-Gesamtpersistenz 11.73 vs. Weiss 13.40; Schwarz 4.40 bei n=484 nicht direkt groessenvergleichbar; H1-max: Remis 0.043 < Schwarz 0.072 < Weiss 0.080; Betti0 @ 0.25: Remis 13 < Weiss 32 / Schwarz 26). Das strukturelle Muster "Remis = am wenigsten fragmentiert" reproduziert sich in Turnierdaten.

**Geodaeten von der Startstellung (Isomap-k-NN, k=10):** d(Start, Remis) = **0.851** < d(Start, Weiss) = 1.214 < d(Start, Schwarz) = 1.311. Fruehes Mittelspiel (Ply 8-20): Remis 0.775 < Weiss 1.073 < Schwarz 1.443 — konsistent.

**Bootstrap-Test (2.000 Resamples, einseitig):**

- d(Start,Remis) - d(Start,Weiss): mean = **-0.363**, 95 %-CI **[-0.396, -0.330]**, p = 0.000

- d(Start,Remis) - d(Start,Schwarz): mean = **-0.461**, 95 %-CI **[-0.518, -0.407]**, p = 0.000

**Kriterium (a) ERFUELLT** — auch in echten Turnierdaten fliesst die Startstellung signifikant ins Remis-Becken. Robust ueber beide Datenverteilungen (Selfplay #7 und Lichess #9).

**Kohaerenz:** k-NN-Becken-Accuracy = **68.4 %** (vs. 82.8 % Selfplay) — ruecklaeufig, plausibel: Turnierpositionen liegen enger beieinander, die Beckengrenzen sind diffuser.

**Vorhersagekraft vs. Baseline (Kriterium b):**

- Topologische k-NN-Genauigkeit: **68.4 %**

- Stockfish-depth-1-Baseline: **99.50 %** (all und masked)

**ZENTRALER BEFUND: Die depth-1-Baseline leidet NICHT — sie bleibt bei 99.50 % (Selplay #7: 99.74 %), also klar ueber der 95 %-Schwelle.** Kriterium (b) bleibt damit auch in Turnierdaten unfairellbar; die in Abschnitt 4 erhoffte "Entzauberung" der ~100 %-Baseline ist ausgeblieben. Grund: Die Beckenlabels stammen aus depth-12-Evals desselben Engines, dessen depth-1-Ansicht fast perfekt damit korreliert — unabhaengig von der Stellungsverteilung.

**Urteil nach der vorregistrierten Entscheidungsregel (Run #9):**

- (a) ERFUELLT (p < 0.001, enge CIs, Remis-Becken signifikant naeher an der Startstellung)

- (b) NICHT ERFUELLT (68.4 % vs. 99.5 % Baseline, Rueckstand ca. 31 Prozentpunkte)

-> **H1 weiter NICHT GESTUETZT.** Der strukturelle Kernbefund (Remis-Attraktor, kompakte Remis-PH-Signatur, Selfplay-Weiss-Bias als Datenartefakt) ist robust; der praediktive Eigenwert gegenueber einer Ein-Ply-Suche ist in beiden Datenverteilungen nicht nachweisbar. Fuer einen fairen Test von (b) muesste die Baseline von den Engine-Labels entkoppelt werden (z. B. Material-only-Linearprobe oder Spielausgang als Label statt Engine-wp) — Datenverteilerweiterung allein (Turnierdaten) macht Kriterium (b) nicht fair.

## 7. Dynamik: Drift-Alignment, Remis-Ursachen & Tilt (Run #11, Profil lichess, 2026-11-02)

**Setup:** Dynamik-Schritt [4b/5] (Commits c10a2543/45773501), 4.224 Trajektorien-Schritte aus 404 Turnierpartien (Lichess, Elo >= 2200). Erstmals direkter Test der These (Jonny): *Remis ist Ergebnis eines unperfekten Moves, fuer Weiss vermeidbar.*

### 7.1 Drift-Alignment (Gradientenfluss-These)

Kosinus-Alignment zwischen Embedded-Drift (Spielzug-Folge) und lokalem Nutzengradienten-Abstieg (n = 3.985 Schritte):

- cos mean = **0.0018**, 95 %-CI **[-0.0054, +0.0101]** — Null klar im CI

- frac_downhill = **50.2 %** — exakt Zufallsniveau

**Gradientenfluss-These NICHT GESTUETZT:** Spielzuege folgen *nicht* dem lokalen Abstieg im Einbettungsraum. Die Mannigfaltigkeit ist kein Nutzenlandscape, auf dem die Spieler downhill rutschen; die Dynamik ist lokal unkorreliert mit dem Gradienten (ordnungsgemaesses Spiel >= 2200 Elo ist lokal nahezu richtungsneutral).

### 7.2 Remis-Ursachen (quiet vs. squandered, n_draws = 111)

Klassifikation der 111 Remispartien nach Trajektorie (quiet = nie aus dem Remis-Becken herausgekommen; squandered = Becken-Schwellen-Vorsprung erreicht, Partie endete trotzdem 1/2-1/2):

- **quiet_draws = 16.2 %** — nur ~1 von 6 Remis ist ein echtes Gleichgewichts-Remis

- **white_squandered = 73.9 %** — ~3 von 4 Remis: Weiss hatte zeitweise einen

Vorsprung (>= Becken-Schwelle 0.55) und hat ihn nicht konvertiert

- black_squandered = 29.7 % (Schwarz-Vorsprung in einem Remisspiel)

Basin-Exit-Raten (Kontext): Weiss-Siege verlassen das Remis-Becken in 97.7 % der Faelle, Schwarz-Siege 92.3 %, Remispartien nur 83.8 %.

### 7.3 Konversionsanalyse: white_edge_conversion (direkter Test der Jonny-These)

Ueber alle 314 Partien, in denen Weiss mindestens einmal einen Vorsprung (wp >= Becken-Schwelle) erreichte:

- **50.0 % gewonnen, 26.1 % remis, 23.9 % verloren**

Weiss konvertiert einen einmal erreichten Vorsprung nur zur Haelfte in einen Sieg; in einem Viertel der Faelle verspielt es den Vorsprung so weit, dass sogar Schwarz gewinnt (Tilt). **Die These ist damit in den ZAHLEN REFLEKTIERT, aber nicht als "Becken-Dynamik" lesbar:** Remis ist hier weniger Gleichgewichtszustand als *Nicht-Konvertieren eines errungenen Vorsprungs* — "vermeidbar, aber teuer", denn die Alternative zum Nicht-Konvertieren ist keineswegs sicher Sieg (50 %), sondern inklusive 24 % sogar Verlust. Der wp-Swing (Mittelwert 0.22 unter Elo 2400, 0.19 bei 2400-2600) zeigt: Grosse Bewertungsspruenge bleiben auch bei hoeherem Elo die Norm — Remis entsteht typischerweise nach erratic history, nicht aus einem stabilen Gleichgewicht (nur 16 % quiet).

### 7.4 Tilt-Beispiele (groesste wp-Spruenge, Top-3)

1. ply 40, d_wp = **-0.52** — 'r1b2rk1/pp2q3/2n1p1pp/6N1/2Bp1n1Q/8/PPP3PP/R2N1RK1 w - - 0 21'

2. ply 44, d_wp = **-0.46** — 'r4rk1/ppp1qpp1/3b3p/8/3P2N1/2P5/PPB2P2/R1BQ2K1 w - - 3 23'

3. ply 28, d_wp = **+0.42** — '2k2b1r/ppp4p/2q1bp2/8/3rp2N/8/PPP2QPP/R1B2RK1 w - - 0 15'

Morse-Kipp-Kandidaten: Einzelzuege, die die Trajektorie von einem Becken in ein anderes kippen (10 Beispiele in results/summary.json, |d_wp| 0.38-0.52).

**Fazit Dynamik:** Die Gradientenfluss-Metapher (Remis als ATTRAKTOR, in den die Dynamik fliessend abgleitet) ist auf Zugebene widerlegt (Alignment ~ 0). Die Daten unterstuetzen eher Jonnys Lesart: Remis ist ueberwiegend *Fehlerfolge* (74 % white-squandered), aber "vermeidbar" nur im Sinne eines 50/50-Risikos — der Preis des Vermeidens ist erhebliche Varianz, nicht garantiertes Gewinnen. Nur die Minderheit der Remis (16 %) ist echter Gleichgewichtszustand.

## 7.5 Öffnungsabdeckung: Sind ALLE Eröffnungen dabei? (Run #13, Profil lichess, 2026-11-02)

**Setup:** Opening-Schritt [4c/5] (Commits c6719a6d/b154826), dieselben gesampelten
Plies wie 7.2 (jeder 4. Halbzug, max. plies_per_game). Frischer Lichess-Pull:
331 Partien, 88 Remis, davon 18 quiet draws (20.5 %).

**Abdeckung (Jonny: "sind ALLE openings dabei?"):**

- **107 einzigartige ECO-Codes**, 171 Eröffnungsvarianten, 8 verschiedene erste Züge
- Verteilung: e4 = 167 (50.5 %), d4 = 119 (36.0 %), Nf3 = 17, c4 = 14, b3 = 8, g3 = 3, f4 = 2, b4 = 1
- e4+d4 decken 86.5 % der Partien ab — breite ECO-Streuung, aber Startzug-Konzentration wie im echten Turnierschach

**Robustheit der quiet-Rate (Jonny: "allein der erste Move kann alles entscheiden"):**

- Gesamt: 18/88 = 20.5 % quiet. Ohne e4-Spiele: 18.6 %. Ohne d4: 18.5 %. Ohne Nf3: 20.7 %. Ohne c4: 21.7 %.
- => Die quiet-Rate ist KEIN Artefakt einzelner Eröffnungen; sie bleibt um ~19-22 % stabil.

**ABER: quiet draws sind NICHT uniform über Eröffnungsfamilien:**

- C (offene Spiele) 36.8 % und D (geschlossene Spiele) 38.5 % quiet — ÜBERREPRESENTIERT
- B (Sizilianer) 11.5 %, A (Flankenvorläufer) 10.0 %, E (Inder) 10.0 % — UNTERREPRESENTIERT
- Nach erstem Zug: e4-Spiele 22.2 % quiet, d4-Spiele 23.5 % — Nf3/c4-Spiele 0 % (kleines n)

**Die 18 quiet draws im Einzelnen** (vollständige Liste in results/summary.json ->
openings.quiet_draw_games, mit FENs): 2x B32 Sizilianer Löwenthal, C01 Französisch
Tauschvariante, D02 Queen's Pawn symmetrisch (2x), C02 Französisch Vorstoß, A45
Indisch beschleunigt, C48 Vier-Springer-Spanier, A48 Barry, C45 Schotten Schmidt,
E11 Bogoljubow-Tausch, D00 Colle, B20 Sizilianer Mengarini, D37 QGD Tarrasch,
C47 Vier-Springer-Schotte, C42 Russisch, D05 Colle-System, C54 Italienisch klassisch.

**Lesart:** Echte Gleichgewichts-Remis konzentrieren sich in symmetrischen,
klassischen Strukturen (Italienisch, Vier Springer, Französisch-Tausch, QGD,
Colle) — genau die Eröffnungen, in denen keine Seite je die Becken-Schwelle
erreicht. In scharfen Eröffnungen (Sizilianer, Inder) gibt es fast keine quiet
draws: Dort entsteht Remis nach erratic history (squandered). Jonnys These
"Remis = Folge unperfekter Moves" gilt ALSO vor allem für die scharfen
Eröffnungen; in den klassischen hat Remis stattdessen Gleichgewichts-Charakter
(~37-38 % quiet statt 10 %). Der erste Move entscheidet nicht über quiet vs.
squandered (beide Hauptzüge ~22-24 %), aber die EröffnungsFAMILIE schon.
