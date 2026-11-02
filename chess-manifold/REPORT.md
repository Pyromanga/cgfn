<FILE>
## 6. Lichess-Robustheitscheck (Run #9, Profil lichess, 2026-11-02)

**Setup:** Echte Turnierpartien statt Random-Selfplay — Top-25-Classical-Spieler via
offizielle Lichess-API (Commits 2c863abe/292d5f0c), je 40 Partien, Elo >= 2200,
Spielausgang in `positions.csv` (Spalte `outcome`) mitgespeichert.
Run #9 gruen, Ergebnisse committet (n = 3.809 Positionen; Train 3.429 / Val 380).
Datenquelle: `report.json` -> `config.source = "lichess"`.

**Beckengroessen** (wp-Thresholds 0.55/0.45): **Weiss 1.245 (32.7 %), Remis 2.080 (54.6 %),
Schwarz 484 (12.7 %)** — vs. Selfplay #7 (6.588 / 1.206 / 2.365 = 64.8 % / 11.9 % / 23.3 %):
Die Verteilung dreht sich von Weiss-Dominanz zu Remis-Dominanz, wie fuer Turnierdaten
erwartet; der Selfplay-Weiss-Bias ist damit als Artefakt der Datenquelle bestätigt.
*Hinweis:* positions.csv ist nicht im Repo (data/.gitignore); die Spalte `outcome`
ermoeglicht erstmals den Abgleich Becken-vs-Spielausgang — fuer den naechsten Run
sollten die outcome-Haeufigkeiten direkt in summary.json landen.

**Topologische Signaturen (PH, je Becken):** Remis bleibt das kompakteste Becken
(H1-Gesamtpersistenz 11.73 vs. Weiss 13.40; Schwarz 4.40 bei n=484 nicht direkt
groessenvergleichbar; H1-max: Remis 0.043 < Schwarz 0.072 < Weiss 0.080; Betti0 @ 0.25:
Remis 13 < Weiss 32 / Schwarz 26). Das strukturelle Muster "Remis = am wenigsten
fragmentiert" reproduziert sich in Turnierdaten.

**Geodaeten von der Startstellung (Isomap-k-NN, k=10):**
d(Start, Remis) = **0.851** < d(Start, Weiss) = 1.214 < d(Start, Schwarz) = 1.311.
Fruehes Mittelspiel (Ply 8-20): Remis 0.775 < Weiss 1.073 < Schwarz 1.443 — konsistent.

**Bootstrap-Test (2.000 Resamples, einseitig):**
- d(Start,Remis) - d(Start,Weiss): mean = **-0.363**, 95 %-CI **[-0.396, -0.330]**, p = 0.000
- d(Start,Remis) - d(Start,Schwarz): mean = **-0.461**, 95 %-CI **[-0.518, -0.407]**, p = 0.000

**Kriterium (a) ERFUELLT** — auch in echten Turnierdaten fliesst die Startstellung
signifikant ins Remis-Becken. Robust ueber beide Datenverteilungen (Selfplay #7 und
Lichess #9).

**Kohaerenz:** k-NN-Becken-Accuracy = **68.4 %** (vs. 82.8 % Selfplay) — ruecklaeufig,
plausibel: Turnierpositionen liegen enger beieinander, die Beckengrenzen sind diffuser.

**Vorhersagekraft vs. Baseline (Kriterium b):**
- Topologische k-NN-Genauigkeit: **68.4 %**
- Stockfish-depth-1-Baseline: **99.50 %** (all und masked)

**ZENTRALER BEFUND: Die depth-1-Baseline leidet NICHT — sie bleibt bei 99.50 %
(Selplay #7: 99.74 %), also klar ueber der 95 %-Schwelle.** Kriterium (b) bleibt damit
auch in Turnierdaten unfairellbar; die in Abschnitt 4 erhoffte "Entzauberung" der
~100 %-Baseline ist ausgeblieben. Grund: Die Beckenlabels stammen aus depth-12-Evals
desselben Engines, dessen depth-1-Ansicht fast perfekt damit korreliert — unabhaengig
von der Stellungsverteilung.

**Urteil nach der vorregistrierten Entscheidungsregel (Run #9):**
- (a) ERFUELLT (p < 0.001, enge CIs, Remis-Becken signifikant naeher an der Startstellung)
- (b) NICHT ERFUELLT (68.4 % vs. 99.5 % Baseline, Rueckstand ca. 31 Prozentpunkte)

-> **H1 weiter NICHT GESTUETZT.** Der strukturelle Kernbefund (Remis-Attraktor, kompakte
Remis-PH-Signatur, Selfplay-Weiss-Bias als Datenartefakt) ist robust; der praediktive
Eigenwert gegenueber einer Ein-Ply-Suche ist in beiden Datenverteilungen nicht
nachweisbar. Fuer einen fairen Test von (b) muesste die Baseline von den Engine-Labels
entkoppelt werden (z. B. Material-only-Linearprobe oder Spielausgang als Label statt
Engine-wp) — Datenverteilerweiterung allein (Turnierdaten) macht Kriterium (b) nicht fair.
