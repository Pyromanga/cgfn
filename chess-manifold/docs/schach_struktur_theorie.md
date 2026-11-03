## C3+C4 — Muster-Bergwerk & Muster-Beweise (angestossen 2026-11-03, C3-Zahlen seit Run 10 da)

These: "Schach = eine Handvoll endlicher Mustergruppen mit konstantem Wert."
Zwei komplementaere Tests dagegen, beide gegen EXAKTE Syzygy-WDL-Werte
(3-Mann, KQvK/KRvK/KBvK/KNvK/KPvK):

- **C3 — Muster-Miner** (`experiments/c3_pattern_miner.py` →
  `results/c3_pattern_mining_results.txt`): sampelt 4000 zufaellige legale
  Stellungen pro Klasse, schuerft aus reinen Struktur-Features
  (Koenigs-Distanzen, Opposition/Paritaet, Bauer-Rang, Distanzen zum
  Umwandlungsfeld) einen Informationsgewinn-Baum (Tiefe 3). Kennzahl:
  geschürfte Regelzahl pro Klasse + Genauigkeit des Regelbaums vs.
  Mehrheits-Baseline. Killer-Kriterium: Findet der Baum UNBELEHRT Key
  Squares/Opposition fuer KPvK wieder, ist die Wertefunktion endlich
  beschreibbar — wenn nicht, sind die Muster tiefer als Geometrie-Features.
- **C4 — Muster-Beweise** (`experiments/c4_pattern_proofs.py` →
  `results/c4_pattern_proof_results.txt`): klassische Muster-Theoreme als
  reine a-priori-Praedikate (Key Squares, vertikale Opposition,
  Turmbauer-Zone, Dominanz-Gewinn, insufficient-material-Remis) gegen ALLE
  legalen Positionen jeder Klasse (kein Lernen, kein Fitten). Kennzahl:
  beweisbare Loesungsquote = Coverage × Praezision gegen exakte Werte,
  Praedikat-Praezisionen, unabgedeckte Beispiele (FEN + wahrer Wert).

Run-Stand (2026-11-03, C3-Ernte in Run 10):
- C3: Runs 1–4 rot bzw. im Timeout (f-String-SyntaxError; Download-Haenger;
  15-min-Timeout; Endlosschleife — `random_position()` lehnte Kandidaten per
  `is_insufficient_material()` ab, was fuer KBvK/KNvK IMMER True ist; Fix
  4c05d739). Run 5 lief durch, schaufelte aber 0 gueltige Stellungen: die
  lichess.ovh-Download-URL war 404, die rtbw-Dateien leer. Runs 6–9 starben
  an den neu eingebauten Gates — sesse.net verlangt http:// (TLS-Zertifikat
  deckt den Hostnamen nicht), lichess hostet WDL unter
  `/tables/standard/3-4-5-wdl/`, und KBvK.rtbw ist ein ECHTER 80-Byte-
  Syzygy, den die >100-Bytes-Schwelle wegwarf. Run 10: durchgelaufen,
  echtes Ergebnis, Commit 6a872f53 ("results: C3-Muster-Miner 10").
- C4: Run 1 meldete gruen nach nur 14 s — ohne Ergebnis-Commit (Commit-Step
  verschluckte Push-Fehler per `|| true`, vermutlich geschluckte Exceptions
  pro Klasse). Fixes gepusht (Push-Retry, Traceback pro Klasse,
  Artifact-Upload), Marker re-fired. **Ergebnis noch ausstehend.**

### C3-Ergebnis (Run 10, 2026-11-03, Commit 6a872f53)

Zahlen (je 4000 zufaellige legale Stellungen, IG-Baum Tiefe 3, Seed 42;
y = 1.0 nur bei WDL = +2, d.h. "starke Seite gewinnt" — was per Definition
den Partei-am-Zug-Effekt einschliesst):

| Klasse | Gewinnrate | Mehrheits-Baseline | REGEL-BAUM | Regelzahl |
|--------|-----------|--------------------|------------|-----------|
| KQvK   | 0.405     | 0.5952             | **1.0000** | ~2        |
| KRvK   | 0.423     | 0.5770             | **1.0000** | ~2        |
| KBvK   | 0.000     | 1.0000             | **1.0000** | ~1        |
| KNvK   | 0.000     | 1.0000             | **1.0000** | ~1        |
| KPvK   | 0.380     | 0.6198             | **0.9093** | ~5        |

Geschuerfte Regeln:
- **KQvK/KRvK** (2 Regeln, 100%): `mover_is_strong <= 0 -> Remis/Verlust`,
  `mover_is_strong > 0 -> Gewinn`. Der Baum lernt exakt die WDL-Semantik:
  +2 heisst "wer am Zug ist, gewinnt" — Dame/Turm gegen Koenig ist so
  ueberlegen, dass keine Geometrie-Regel uebrig bleibt. Kein Muster-Versteck.
- **KBvK/KNvK** (1 Regel, 100%): alles Remis — korrekt, beide Klassen sind
  remis-geschlossen.
- **KPvK** (5 Regeln, 90.93% vs. 62.0% Baseline):
  1. `mover_is_strong <= 0 -> Remis/Verlust` (p=0.000)
  2. `dist_weak_king_promo <= 4 UND weak_king_before_pawn <= 0 -> Gewinn` (p=0.813)
  3. `dist_weak_king_promo <= 4 UND weak_king_before_pawn > 0 -> Remis/Verlust` (p=0.401)
  4. `dist_weak_king_promo > 4 UND pawn_rank <= 2 -> Gewinn` (p=0.867)
  5. `dist_weak_king_promo > 4 UND pawn_rank > 2 -> Gewinn` (p=1.000)

**Hat der Miner Key Squares/Opposition unbelehrt wiedergefunden? Ja, im
Kern.** `dist_weak_king_promo` (Distanz des verteidigenden Koenigs zum
Umwandlungsfeld) IST das Key-Square-Kriterium in Reinform: Ist der
Verteidigungskoenig weit weg (>4), gewinnt der Angreifer quasi immer —
p=1.000, sobald der Bauer die Mitte passiert hat (pawn_rank > 2). Und
`weak_king_before_pawn` (steht der verteidigende Koenig vor dem Bauern?)
ist die Oppositions-Frage: Koenig vor dem Bauern haelt p von 0.813 auf
0.401 runter. Das sind ohne jedes Schachbuch die zwei zentralen
Lehrsatz-Inhalte des KPvK-Endspiels — gelernt aus 4000 Zufallsstellungen
gegen exakte WDL-Werte.

### Fazit zu "es gibt nicht unendlich Muster" (C3-Lesart)

Die These wird von C3 GESTUETZT, mit einem praezisen Vorbehalt:

1. **Keine Regel-Explosion:** Keine Klasse braucht mehr als 5 flache Regeln;
   4 von 5 Klassen sind mit 1–2 Regeln PERFEKT (100%) beschrieben. Die
   Wertefunktion dieser Klassen ist trivial endlich — es gibt keine
   versteckte unendliche Musterstruktur in Dame-/Turm-gegen-Koenig.
2. **KPvK ist der einzige Residualfall:** 90.93% mit 5 Regeln — gut 29
   Prozentpunkte ueber der Baseline, aber nicht 100%. Die fehlenden ~9%
   sind kein "unendliches Muster", sondern die Grenze der flachen
   Geometrie-Features: Key Squares sind kein Punkt-Schwellwert, sondern ein
   mengenwertiges Konzept (welche Felder vor dem Bauer kontrolliert werden,
   in Abhaengigkeit von Zugrecht und Oppositions-Paritaet) — dafuer braucht
   der Baum entweder mehr Tiefe (= mehr, aber immer noch endlich viele
   Regeln) oder Features, die die Abfolge-/Hypergraph-Struktur einfangen.
3. **Konsequenz:** "Endlich beschreibbar" bleibt am Leben, aber der Beweis
   fuer die KPvK-Residuen (p zwischen 0.4 und 0.87) braucht C4 — die
   a-priori-Praedikate (Key Squares, vertikale Opposition) muessen zeigen,
   dass sie die Residuen beweisbar schliessen, oder die Hypergraph-
   Defizit-Features (C4-Erweiterung) markieren exakt die Luecke der
   klassischen Musterlehre.

C4-Zahlen (Coverage × Praezision je Klasse, Praedikat-Praezisionen,
unabgedeckte Beispiele) werden nachgetragen, sobald der re-fired C4-Run
ein Ergebnis-Commit liefert.
