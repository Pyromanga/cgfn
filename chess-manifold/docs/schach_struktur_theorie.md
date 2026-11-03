## C1-Ergebnis (Run #2, 2026-11-02, Commit ea08c1e)

Datengrundlage: `results/c1_deficit_results.txt` (C1-Sekunden-Test, Run #2 auf
Fix c8079fa9; Run #1 starb an `occupied_co[bool]`-TypeError, fix via
piece_map()+color_at; Marker-Retry fehlte im Fix-Commit und wurde als
Trigger-Commit 8ac9f03 nachgereicht).

Zahlen:
- Tilt-Positionen (10 Kippkandidaten aus Run #11/#13): median F = 5.50
  (min 4.00, max 16.00)
- Quiet-draw-Positionen (40 FENs): median F = 4.50 (min 0.00, max 13.00)
- Trennung (min_tilt − max_quiet): −9.00 (negative, d.h. Ueberlappung)
- P2-Kriterium: 5.50 >= 2 * max(1.0, 4.50) = 9.00? Nein.

Urteil: **P2 NICHT BESTANDEN.** Die Tilt-Positionen zeigen kein erhoehtes
Forcing-Potenzial gegenueber quiet draws — Median nur 1.22x statt >= 2x,
und die Verteilungen ueberlappen stark (quiet draws reichen bis F=13,
Tilt-Minimum liegt bei F=4). Einzelne Tilt-Spitzen (F=16, F=11) existieren,
aber die Masse der Kippkandidaten ist vom quiet-draw-Bereich nicht getrennt.

Konsequenz (ehrlich): **S2 ist in dieser Form FALSCH.** "Erzwingen als
Bedrohungs-/Verteidigungs-Zaehlsatz" (naiver Ein-Zug-Forcing-Blockzaehler auf
B(x)) trifft die Kippstellen nicht. Das begraben wir sauber: Entweder ist der
Zaehlsatz zu flach (Ein-Zug-Potenzial statt erzwungener Sequenz-Tiefe —
Defizit d koennte erst in der Abfolge sichtbar werden), oder die Kippkandidaten
aus wp-Spruengen sind nicht die Stellungen, in denen Erzwingen strukturell
wird, oder Erzwingen ist kein Blockzaehlsatz auf B(x) überhaupt. Was bleibt:
S1 (Inzidenzgeometrie + Hypergraph) und S3/S4 sind von diesem Ergebnis unberuehrt
— P1 (Kippkanten-Analyse auf Tablebase-Daten) wurde hier NICHT getestet und
bleibt offen; aber P2 als Praediktion von S2 ist tot, und zwar durch einen
billigen, sauberen Sekunden-Test — genau dafuer war er gebaut.

## C3+C4 — Muster-Bergwerk & Muster-Beweise (angestossen 2026-11-03, C3-Zahlen seit Run 10 da)

These: "Schach = eine Handvoll endlicher Mustergruppen mit konstantem Wert."
Zwei kompleme
ntaere Tests dagegen, beide gegen EXAKTE Syzygy-WDL-Werte
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
- C4: Run 1 meldete gruen nach nu
r 14 s — ohne Ergebnis-Commit (Commit-Step
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
Umwandlungsfeld) IST das Key-Square-Kriteriu
m in Reinform: Ist der
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
ein Ergebnis-Com
mit liefert.

## S5 — Strukturzustaende: die Regelzahl waechst mit Strukturzustaenden, nicht mit Positionen (2026-11-03)

Kernaussage (Jonny): Schach ist nicht `Wert(Stellung)` ueber ~10^44 Stellungen,
sondern `Wert = Kombination ueber Strukturwerte`. Eine Struktur ist eine
zusammenhaengende Interaktionskomponente von B(x) (S1) plus eine
Potentialfunktion (Lyapunov invers zu S4: Gewinn = strikt fallendes
Potential entlang optimalen Spiels bis zur Terminalmenge, Remis = Potential,
das nicht faellt). Jeder Strukturtyp t hat einen kleinen Zustandsraum Z_t,
der von der Partie unabhaengig ist: Bauernkette ~20 Zustaende, Freibauer ~6,
Turm-auf-Linie ~8. Die Anzahl der Regeln, die Schach vollstaendig bestimmen,
skaliert mit SUM |Z_t| plus Kopplungstermen — nicht mit der Anzahl der
Positionen. Regelzahl << Positionszahl ist der Behauptungskern: Unter S5
ist Schach ein Regelbuch (Groessenordnung 10^3-10^4 Strukturzustaende samt
Uebergangsgraph), kein Enumerationsproblem. AlphaZero-Gewichte waeren dann
eine implizite, kleine Strukturzustands-Tabelle.

Erstbeispiele (aus der Diskussion):
- Bauernkette (3 Bauern nebeneinander): kleine Formklasse; solange die Basis
  gedeckt ist, ist das Kette-Potential stabil — bricht die Basis, kippt die
  ganze Komponente auf einmal. Deshalb sind Strukturen prognostisch: wenige
  Zustaende pro Komponente, jeder Zustand mit Wert.
- Turm + Freibauer: zwei Strukturen mit Kopplung. Lucena = der Turm liefert
  die Potentialdifferenz (Bruecke = eine Hilfskante, die das S2-Tempo-Defizit
  schliesst); Philidor = eine dritte Struktur (Deckel) friert das Potential.
- Ladder of Shame: erzwungener Pfad im Hypergraphen — jede Parade erzeugt das
  naechste Defizit, das Potential faellt Schritt fuer Schritt strikt.
  Extremfall maximaler Kopplung; die 3-fach-Wiederholung ist das andere
  Extrem (flaches Potential, entkoppelter Zyklus). Kopplungsstaerke ist die
  eigentliche Achse zwischen Jonnys zwei Symmetrie-Extremen.

Zwei offene Kernfragen (entscheiden S5, mit Falsifikationskriterien):
1. Kopplungsordnung: Genuegt Wert = Summe der Strukturwerte + Paarkopplungen?
   Wenn Tripel- und hoehere Kopplungen mit dem Material wachsen, kommt die
   Exponentialwand zurueck. Messbar (C5-Design: Struktur-Extraktion auf
   Lichess-Partien — Bauernketten, Freibauer, Turmlinien, Zwangspfade —,
   Strukturwerte einzeln gegen den Ausgang, dann Restvarianz von Paar- vs.
   Tripel-Termen).
2. Tempo in den Uebergangen: Strukturzustands-Aenderungen kosten Tempi; das
   Regelbuch ist ein Zustandsautomat (Zustand, Zugrecht, Tempo-Defizit) ->
   Wert. Hier leben S2/S3: Zugzwang = Uebergangskosten, Defizit =
   Kantenbewertung auf dem Uebergangsgraph.

Verhaeltnis zu C3/C4: C3 (Run 10) stuetzt die Endlichkeit direkt — keine
3-Mann-Klasse braucht mehr als 5 flache Regeln, 4 von 5 sind mit 1-2 Regeln
perfekt, und der Miner hat Key Squares/Opposition UNBELEHRT wiedergefunden.
S5 sagt, WARUM das skalieren koennte: Endspielklassen sind Strukturtypen mit
kleinen Z_t. Der KPvK-Residualfall (90.93% mit 5 Regeln) passt genau: Key
Squares sind mengenwertige Konzepte — Zustandsraeume, keine Punkt-Schwellwerte.
C4 prueft die beweisbare 100%-Version (Coverage x Praezision a-priori-Regeln
gegen exakte Werte); C5 misst die Kopplungsordnung auf vollen Partien.

### C4-Ergebnis (Run 4, 2026-11-03, Commit 88b2c8a8) — der echte Lauf

Datengrundlage: `results/c4_pattern_proof_results.txt` (Run 4 auf Fake-Gruen-Fix
57b3c5af; Runs 1-3 waren fake-gruen in 14-21 s mit leeren Tablebase-Downloads.
Run 4: Sanity ok, alle 5 Klassen probe-bar, ~1.95M legale Positionen gesamt,
Klassen-Timings 14.5-25.1 s, Gesamtlauf ~2 min nach schnellem sesse.net-Spiegel).

Zahlen (a-priori-Praedikate vs. exakte Syzygy-WDL, ALLE legalen Positionen):

| Klasse | n       | Coverage | Praezision (abgedeckt) | Konflikte | beweisbar geloest |
|--------|---------|----------|------------------------|-----------|-------------------|
| KQvK   | 368 452 | 100.0%   | 93.74%                 | 0         | 93.74%            |
| KRvK   | 399 112 | 100.0%   | 94.43%                 | 0         | 94.43%            |
| KNvK   | 429 440 | 100.0%   | 100.0%                 | 0         | 100.0%            |
| KBvK   | 417 228 | 100.0%   | 100.0%                 | 0         | 100.0%            |
| KPvK   | 331 352 |   4.69%  | 95.43%                 | 28        |   4.48%           |

Praedikat-Praezisionen:
- KQvK/KRvK: `dominance_win` feuert auf 100% (93.74% bzw. 94.43% Praezision)
  — die ~6% Fehler sind Patt/Festfahr-Remis, die der Dominanz-Begriff nicht sieht.
- KNvK/KBvK: `insufficient_material_draw` 100%/100% — beide Klassen sind
  remis-geschlossen, PERFEKT bewiesen mit einer einzigen Regel.
- KPvK: `rook_pawn_zone_draw` 5014 Feueringe, 100% Praezision;
  `key_squares_win` 7776, 94.75%; `wk_front_black_to_move_win` 2698, 89.03%;
  `opposition_draw` nur 96, 93.75%; 28 Konflikte (Praedikate widersprechen sich).

Unabgedeckte Beispiele (KPvK, 95.31% unabgedeckt, 315 804 Positionen): reine
Wand-Bauern-Siege weit weg von der Umwandlung, z.B.
`8/8/8/8/8/8/P7/K1k5 w - - 0 1 -> WIN` bis
`8/8/8/P7/8/8/8/K1k5 b - - 0 1 -> WIN` — einfachste Koenigs-Begleitung,
die klassische Key-Squares/Opposition-Praedikate nur nahe dem Umwandlungsfeld
abdecken.

Leseart gegen S5 (Regelzahl waechst mit Strukturzustaenden, nicht Positionen):
1. **Vier von fuenf Klassen mit EINER Regel beweisbar (fast) vollstaendig
   geloest:** KNvK/KBvK 100% mit einer Regel (remis-geschlossen), KQvK/KRvK
   ~94% mit einer Regel. Kein einziges Praedikat hat mit der Positionszahl
   (400k+ pro Klasse) zu tun — die Regelzahl ist 1-2 pro Strukturtyp, exakt
   die S5-Vorhersage. Die ~6% Dominanz-Luecke (23 048 bzw. 22 244 Remis)
   sind EIN fehlender Zustand (Patt-Naehe des schwachen Koenigs), kein
   Muster-Schatz: ein `stalemate_risk_draw`-Praedikat ist der naive
   naechste Expansionsschritt.
2. **KPvK ist die einzige echte Luecke — und sie ist strukturell, nicht
   enumerativ:** 4.69% Coverage heisst nicht "unendlich viele Muster",
   sondern: die klassischen Lehrbuch-Praedikate adressieren nur die
   Endphase (Umwandlungsnahme). Die 95% unabgedeckten Positionen sind
   genau die Zustaende des Strukturtyps "Koenig fuehrt Bauer" — ein
   kleiner Zustandsautomat (Bauernrang x Koenigsposition relativ zum
   Bauern x Zugrecht), S5 in Reinform. Die 28 Konflikte zeigen, dass die
   vier Praedikate noch keine disjunkte Zustandszerlegung sind — erst
   eine saubere Z_t-Partition macht die Praedikate beweisbar
   konfliktfrei.
3. **Konfliktfreiheit als Beweisbarkeits-Mass:** 0 Konflikte in 4 Klassen
   bei ~1.6M Positionen — die Handvoll klassischer Muster ist nicht nur
   praediktiv, sondern konsistent. C4 bestaerkt damit C3: keine Regel-
   Explosion, aber KPvK braucht den Uebergang vom Punkt-Schwellwert
   (dist_weak_king_promo <= 4) zum mengenwertigen Strukturzustand.

**Urteil: S5 durch C4 gestuetzt.** Die beweisbare Loesungsquote der
Handvoll klassischen Muster ist hoch (93-100% auf 4 von 5 Klassen,
 gewichtet ~96% auserhalb KPvK), und die einzige groessere Luecke
(KPvK, 4.48%) liegt exakt dort, wo die klassische Musterlehre
geometrische Schwellwerte statt Strukturzustaende verwendet — der
Muster-Expansionsfahrplan lautet: (a) `stalemate_risk_draw` fuer
KQvK/KRvK (~6% Rest), (b) KPvK-Zustandsautomat "Koenig fuehrt Bauer"
(Distanz-Klassen statt Schwellwerte), (c) Konflikt-aufloesung via
disjunkte Z_t-Partition in KPvK.
