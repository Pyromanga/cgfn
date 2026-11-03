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

## C3+C4 — Muster-Bergwerk & Muster-Beweise (angestossen 2026-11-03, Zahlen ausstehend)

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

Run-Stand bei Ernte (2026-11-03):
- C3: Runs 1–4 rot bzw. in den Timeout gelaufen (SyntaxError in f-Strings;
  serielle Download-Haenger; 30-min-Timeout trotz Downloads-Fix). Klare
  Ursache gefunden: `random_position()` lehnte Kandidaten per
  `is_insufficient_material()` ab — das ist fuer KBvK/KNvK IMMER True,
  also Endlosschleife, der Miner terminierte nie. Fix gepusht
  (4c05d739), Marker re-fired.
- C4: Run 1 meldete gruen nach nur 14 s — aber ohne Ergebnis-Commit im
  Repo. Zwei offene Punkte: Der Commit-Step verschluckte Push-Fehler per
  `|| true`, und 14 s sind fuer ~2 Mio. Positionen unplausibel kurz —
  vermutlich pro Klasse geschluckte Exceptions. Fixes gepusht
  (Push-Retry-Loop statt `|| true`, Traceback pro Klasse, Artifact-Upload),
  Marker re-fired.

Einordnung gegen die These (ausstehend bis die re-fired Runs gruen sind):
- C3 liefert die Datenbasis: Ist die Regelzahl pro Klasse klein und liegt
  die Regelbaum-Genauigkeit weit ueber der Mehrheits-Baseline, spricht das
  fuer "endlich beschreibbar"; KQvK/KRvK sollten trivial sein, KPvK ist der
  harte Beweisfall.
- C4 macht die These beweisbar falsifizierbar: 100% Coverage × Praezision
  pro Klasse = Klasse ohne Bruteforce geloest; jede Luecke (unabgedeckte
  Menge oder Praedikat-Missbrauch) markiert exakt, wo die klassische
  Musterlehre unvollstaendig ist — das ist der naive Expansionskandidat
  fuer die naechste Mustergruppe.

Zahlen (Regelzahl, Informationsgewinn-Regeln, Coverage × Praezision je
Klasse, Praedikat-Praezisionen, unabgedeckte Beispiele) werden hier
nachgetragen, sobald die Ergebnis-Commits der re-fired Runs vorliegen.
