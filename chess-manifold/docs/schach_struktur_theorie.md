
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
