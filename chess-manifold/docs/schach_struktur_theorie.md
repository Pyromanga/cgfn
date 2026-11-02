# Schach als Hypergraph-Stabilitaet — Strukturtheorie (Entwurf I) und Sekunden-Test

Jonny: "Ich kann es nicht beschreiben — ich dachte du koenntest Schach mathematisch
uebersetzen." Antwortversuch: die Muster aussprechen, und sofort so bauen, dass
sie in Sekunden sterben koennen, wenn sie falsch sind.

## Die vier Saetze

**S1 — Das Spielobjekt.** Schach ist eine Inzidenzgeometrie (Linien: Reihen,
Dateien, Diagonalen; Figuren als Blocker/Projektoren) plus ein bipartiter
Bedrohungs-Verteidigungs-Hypergraph B(x) pro Stellung x. Materiale und alle
klassischen Konzepte sind lineare Funktionsschatten der nichtlinearen
Interaktion dieses Graphen.

**S2 — Erzwingen ist ein Zaehlsatz.** Eine erzwungene Sequenz existiert genau
dann, wenn der Angreifer die Bedrohungszahl ueber der Verteidigungstempo-Zahl
halten kann: Defizit d = Drohungen − parierende Tempi >= 1 UND jede Verteidigung
erzeugt mindestens eine neue offene Drohung. Gabel, Doppelangriff, Ueberladung,
Spiess sind derselbe Satz in verschiedenen Konstellationen.

**S3 — Zugzwang ist Optionenkollaps; Endspieltheoreme sind seine Beweise.**
Opposition = Paritaetssatz, Quadratregel = Geometriesatz, Triangulation =
Temposatz. Endspiele sind beweisbar, weil B(x) so klein wird, dass nur Paritaet
und Reichweiten-Geometrie uebrig bleiben. Das Mittelspiel ist derselbe
Mechanismus mit grossem (nicht mehr lokal schliessbarem) Graphen.

**S4 — Remis ist Stabilitaet, kein Attraktor.** Eine Stellung ist remis-stabil,
wenn jede Drohung-Konstellation ihre Aufloesung schon in der Stellung traegt
(Gegenkonstellations-Reserve). Perfekte Remispartien sind Trajektorien, auf
denen eine Balance-Funktion ihr Einzugsgebiet nie verlaesst (Lyapunov-Kriterium
auf B). Gewinn existiert genau dann, wenn der Angreifer eine Stellung erreicht,
in der die Aufloesbarkeit STRUKTURELL (nicht nur zeitweise) versagt.
Konsequenz fuer die alte Pipeline: Der Drift-Alignment ~ 0 war kein Misserfolg,
sonst die richtige Messung am falschen Objekt — die Dynamik laeuft im
Hypergraphen, nicht im Einbettungsraum.

## Praediktionen (falsifizierbar)

P1: Kippkanten (V-Wechsel) konzentrieren sich dort, wo das Bedrohungs-
     Verteidigungs-Defizit das Vorzeichen wechselt — NICHT dort, wo Material fliesst.
P2: Tilt-Positionen (groesste wp-Spruenge, Run #11/#13: 10 Kandidaten) zeigen
     ein Forcing-Potenzial (Best-Drohungswert gegen unterverteidigte Ziele),
     das mindestens doppelt so hoch ist wie in quiet-draw-Positionen.
P3: Quiet draws zeichnen sich durch maximale Gegenkonstellations-Reserven aus
     (Forcing-Potenzial der GEGNER konvergent null).

## Der Sekunden-Test (C1)

Kein Engine, keine Wahrscheinlichkeit: reines python-chess, ein Forcing-Score
pro Stellung (Bedrohungen gegen unzureichend verteidigte Ziele, SEE-artig,
plus Schachgebot/Matt). Vergleich: 10 Tilt-FENs (Kippkandidaten) vs. alle
quiet-draw-FENs (erste + letzte gesampelte Stellung je Partie). Laufzeit:
Sekunden. Urteil:
  - P2 bestanden: median(F_tilt) >= 2 * max(1.0, median(F_quiet)) ohne grosse Ueberlappung
  - P2 tot: Tilt-Positionen zeigen kein erhoehetes Forcing-Potenzial =>
    S2 in dieser Form falsch (begraben, sauber dokumentiert).

Das ist der Punkt von "wenn du Recht hast, brauchst du nicht viel": Wenn Erzwingen
ein Zaehlsatz auf B(x) ist, muss er sich mit einem Blockzaehler auf den 10
bekannten Kippstellen zeigen lassen. Wenn nicht — falsche Theorie, naechste.
