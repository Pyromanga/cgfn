# Das perfekte Spiel als berechenbares Objekt — formales Modell

Jonny: "Zu viel menschliches Fehlverhalten, zu wenig Schach als mathematisches Modell."
Dieses Dokument übersetzt die Schach-Befunde (Runs #6–#15) in ein spieltheoretisches
Modell, in dem das perfekte Spiel keine Metapher ist, sondern ein Grenzwert
berechenbarer Folgen.

## 1. Spielgraph und Wertfunktion (Zermelo)

- X: endliche Menge legaler Stellungen (mit 50-Züge-/Wiederholungsregel als
  Terminalisierung => azyklischer Spielgraph).
- T Teilmenge X: Endstellungen, Auszahlung u: T -> {+1, 0, -1} (Weiss/Remis/Schwarz).
- Wertfunktion (Rueckwaertsinduktion / Minimax):

  V(x) = u(x)                      falls x Endstellung
  V(x) = max_{x->y} V(y)           falls Weiss am Zug
  V(x) = min_{x->y} V(y)           falls Schwarz am Zug

- V(x0) ist fuer die Startstellung wohldefiniert und EINDEUTIG.
  Konjektur (Konsens, unbewiesen): V(x0) = 0 — Remis.
- Perfekt vs. perfekt = beide folgen V-erhaltenden Strategien. Der WERT ist
  eindeutig, der PFAD nicht: Pi* = {Pfade mit V konstant} kann gross sein
  ("Flussbett", kein einzelner Faden).

## 2. Die drei Becken sind Niveaumengen von V

  B_W = V^{-1}(+1),  B_D = V^{-1}(0),  B_B = V^{-1}(-1)

Das Remis-Becken B_D ist im perfekten Spiel absorbierend fuer beide Spieler:
kein V-erhaltender Zug verlaesst es. Unsere gemessenen wp-0.55/0.45-Becken sind
Schaetzungen B_D-hat auf Basis von V-hat. Kriterium (a) der Pipeline (Geodaete
d(Start, Remis-Becken) signifikant minimal, p < 0.001, Selfplay UND Turnierdaten)
heisst formal: x0 liegt im INNEREN von B_D-hat — die Startstellung ist eine
Remisstellung im Schaetzmaz.

## 3. Elo-Spieler als temperaturgestoerte optimale Politik

Ein Elo-Spieler ist kein neues Objekt, sondern V-optimale Politik + Rauschen:

  pi_T(a|x) proportional exp(Q(x,a)/T),   T = T(Elo) -> 0

- Q(x,a) = Wert nach Zug a; T = Temperatur (MESSBAR, siehe Run #15).
- Kramers-Exit: Wahrscheinlichkeit, dass eine Partie das Remis-Becken
  verlaesst, ~ exp(-Delta_/T), wo Delta_ die Barrierenhoehe ist = Rückstand,
  den ein Fehler kosten muss.
- Beobachtbare Rauschamplitude: wp-Swing sigma proportional T. Der
  Perfektions-Gradient (Run #15) misst sigma je Elo-Bucket => Fit T(Elo) =>
  Extrapolation T -> 0 IST das perfekte Spiel — nicht metaphysisch, sondern
  Grenzwert einer messbaren Folge.
- Falsifizierbare Vorhersage: Basin-Exit-Raten (gemessen: Weiss-Siege 97.7 %,
  Schwarz-Siege 92.3 %, Remis 83.8 %) skalieren mit exp(-Delta/T).

## 4. Die Engine als Schaetzer von V — Berechenbarkeit als Konvergenz

V-hat_d(x) = Stockfish-Tiefe-d-Bewertung. Das perfekte Spiel ist berechenbar
als Grenzwert:

  V-hat_d -> V-hat_2d -> ... -> V

wie pi durch Numerik — nie "fertig", aber mit MESSBARER Konvergenzrate.
Das macht aus "man muesste nur alles durchrechnen" (Shannon 10^120, praktisch
unmoeglich) eine messbare Frage: wie schnell konvergiert die Folge?

## 5. Falsifizierbare Experimente (naechste Pipeline-Schritte)

E1 — Depth-Leiter: dieselben n Positionen mit Tiefe d in {4, 8, 16, 32, 64}
bewerten; Konvergenz ||V-hat_d − V-hat_2d|| misst die Reserve zwischen unserer
Messung und dem perfekten Spiel. Kollabiert die Verteilung gegen |wp − 0.5| ~ 0,
kollabieren die Schaetzbecken auf das echte B_D und quiet draws werden zu
Samples perfekter Trajektorien.

E2 — Temperatur-Fit: wp-Swing je Elo-Bucket => T(Elo); Kramers-Exit-Rate
vorhersagen und gegen gemessene Basin-Exit-Raten testen.

E3 — Tablebase-Kalibrierung: Endspielstellungen mit bekanntem V(x) (7-Mann-
Tablebase) gegen V-hat_d testen — erste echte Wahrheit ueber V-hat -> V ohne
Perfektionsannahme.

## 6. Was das Modell leistet (und was nicht)

- Loest NICHT: den Wert der Startstellung (bleibt Konjektur bis Tablebase/Beweis).
- Liefert: eine Messkette V-hat_d -> V, T(Elo) -> 0, B_D-hat -> B_D, in der
  jedes Glied falsifizierbar ist.
- Jonnys Remis-Ursachen-These in Modellsprache: Remis aus Fehlerfolge =
  Rauschexkursion T > 0; Gleichgewichts-Remis = Trajektorie in B_D unter
  T -> 0. Beides sind Grenzfaelle EINER Dynamik — kein Widerspruch.
