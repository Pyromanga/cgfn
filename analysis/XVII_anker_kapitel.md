# XVII — Das Anker-Kapitel: Kontrolle, Kapazitaet und die Anatomie des Seed-Lottos
Konsolidierung der Serie G1-G8, eingefroren 2026-11-06. Labor: GitHub-Runner (CPU), Kostenrahmen ~40 EUR/Monat.
Alle Messungen: registriert vor dem Lauf (Docstring), Freeze-Regel, keine Nachverhandlung.

## 1. Verlauf und Verdikte

| Stein | Frage | Verdikt | Kern-Zahl |
|---|---|---|---|
| G1 | Hat der spektrale Knob Kontroll-Autoritaet ueber den getriebenen ben1? | KILL | normaler Arm max 4/10 Treffer (registriert: >=5/10) |
| G2 | Woraus besteht das Seed-Lotto: Basis oder Spektrum? | KILL | std(nur-Basis) ~ std(alles) bei fixem Spektrum |
| G3 | Ist die Antriebs-Ausrichtung der Knopf? | KILL | std_A/std_B = 0.62..1.11, nowhere near 0.5 |
| G4 | Ist die elementweise tanh der Anker? | UNENTSCHEIDEN (Gate blind durch Bias-Confound) | NC kollabierte exakt in 3/4 Zellen |
| G5 | Anker-These in float64, Instrument repariert | PASS | std_NC bis 1e-12 (bis 1e12x Kollaps), Instrument-Delta 0.0 |
| G6 | Traegt der Anker-Knopf Aufgabenleistung? | KILL | MC -40% bis -79% bei std=0 (NC) |
| G7 | Ist der Anker-Winkel ein kontinuierlicher Regler? | KILL | Varianz-Kollaps nur als Kliff bei w=1 |
| G8 | Ist der Zufalls-Anker ein Kapazitaets-Suboptimum? | PASS | +16%/+41% MC bei halbierter Streuung (20 Seeds) |

## 2. Der Mechanismus (stehende These, G5/G8 bewiesen, soweit bekannt unpubliziert)

Das System: dz/dt = -z/tau + W tanh(z) + U u(t). Die elementweise tanh sitzt in
ABSOLUTKOORDINATEN und ist damit der einzige feste Anker im Raum. Bei fixem
Spektrum sind alle W_seed = Q_seed T Q_seed^T orthogonal aehnlich — linear waere
jedes Seed-Lotto unmoeglich. Es entsteht erst durch die Rotation der
Reservoir-Basis gegen die festen tanh-Achsen. Mitfahrende (rotations-
aquivariante) Nichtlinearitaet kollabiert das Lotto auf Rundungsniveau (G5).
Volle Symmetrie aber zerstoert die Kapazitaet (G6): das Lotto ist nicht Rauschen
neben der Rechenleistung, sondern dasselbe Substrat. Partielle Ausrichtung
(G8) ist der sweet spot: mehr Kapazitaet ALS Zufall bei geringerer Streuung.

## 3. Was die Allgemeinheit falsch macht (belegt)

1. Seed-Streuung wird als irreduzibles Rauschen berichtet (mean+-std), obwohl
   sie eine benennbare, konstruierbare Geometrie-Groesse ist (Anker-Winkel).
2. Versteckter Bias-Lotto: model.U ist nn.Linear MIT zufaelligem Bias pro
   Konstruktion. Der Bias allein lottert ben1 staerker (std bis 0.138) als die
   gesamte geometrische Rotation. Gratis-Fix: Bias nullen.
3. Das Feld tunes das falsche Ventil: Spektrum/Radius-Knoepfe haben keine
   Kontroll-Autoritaet (G1); die relevante Achse (Nichtlinearitaets-Rahmen vs.
   Basis) steht auf keinem Standard-Knob-Panel.

## 4. Das Rezept (Stand G8, quantifiziert)

Richtig bauen: (1) Bias nullen. (2) Anker-Fehlwinkel fest setzen (w~0.6,
nicht zufaellig, nicht voll ausgerichtet). (3) Spektrum in Ruhe lassen.
Ergebnis: +16% bis +41% Memory Capacity, Seed-Streuung halbiert bis
gedrittelt, gegenueber dem Feld-Standard. Grenzen: gemessen an getriebenen
Spielzeug-Reservoirs (H<=128, Memory Capacity); keine Aussage ueber echte
Netzwerke oder Aufgaben jenseits delayed recall.

## 5. Bug-Protokoll (ehrlich, fuer die Reproduktion)

- G1: Import-Crash (transient_growth liegt in cgfn.resource, nicht cgfn.field);
  Fix 0c47b6b. Spaeter starb der H=128-Lauf im Diagnostik-Arm (matrix_exp-
  Overflow bei transienter Vergroesserung >1e60) — Verdikt war am H=32-Gate
  bereits besiegelt, kein Rerun.
- G4: Instrument-Gate verglich Bibliothek (MIT Bias) vs. lokal (OHNE Bias) —
  blind; entlarvte den Bias-Confound. Run-1-Crash: Spektrum-Entpackfehler.
- G7: Tangenten-Frame-Bug (dz@F rotiert den Tangentenvektor-Index statt der
  Komponenten): ben-Kurve ungueltig (Chimaere), Primaer (MC, nur Zustand)
  unberuehrt. G5-T-Form bleibt Referenz.
- Umlaut-Gate: mehrfach fast Umlaute in ASCII-Repo gepusht — Scripture-Disziplin
  ae/oe/ue ist Pflicht.

## 6. Recherche-Stand (2026-11-06)

Nicht gefunden (also naechste Nachbarn): rotations-aquivariante Nichtlinearitaeten
als Konstruktionswerkzeug (Tensor Field Networks, CVPR 2021); Schur-Analysen
trainierter RNNs (Task-Restricted Symmetries, 2606.18457); Seed-Varianz als
Folklore mit mean+-std-Bericht (Jaeger/Lukosevicius Review; Nature-RC-Review
2024). Die Kombination "Seed-Lotto = Koordinaten-Artefakt des Ankers, exakt
kollabierbar, mit Bias-Confound, und Zufalls-Anker = Kapazitaets-Suboptimum"
ist nach unserer Recherche unpubliziert. Prioritaets-Anspruch bleibt vorsichtig:
elementarer Mechanismus, pruefbar in 50 Zeilen.

## 7. Offene These (G9, noch nicht gefeuert)

Warum gewinnt ein halb-ausgerichteter Anker gegen Zufall? Hypothese: Der
Fehlwinkel VERTEILT die Nichtlinearitaets-Ressource ueber die Spektral-Moden.
Zufall streut ungleichmaessig, w=1 konzentriert alles auf einen Modus (tot),
w~0.6 verteilt gleichmaessig (optimal). G9 testet: gleichverteilter Fehlwinkel
(UNI) gegen Zufall (RND) und gegen bewusste Konzentration (CONC). Registriert
vor dem Lauf; PASS nur wenn UNI beide Kontraste in >=3/4 Zellen schlaegt.

## 8. Methodik-Regeln dieses Labors

Registrierung im Docstring VOR dem Lauf; Freeze; keine Nachverhandlung;
UNENTSCHEIDEN als ehrenvoller Ausgang bei Instrument-Versagen; Bug-Beichte in
der Konsolidierung; Recherche vor jeder "unbekannt"-Behauptung; keine
Erzaehlung ohne Gate. Das Labor ist billig, die Messung ist es nicht.

---

# Nachtrag (2026-11-06, spaeter): G9 und G10, Kapitel-Anschluss

## G9 — KILL (registriert), mit zwei diagnostizierten Design-Defekten
UNI vs CONC vs RND. Defekte (ehrlich protokolliert): (D1) CONC konzentrierte
auf den GETRIEBENEN Modus und richtete alle anderen versehentlich symmetrisch
aus -> toter Arm (0.4722, std=0), mass "versehentliche Symmetrie" statt
"Konzentration". (D2) Einzel-Seed-Kalibrierung in s=0.8-Zellen (std~0.74) ist
Rauschen. In den praezisen s=0-Zellen hielt die Verteiler-Ordnung (UNI +16/+41%
gegen RND). Verdikt unangetastet; Defekte begruendeten die erlaubte
Design-Reparatur G10 (einmalig, wie G4->G5).

## G10 — KILL (registriert als LETZTER VERSUCH, These endgueltig gefallen)
Budget-matched (sum theta_i gleich): UNI ~ CONC2 ~ SPEC-L (Differenzen im
Rauschen, 4 Zellen), SPEC-H durchgaengig am schlechtesten (Richtung L>H 4/4,
als Kosten-Asymmetrie). Beifang-Erkenntnis: Die Memory Capacity ist bei
budget-matchedem Fehlwinkel eine SKALAR-Funktion des Budgets — die
Verteilung ist (bis auf "nicht auf schnelle Modi") kein designbarer
Freiheitsgrad.

## Endstand des Anker-Kapitels (G1-G10)
1. Bewiesen: Seed-Lotto = Koordinaten-Artefakt der elementweisen tanh (G5,
   exakt, float64); versteckter Bias-Lotto als Confound (G5-Audit).
2. Positiv: Partielle Ausrichtung des Nichtlinearitaets-Rahmens (Budget ~0.6
   rad, nicht auf schnelle Modi) schlaegt den Zufalls-Anker um +16% bis +41%
   MC bei halbierter bis gedrittelter Streuung (G8, 20 Seeds).
3. Negativ (endgueltig): Volle Symmetrie toetet Kapazitaet (G6); kein
   kontinuierlicher Zuverlaessigkeits-Regler (G7); kein Verteiler-Mechanismus
   oberhalb des Skalar-Budgets (G9/G10).
4. Rezept final: Bias nullen, Fehlwinkel-Budget ~0.6 rad flach (oder auf
   langsame Modi), Spektrum unangetastet. Ein Skalar, kein System.
5. Offen (naechstes Kapitel, T-Serie): Transfer der beiden Allgemein-Befunde
   (Bias-Confound, Anker-Framing) in minimale echte Netze.
