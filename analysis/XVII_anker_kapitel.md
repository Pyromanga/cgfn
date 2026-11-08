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


---

# Nachtrag 2 (2026-11-06, Nacht): T1 — Feldgesetz-Reparatur, KILL

## T1 — KILL (registriert; Start der T-Serie "Tiefgang")
These: Das repo-eigene Feldgesetz (F17/F18, J = -I/tau + W diag(d), d =
selbstkonsistente sech^2-Gains) ignoriert den Rahmen; die wahre Linearisierung
ist J = -I/tau + W F diag(<sech^2(y)>) F^T mit y = F^T z. Messung: reproduziert
das gerahmte Gesetz den Benettin-ben1 (P1) UND die Rangfolge des Seed-Lottos
(P2) aus Eigenwerten + Einmal-Gain-Messung? Punkte H in {32,128} x s in
{0.0,0.8} x 10 Seeds x lam, Arme N (F=I, alt==neu) und NC (F=Q).

Ergebnis: P1 NEIN (nur 1/4 Zellen <= 0.05), P2 NEIN (2/4 Zellen Spearman
>= 0.8), P3 JA (alt-Fehler 0.73 vs neu 0.085, knapp 2x auf NC aggregiert —
die Reparatur ist real, aber nicht ausreichend).

Lesart (ehrlich, zweischneidig):
1. Die Reparatur selbst ist richtig: Auf NC schlaegt das gerahmte Gesetz das
   alte in ALLEN 4 Zellen (z.B. H=128/s=0.8: 0.183 vs 2.01), und die
   Rangfolgen-Korrelation auf NC ist stark (Spearman 0.855-0.915). Das Feld
   hat seit F17 tatsaechlich am falschen Jacobianer gerechnet.
2. Aber das registrierte Ziel war Vorhersage-Kontrolle, nicht Naeherung: In
   den getriebenen s=0.8-Zellen liegt der Fehler bei 0.09-0.18, und die
   Rangfolge im Standard-Arm N bricht zusammen (H=128/s=0.8: Spearman -0.010
   bei eingeklemmter Vorhersage-Streuung, std 0.08 vs 0.27 gemessen).
   Der Lotto ist keine Eigenwert-Eigenschaft: die Gain-Fluktuation entlang
   der Traektorie (sech^2 ist zeitraegher und schmal) schlaegt jede
   Einmal-Messung des Gain-Mittels. Er bleibt Trajektorien-Eigenschaft.
3. Konsequenz fuer das Rezept: unveraendert (Bias nullen, Budget ~0.6 rad,
   Spektrum in Ruhe). Das Feldgesetz wird in der Dokumentation als
   Mittelwert-Naeherung mit Rahmen-Term gefuehrt, nicht als
   Vorhersage-Instrument.

Ende T1. Kein Rerun (Freeze). Naechster Kandidat T2 nur nach neuer
registrierter These.


---

# Nachtrag 3 (2026-11-07): T2 — Transfer-Stein, ZWEI STRIKES, UNENTSCHEIDEN (mit Explorations-Signal)

## T2a — UNENTSCHEIDEN (Design-Defekt, Signalkuez)
Delayed-XOR mit Luecken 50-90 Frames bei Leck-Halbwertszeit ~3
Frames: alle 4 Arme exakt 0.5000, keine Divergenzen, nichts
gelernt. Fehler des Designers (Uebermut), abgefangen vom
Sanity-Gate. Einmalige Reparatur erlaubt (Praezedenz G4->G5).

## T2b — UNENTSCHEIDEN (ZWEI STRIKES, T2 endgueltig geschlossen)
Reparatur: XOR-Delay <=19 Frames, RK2/tau=1.0, lam_max=0.6,
Thesen und Tore unveraendert. Ergebnis:
  B0_RND:  median 0.4785  std 0.0247   (Chance, alle 10 Seeds)
  B0_AL06: median 0.4785  std 0.0176   (Chance, alle 10 Seeds)
  BR_RND:  median 0.5215  std 0.2106   (bimodal: 0.48..1.00)
  BR_AL06: median 1.0000  std 0.2032   (mehrheitlich perfekt)
Registrierte Auswertung: P1 (Bias-Transfer) JA — std(BR)=0.207
>= 1.5*std(B0)=0.021 (faktor ~9.8x). P2 (Frame-Transfer) NEIN
(1/2 Bias-Zellen). Sanity: Gesamt-Median 0.4785 < 0.60 ->
UNENTSCHEIDEN, kein Verdikt. ZWEI-STRIKES-REGEL: T2 geschlossen.

## Explorations-Befunde (UNREGISTRIERT, nur Material, kein Verdikt)
1. SYMMETRIE-BLOCK (nachtraeglich exakt begruendbar, nicht nur
   empirisch): Mit b=0 und z0=0 ist die Dynamik UNGERADE in
   (x,z). Der Aufgaben-Spiegel (a,b)->(-a,-b) dreht x um, damit
   z und damit die Logits um — aber das Label bleibt gleich.
   Ein B0-Netz mit linearem Readout kann Paritaet daraus
   PROVABELL nicht lernen: alle 20 B0-Runs exakt bei Chance
   sind kein Lotto, sie sind ein Theorem. Konsequenz: Das
   G-Rezept "Bias nullen" ist AUFGABEN-ABHAENGIG — Paritaets-
   Aufgaben brauchen den Bias als Symmetrie-Brecher.
2. FRAME-SIGNAL (nur in BR sichtbar): BR_AL06 median 1.0000 vs
   BR_RND median 0.5215 — die Mehrheit der AL06-Runs loest die
   Aufgabe PERFEKT, RND bleibt mehrheitlich Chance. Das ist
   das staerkste Unregistrierte Signal seit G8: moeglicherweise
   war P2 nur deshalb NEIN, weil die B0-Zellen symmetrie-
   blockiert waren (Chance vs Chance ist kein Vergleich).
3. P1 hat mit Faktor ~9.8x ueberlebt — der Bias-Lotto ist im
   trainierten Netz massiv sichtbar.

Naechster Schritt: T3 als NEU registrierter Stein (keine T2-
Reparatur), Design lernt aus dem Symmetrie-Block: Bias in allen
Armen (als Symmetrie-Brecher unentbehrlich), registrierte Tore
VOR dem Lauf, frische Seeds. Kein Verdikt aus T2 ableitbar.


---

# Nachtrag 4 (2026-11-07): T3 — Primartoere PASS, Sanity-Klausel selbst gemurkst, formales Verdikt UNENTSCHEIDEN

## T3 — Anker auf Paritaet (frische Seeds 10..19, Bias in allen Armen)
  RND:  median 0.5215 | Perfekt-Fraktion 0.10 (nur Seed 13 perfekt)
  AL06: median 1.0000 | Perfekt-Fraktion 0.70 (7/10 perfekt, 1x 0.72)
  P1 (Effekt >=0.15): JA — Effekt 0.4785, dreifache Schwelle.
  P2 (Perfekt >=0.5 und >=2x RND): JA — 0.70 vs 0.10, siebenfach.
  Formales Verdikt: UNENTSCHEIDEN, denn die Sanity-Klausel
  (Gesamt-Median >= 0.60) riss: median(0.5215, 1.0000) = 0.5215.

## Ehrliche Fehleranalyse: DIE KLAUSEL WAR DEFERT, NICHT DER VERSUCH
  Die Sanity-Klausel verlangte, dass das GESAMT-Netz lernt — aber
  die These SAGT VORHER, dass der Zufalls-Anker NICHT lernt (das
  ist in G2/G5 das Schicksal des Zufalls-Rahmens). Ein Gate, das
  den experimentell vorhergesagten Misserfolg des Kontrollarms als
  Abbruchkriterium zaehlt, ist ein Registration-Defekt: Das Gate
  hat die These nicht verstanden, die es bewachen sollte. Richtig
  waere gewesen: Sanity = AL06 lernt (median(AL06) >= 0.60),
  RND darf scheitern (das ist das RESULTAT, nicht ein Abbruch).

## Stand der Evidenz (unaufgeregt zaehlen)
  T2b (Seeds 0..9): AL06 8/10 perfekt vs RND 2/10 (bimodal).
  T3  (Seeds 10..19, frisch): AL06 7/10 perfekt vs RND 1/10.
  Kombiniert 20 frische Seeds pro Arm: 15/20 vs 3/20 perfekt,
  identisches Instrument, identisches Spektrum, nur der Rahmen
  lottert. Das ist kein Zwanzigstel-Signal mehr — das ist eine
  Wand. ABER: formales Verdikt bleibt UNENTSCHEIDEN, Freeze
  gilt, keine Nachverhandlung am geschlossenen Lauf.

## Konsequenz (registrierungstreue)
  Die Anti-Steinchen-Klausel von T3 gilt: KEIN vierter Versuch
  auf Paritaet. Der Weg nach vorn ist GENERALISIERUNG, nicht
  Wiederholung: T4 mussel dieselbe These (Anker schlaegt
  Zufall im echten Training) auf EINER ANDEREN nichtlinearen
  Aufgabenfamilie mit korrigierter Sanity-Klausel (lernender
  Arm muss lernen, Kontrollarm darf scheitern) pruefen.
  Pass auf Paritaet + Pass auf zweiter Familie = Transfer
  bewiesen; Misserfolg auf zweiter Familie = Paritaets-
  Spezialfall, ehrlich verbuchen.


---

# Nachtrag 5 (2026-11-07): T4/T4b — Produkt-Familie TOT (zwei Strikes, UNENTSCHEIDEN)

## T4 (Delay 8, Seeds 20..29) — UNENTSCHEIDEN (Strike 1)
Beide Arme R2 ~ 0.000, Effekt exakt 0, keine Divergenzen.
Diagnose: Delay 8 jenseits der bewiesenen f34-Basis (Delay 3);
u_{t-8} nur mit ~2% Amplitude im Zustand; Produkt braucht
zwei Faktoren multiplikativ. Signalmangel, kein
These-Statement. Korrigiertes Sanity-Gate (T3-Lehre)
funktionierte exakt wie beabsichtigt.

## T4b (Delay 3, 1200 Schritte, Seeds 30..39) — UNENTSCHEIDEN (Strike 2)
  RND:  median R2=-0.0006, geloest 1/10 (nur Seed 32: 0.972)
  AL06: median R2=-0.0005, geloest 2/10 (Seeds 32, 34: ~0.97)
Mediane bei 0, Effekt 0.0001, P1/P2 NEIN, Sanity gerissen
(median AL06 < 0.5). Produkt-Familie per Registrierung TOT.

## Lesart (explorativ, unregistriert)
1. Auch mit f34-Basis-Delay ist das Trainingsverhalten ALL-OR-
   NOTHING: Runs erreichen entweder ~0.97 oder exakt ~0 — die
   Mediane beider Arme bleiben bei 0, weil die Mehrheit im
   Optimierungs-Loch sitzt (Readout startet bei 0, MSE-Gradient
   ueber 200 RK2-Schritte zu flach). Das ist ein OPTIMIERUNGS-
   Phaenomen, kein Kapazitaets-Statement: Die geloesten Runs
   beweisen, dass das Netz die Aufgabe loesen KANN.
2. Schwaches Echo des Paritaets-Musters in der geloest-Fraktion
   (AL06 2/10 vs RND 1/10) — nicht signifikant, kein Verdikt,
   nur Material.
3. Ehrliches Protokoll-Ergebnis (registriert verbucht):
   BPTT-Regression auf Produkt-Aufgaben ist im 40-EUR-Labor
   nicht sauber feuerbar — der Flaschenhals ist der Adam-BPTT-
   Optimierer auf flacher MSE-Landschaft, nicht das Netz.

## Stand der T-Serie nach T4b
  T1  KILL    — Feldgesetz als Vorhersage-Tool tot; Reparatur
                real (Rahmen-Term), Lotto bleibt Traektorien-
                Eigenschaft. Echter negativer Tiefgang-Befund.
  T2  2 Strikes UNENTSCHEIDEN — aber P1 (Bias-Lotto) mit
                Faktor ~9.8x bestanden und SYMMETRIE-THEOREM
                entdeckt: Ohne Bias ist Paritaet provabel un-
                loesbar (ungerade Dynamik + linearer Readout).
  T3  UNENTSCHEIDEN (Gates P1/P2 beide PASS mit grossem
                Abstand; Sanity-Klausel war defekt und hat das
                vorhergesagte Kontrollarm-Scheitern als Abbruch
                gezaehlt). Evidenz-Wand: 15/20 vs 3/20 perfekt
                (T2b+T3, 20 frische Seeds je Arm).
  T4/T4b UNENTSCHEIDEN, Produkt-Familie TOT (Optimierungs-
                Phaenomen, all-or-nothing).

## Offene Frage, die die Serie aufwirft
Das Paritaets-Signal ist die staerkste unveraerifizierte
Evidenz der Serie. Zwei mal zwei Familien-Tests wurden von
Instrument-Gates gebremst, nicht von der Natur. Die naechste
registrierte These sollte MECHANISMUS fragen statt Familie
wiederholen: WARUM trainiert der 0.6-rad-Anker, wo der
Zufalls-Anker im Optimierungs-Loch bleibt (Gradient-Normen,
Verlustkurven, Jakobi-Spektrum waehrend des Trainings) —
oder die Serie ehrlich schliessen.


---

# Nachtrag 6 (2026-11-07): T6 — Lernbarkeits-Spektrum, KILL: Die Geometrie ist Entkopplung

## T6 — Budget-Sweep auf Paritaet (Seeds 40..49, Bias in allen Armen, Spektrum fix)
  Budget 0.00 (exakt diagonal): P_lernt 0.70 | DM 48.0
  Budget 0.15:                 P_lernt 0.60 | DM 31.1
  Budget 0.30:                 P_lernt 0.60 | DM 22.6
  Budget 0.60:                 P_lernt 0.80 | DM 13.5
  Budget 1.20:                 P_lernt 0.30 | DM  5.5
  RND (Zufalls-Rahmen):        P_lernt 0.10 | DM  2.6
  Registriert: P1a (Peak B60) JA, P1b (0.60 minus 0.00 >= 0.3)
  NEIN (Differenz 0.10, Rauschen bei n=10), P2 JA. VERDICT: KILL.

## Was der Kill aufklaert (der eigentliche Ertrag)
1. DIE NICHT-MONOTONIE-These IST TOT: Exakt diagonales W (Budget 0,
   null Geometrie-Freiheit) trainiert bei P_lernt 0.70 — nahezu
   so gut wie das 0.6-Budget (0.80, Rauschen). Volle Ausrichtung
   toetet das Lernen NICHT. Die trainierte Welt kennt kein
   G6-Echo im Fehlwinkel. Der Innen-Peak von 0.8 vs 0.7 ist
   bei n=10 nicht von Rauschen zu trennen.
2. DER ENTKOPPLUNGS-KONFUND IST BESTAETIGT: P_lernt folgt der
   diag-Masse DM fast monoton (48->0.70, 31->0.60, 23->0.60,
   13.5->0.80, 5.5->0.30, 2.6->0.10; B60 minimal ueber dem
   DM-Trend, innerhalb des Rauschens). Die T2b/T3-Wand wird
   von Selbst-Schleifen-Masse getragen, nicht von Anker-
   Geometrie.
3. RETROKLASSE: G8s Geometrie-Befund (+16..41% MC im Reservoir,
   Ridge-Readout, fixes W) ueberlebt im trainierten Netz NICHT
   als eigener Faktor. Die zwei Regime sind verschieden:
   Reservoir = Geometrie-Regime, Training = Entkopplungs-
   Regime. Das ist ein echter, publikationsfaehiger Kontrast-
   Befund, kein Messfehler — aber er begraeibt die Transfer-
   Hoffnung der Anker-Geometrie.

## Ehrliches Ende der T-Serie (Bilanz)
  T1   KILL — Feldgesetz als Vorhersage-Tool tot (Lotto ist
       Traektorien-Eigenschaft). Echter negativer Befund.
  T2b  P1 bestanden (Bias-Lotto 9.8x) + Symmetrie-Theorem
       (ohne Bias ist Paritaet provabel unloesbar).
  T3   Formales UNENTSCHEIDEN (eigener Gate-Defekt);
       P1/P2 beide mit Abstand PASS — Evidenz-Wand war echt,
       aber nach T6 als Entkopplung erklaert, nicht als
       Geometrie.
  T4/T4b UNENTSCHEIDEN, Produkt-Familie tot (Adam-BPTT-
       Optimierungsphaenomen, all-or-nothing).
  T6   KILL — Geometrie = Entkopplung; kein Innen-Peak.
  UEberlebende echte Befunde der T-Serie:
    (i) Symmetrie-Theorem (Bias als lern-noetiger Symmetrie-
        Brecher auf Paritaet),
   (ii) Bias-Lotto im trainierten Netz (9.8x),
  (iii) Reservoir-vs-Training-Regime-Kontrast (Geometrie vs
        Entkopplung),
  (iv) Optimierungs-Loch: all-or-nothing-Trainierbarkeit ist
        der Flaschenhals im 40-EUR-Labor, nicht Kapazitaet.


---

# Nachtrag 7 (2026-11-08): T7 — KILL, aber mit zurueckgenommenem Verdikt-Text: MC ist im T-Bau FLAT; G8 unberuehrt

## T7 (T-Bau-Budget-Sweep, Reservoir-MC, Seeds 50..59, b=0)
  B00 1.9363 | B15 1.9534 | B30 1.9488 | B60 1.9322 | B120 1.9241
  | RND 1.9267. DM variiert um Faktor 19 (48.0 bis 2.5), MC
  um 1.5%. Registriert: P1b riss (1.93 vs 2.13) -> KILL.

## Korrektur des eigenen Verdikt-Texts (Selbstanklage)
  Der registrierte KILL-Text behauptete "G8 war Entkopplung"
  und eine Rueckziehung. FALSCH. Die Daten zeigen ein
  Drittes: MC ist im T-Bau FLAT — weder Geometrie (kein
  Innen-Peak, B15 vorne mit 0.9% = Rauschen) NOCH Entkopplung
  (19x DM aendert MC um 1.5%) bewegt die Kapazitaet. Der
  T-Knopf ist fuer Reservoir-MC schlicht irrelevant.

## Versoehnung mit G8 (Original-Code nachgelesen)
  G8 rotiert NICHT Ws Eigenbasis, sondern den tanh-Rahmen:
  W = Q T Q^T pro Seed FIX, F(w) = Q @ QR(w*I + (1-w)*P0),
  Antrieb U = Q[:,0] (Top-Eigenmode). In G8s Bau ist DM ueber
  das gesamte w-Gitter PER KONSTRUKTION konstant. Der
  Entkopplungs-Angriff trifft G8 von vornherein nicht.
  T7 testete den T-Serien-Knopf — nicht G8s Knopf.

## Neue Karte (Stand nach T7)
  T-Knopf  (Eigenbasis vs. tanh-Koordinaten): wirkt auf
            TRAINIERBARKEIT via DM (T6), NICHT auf
            Reservoir-MC (T7, flat).
  G8-Knopf (tanh-Rahmen vs. Eigenbasis, W fix, DM konstant,
            Eigenmoden-Antrieb): wirkt auf Reservoir-MC
            (G8, +16..41%), im Training NIE getestet.
  Offene Frage -> T8: G8s Mechanismus im trainierten Netz.
  Da DM dort per Konstruktion konstant ist, ist jeder
  Trainierbarkeits-Unterschied PURE Geometrie.


---

# Nachtrag 8 (2026-11-08): T8/T8b — G8-Knopf im Trainingslabor UNENTSCHEIDEN (zwei Strikes); Diagnose: Antriebs-Verhungerung

## T8 (dichte Zufalls-Basis, DM 2.55, Seeds 60..69) — UNENTSCHEIDEN
Alle 50 Runs bei Chance, P_lernt 0.00 ueberall, DM-Konstanz
perfekt. Post-Mortem: Basis-Lernbarkeit dieser Familie ist
laut T6 ~0.10 (RND-Punkt) — 0/50 perfekt ist bei dieser
Basis-Rate Statistik, kein These-Statement. Reparatur erlaubt.

## T8b (trainierbare T6-B60-Familie, DM 13.63, Seeds 70..79) — UNENTSCHEIDEN (Strike 2)
Erneut ALLE 50 Runs bei Chance, P_lernt 0.00 in allen fuenf
w-Armen, DM-Konstanz perfekt (Streuung exakt 0). ZWEI STRIKES:
G8-Knopf-Linie im Trainingslabor geschlossen, keine dritte
Reparatur.

## Diagnose (das eigentliche Ergebnis, exakt begruendbar)
Der G8-Mechanismus-F transferiert nicht nur den Rahmen, sondern
auch die ANTRIEBS-KONZENTRATION: Uw = 0.5*R[:,0:2] speist beide
Input-Kanaele NUR in die Top-2-Eigenmoden. In y-Koordinaten
(y = z@F, w=1: F=R) wird das System sichtbar: W wird diagonal,
der Antrieb trifft NUR die Einheiten 0 und 1 — die restlichen
46 Einheiten sind abgekoppelt (diagonales W + lokal getriebene
tanh: keine Kommunikation). Das funktionale Netz schrumpft auf
~2 aktive Einheiten. Paritaet mit 2 hidden units und linearem
Readout: Chance. Das ist kein Rahmen-Effekt, das ist
ANTRIEBS-VERHUNGERUNG.
Warum G8 trotzdem funktioniert: Das Reservoir-Ridge-Regime
erntet LINEARE Moden — dort ist Konzentration auf die Top-
Eigenmode ein VORTEIL (Signal-Rausch-Verhaeltnis). Das
Trainingsregime braucht verteilten Antrieb, damit alle Einheiten
am Gradienten teilnehmen. Die zwei Regime verlangen
entgegengesetzte Antriebs-Geometrie.

## Regime-Karte v2 (endgueltiger Stand)
  RESERVOIR-RIDGE-REGIME (fixes W, Ridge, MC):
    - Rahmen-Geometrie (G8-Knopf): WIRKT (+16..41%, G8;
      DM dort per Konstruktion konstant — nicht Entkopplung).
    - T-Knopf (Eigenbasis vs. Koordinaten): wirkungslos fuer
      MC (T7, flat bei 19x DM).
    - Antriebs-Konzentration auf Eigenmoden: vorteilhaft
      (G8-Mechanismus).
  TRAININGS-REGIME (BPTT, Adam):
    - T-Knopf via DM: WIRKT (T6: Lernbarkeit folgt der
      Selbst-Schleifen-Masse, 19x DM -> 0.10 bis 0.80).
    - G8-Knopf (Rahmen bei fixem W): NICHT FEUERBAR mit dem
      G8-Antriebs-Paket (T8/T8b: Verhungerung, 0/50).
    - Antriebs-Konzentration: toedlich (T8b-Diagnose).
  Offen, untestbar im 40-EUR-Rahmen bis jetzt: ob der Rahmen-
  Winkel bei verteiltem Antrieb und fixem W Trainierbarkeit
  moduliert. T8c-mit-diffusem-Antrieb waere die Frage — aber
  dann ist es nicht mehr G8s Mechanismus, sondern ein neues
  Design (neue These, neue Registrierung).

## T-Serie: Finale Bilanz der ueberlebenden Befunde
  1. T1 KILL: Feldgesetz ist keine Vorhersage-Maschine; der
     Lotto ist eine Traektorien-Eigenschaft. (negativ, echt)
  2. T2b: Bias-Lotto im trainierten Netz (std-Faktor 9.8).
  3. T2b: SYMMETRIE-THEOREM — ohne Bias ist Paritaet provabel
     unloesbar (ungerade Dynamik, linearer Readout, Spiegel-
     Argument). Das G-Rezept "Bias nullen" ist aufgaben-
     abhaengig.
  4. T4b: All-or-nothing-Optimierung (Plateau-Loch) ist der
     Flaschenhals im Trainingslabor, nicht Kapazitaet.
  5. T6/T7: Zwei-Knopf-Karte — DM-Knopf wirkt auf Training,
  nicht auf MC; Rahmen-Knopf wirkt auf MC (G8), nicht
     nachweisbar im Training.
  6. T8/T8b: ANTRIEBS-VERHUNGERUNG — die Regime brauchen
     entgegengesetzte Antriebs-Geometrie (konzentriert fuer
     Ridge-Ernte, verteilt fuer Training). Neue, eigene
     Struktur-Variable (Antriebs-Verteilung), entdeckt als
     Nebenprodukt des gescheiterten Transfers.
Ende der T-Serie. Kapitel geschlossen mit der Regime-Karte v2.


---

# Nachtrag 9 (2026-11-08): T9 — UNENTSCHEIDEN (Sanity gerissen); Instrument-Diagnose: kohaerenter Moden-Antrieb verhungert bei JEDEM d

## T9 (Antriebs-Rang d in {2,4,8,16,48}, Seeds 80..89, W=T6-B60-Familie fix, DM 13.8515 exakt konstant)
Alle 5 Arme: P_lernt = 0.00, Paritaet bei Chance (0.4785/0.5215), 0 Divergenzen, DM-Streuung exakt 0. VERDICT: UNENTSCHEIDEN — Sanity gerissen (auch d=48 lernt nicht, Instrument defekt gegenueber T6-Basis). Registriert: EINE Reparatur erlaubt (neue Linie) — angetreten als T9b. Laufzeit 12.4 min, Run-Protokoll sauber (Run-1-Crash: NameError math vor Fix; Run-2: Infra-Tod vor Script-Start, kein Log; Run-3 sauber).

## Sekundaer, aber scharf: die MC-Kurve
MC steigt STRICT MONOTON mit d: 0.7775 (d=2), 0.8976 (4), 1.3535 (8), 1.6526 (16), 1.7047 (48) — bei konstantem DM und konstanter Energie pro Kanal. Im Reservoir-Ridge-Regime ist breite Moden-Abdeckung ALSO GUT: P2 (MC(2) >= 0.90*MC(48)) verfehlt klar (0.78 < 1.53). Erste direkte Evidenz RICHTUNG KILL-B: Diffusion ist auch fuer die Ridge-Ernte gut — der G8-Konzentrations-Vorteil (U=Q[:,0]) ist im weissen 2-Kanal-Antrieb nicht reproduzierbar.

## Instrument-Diagnose (der eigentliche Ertrag)
T9 vergass den T6-Referenzpunkt im selben Lauf: T6 lernt mit Koordinaten-Zufalls-Antrieb (Uw = 0.5*randn(H,2), beide Kanaele ueber alle 48 Einheiten, Phasen zufaellig) bei P_lernt 0.70-0.80. T9s Antrieb v_c = R[:,subset] @ ones/sqrt ist pro Kanal RANG-1 in die Eigenbasis, Kanaele mod-getrennt (Kanal 0 -> erste, Kanal 1 -> letzte d/2 Moden), Phasen kohaerent. Selbst d=48 ist KEINE Diffusion im T6-Sinn: keine Phasenmischung, kein Kanal-Overlap, keine Kopplung an den Koordinaten-Anker der tanh. Der Knopf drehte an der Moden-ZAHL, aber die Lernbarkeit haengt vermutlich an RICHTUNG/PHASE. Testbar in einem Lauf.

## T9b (Reparatur-Linie, registriert VOR dem Lauf)
Arme: REF (exaktes T6-Instrument: Uw = 0.5*randn(H,2)) als Lernbarkeits-Sanity + die T9-d-Familie UNVERAENDERT (W T6-B60 pro Seed fix, Spektrum fix, b = 0.1*randn fix, Energie pro Kanal 0.5). SANITY: P_lernt(REF) >= 0.5, sonst UNENTSCHEIDEN und Serie-Post-Mortem (KEINE weitere Paritaets-Reparatur). PRIMAER: P1 P_lernt(48)-P_lernt(2) >= 0.3; P2 MC(2) >= 0.90*MC(48). PASS = Sanity UND P1 UND P2 (Doppeldissoziation: Training braucht Moden-Abdeckung, Ridge braucht sie nicht). KILL-A: P_lernt(2) >= 0.5 (Konzentration trainiert, Verhungerungs-Story falsch). KILL-B: P1 UND nicht P2 (auch Reservoir leidet — Diffusion universell; T9-MC-Kurve deutet hierher). KILL-C (NEU, die Phasen-Wette): Sanity JA, P1 NEIN — Lernbarkeit ist KEINE Funktion der Moden-Abdeckung; ist zusaetzlich P_lernt(48) = 0, verhungert JEDE kohaerente Eigenmoden-Summe: Es ist die RICHTUNG, nicht die ZAHL (Verhungerungs-These verfeinert, T8/T8b vollstaendig erklaert). Sekundaer: MC(REF) vs MC(d)-Kurve, DM, Laufzeit. Seeds 90..99. Workflow-Gate erweitert: Import-Check via runpy (run_name != __main__) nach pip, schliesst die py_compile-Blindstelle (fehlende Imports).


---

# Nachtrag 10 (2026-11-08): T9b — KILL-C. Es ist die RICHTUNG, nicht die ZAHL. Und: MC ist eine Funktion der Moden-Abdeckung, Training eine der Richtungs-Struktur.

## T9b (REF-Arm + d-Familie, Seeds 90..99, W=T6-B60-Familie fix, DM 13.9842 exakt konstant, Laufzeit 10m47s)
SANITY: REF (exaktes T6-Instrument, Uw = 0.5*randn(H,2)) lernt P_lernt = 0.90, 9/10 perfekt (median acc 1.0000; nur Seed 94 versagt) — Instrument faehig, Vergleich legitim. ALLE fuenf d-Arme P_lernt = 0.00 (Chance 0.48-0.52), 0 Divergenzen, DM-Streuung exakt 0. VERDICT: KILL-C — die Phasen-Wette verloren.

## Der Befund (verfeinerte Verhungerungs-These)
Trainierbarkeit ist KEINE Funktion der Moden-Abdeckung: Selbst d=48 (beide Kanaele auf je 24 Eigenmoden, kohaerente Summen, Energie pro Kanal konstant 0.5) verhungert. JEDE kohaerente Eigenmoden-Summe toetet das Training, jede iid-Koordinaten-Antriebsmatrix (REF) lernt. Es ist die RICHTUNG (Phasenmischung / Basis-Bezug), nicht die Zahl der getroffenen Moden. T8/T8b damit endgueltig erklaert: Der G8-F-Transfer schleppt die Antriebs-RICHTUNG mit, und genau daran stirbt das Training — nicht am Rahmen.

## Die saubere Dissoziation der beiden Variablen (Nebenbefund, sekundaer, aber strukturbildend)
MC-Kurve: strikt monoton in d (0.8519, 0.9012, 1.3462, 1.6743, 1.6973). Und: MC(REF) = 1.6549 liegt AUF der Kurve beim 48-Moden-Wert — der Koordinaten-Zufalls-Antrieb regt in der Eigenbasis saemtliche 48 Moden mit zufaelligen Amplituden an. Damit: MC haengt an der ZAHL (Spektral-Verteilung) der angeregten Moden und ist gegenueber der Richtungs-Struktur ROBUST; Training haengt an der RICHTUNG und ist gegenueber der Moden-Zahl FRAGIL. P2 verfehlt (MC(2) < 0.90*MC(48)): Der G8-Konzentrations-Vorteil ist ein Artefakt des Ein-Moden-Ein-Kanal-Designs, kein Regime-Gesetz. Der Regime-Kontrast von Nachtrag 8 (konzentriert gut fuer Ridge, verteilt gut fuer Training) kollabiert teilweise: Fuer die Ridge-Ernte ist Verteilung ebensogut.

## Regime-Karte v3
  RESERVOIR-RIDGE-REGIME (fixes W, Ridge, MC):
    - MC folgt der Moden-Abdeckung (T9/T9b-Monotonie), richtungs-robust (REF auf der Kurve).
    - Rahmen-Geometrie (G8-Knopf): WIRKT bei konzentriertem Ein-Moden-Antrieb (+16..41%),
      aber KEIN generelles Konzentrierungs-Gesetz im 2-Kanal-Regime (T9b P2 NEIN).
    - T-Knopf: wirkungslos fuer MC (T7).
  TRAININGS-REGIME (BPTT, Adam):
    - DM-Knopf (T6) wirkt; Antriebs-RICHTUNG wirkt (T9/T9b: kohaerente
      Eigenmoden-Summen verhungern bei jedem d, REF lernt 0.90).
    - Moden-Abdeckung allein wirkt NICHT (d=48 tot).
  Offen (T10, registriert): WAS an der Richtung: Phasenmischung in der Eigenbasis
    (MODRAND-Test) oder Bezug zur Koordinaten-Struktur (KOORDKOH-Test)?

## T10 (RICHTUNGS-ZERLEGUNG, registriert VOR dem Lauf)
Vier Arme, alle mit voller Moden-Abdeckung bzw. als Kontrollen, Energie pro
Kanal konstant 0.5, W T6-B60 pro Seed fix (DM exakt konstant), b=0.1*randn
fix, Seeds 100..109 (frisch), f64. Training- und MC-Mass identisch T3/T7/T9.
  REF: Uw = 0.5*randn(H,2) — SANITY (P_lernt >= 0.5, sonst UNENTSCHEIDEN,
    Serie-Post-Mortem, KEINE weitere Paritaets-Reparatur).
  KOH48: kohaerente Moden-Summen (T9-Konstruktion) — Negativ-Kontrolle (erwartet 0).
  MODRAND48: v_c = R[:,subset_c] @ g_c, g_c ~ N(0,1)^{24} zufaellige
    Phasen, Kanaele mod-getrennt — PHASEN-TEST.
  KOORDKOH48: v0 = ones/sqrt(H), v1 = alternierendes Vorzeichenmuster/sqrt(H)
    — kohaerent in Koordinaten, damit PHASEN-GEMISCHT in der Eigenbasis
    (R^T ones ist voll) — BASIS-TEST.
REGISTRIERUNG (GENAGELT):
  PASS (PHASEN-THESE): Sanity UND P_lernt(MODRAND) >= 0.5: Phasenmischung
    bei voller Moden-Abdeckung genuegt zum Lernen; die Eigenmoden-Richtung
    war der Killer. (KOORDKOH dann erwartet lernt; berichtet als sekundaer.)
  KILL-A (BASIS-THESE): Sanity UND P_lernt(KOORDKOH) >= 0.5 ABER
    P_lernt(MODRAND) = 0: Eigenmoden-seitige Phasenmischung genuegt NICHT,
    kohaerente Koordinaten-Richtung schon — der Schluessel liegt im Bezug
    zur Koordinaten-Struktur (Anker/tanh-Schleifen leben in Koordinaten).
  KILL-B (ZERLEGUNG SCHEITERT): MODRAND UND KOORDKOH beide < 0.5: Keine der
    beiden Komponenten reicht — nur iid-Koordinaten-Antriebe (REF-Art)
    lernen. "Genuin diffus" bleibt undifferenziert.
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen, DM-Verletzung.
  SEKUNDAER: MC je Arm (Kontrolle: alle 48-Moden-Arme erwartet ~1.65-1.70;
    wenn MC(MODRAND) ≈ MC(REF) aber Training trennt, ist die Richtung fuer
    MC irrelevant — Dissoziation auch innerhalb des Reservoir-Regimes).
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.


---

# Nachtrag 11 (2026-11-08): T10 — KILL-B. Zerlegung scheitert: Nur iid-Koordinaten-Antriebe lernen. MC bleibt richtungs-robust.

## T10 (REF / KOH48 / MODRAND48 / KOORDKOH48, Seeds 100..109, DM 13.2207 exakt konstant, Lauf 14:28-14:38 UTC)
SANITY: REF lernt P_lernt = 0.70 (7/10 perfekt). Alle drei Zerlegungs-Arme P_lernt = 0.00:
  MODRAND48 (iid Phasen auf je 24 Eigenmoden, Kanaele getrennt) — PHASEN-THESE TOT.
  KOORDKOH48 (ones + alternierend, phasen-gemischt in der Eigenbasis) — BASIS-THESE TOT.
  KOH48 (kohaerent) — Negativ-Kontrolle bestaetigt (0.00 wie T9/T9b).
VERDICT: KILL-B — Zerlegung scheitert. "Genuin diffus" bleibt undifferenziert.

## Was bleibt als Unterschied von REF?
REF ist das einzige Design mit ZWEI UNABHAENGIGEN iid-Gauss-Spalten in Koordinaten
(volle Entropie, beide Kanaele im ganzen R^48, keine Struktur). Die toten Arme
haben je eine Rest-Struktur: Kanal-Trennung (MODRAND: Kanaele auf disjunkte
Moden-Bloecke) oder Richtungsdeterminismus (KOORDKOH: DC + Nyquist, Entropie 0)
oder beides (KOH). Drei Kandidaten-Knarren bleiben: (a) KANAL-OVERLAP (beide
Kanaele tragen dieselben Unterraume), (b) TRAEGER-DIMENSION je Kanal (24 vs 48),
(c) ENTROPIE der Richtungen (iid vs deterministisch). T11 zerlegt REF weiter.

## Sekundaer: MC-Richtungs-Robustheit (dritte Bestaetigung)
MC: REF 1.7606, KOH48 1.7199, MODRAND48 1.7190, KOORDKOH48 1.7257 — Spread < 3%
bei maximal getrennten Trainings-Outcomes (0.70 vs 0.00). MC haengt an der
Moden-Abdeckung (hier bei allen 48), NICHT an der Richtung. Die T9/T9b/T10-
Dissoziation ist dreifach reproduziert: MC = Zahl der angeregten Moden,
Training = Struktur des Antriebs.

## T11 (KANAL-STRUKTUR-ZERLEGUNG, registriert VOR dem Lauf)
Vier Arme, Energie pro Kanal konstant 0.5, W T6-B60 pro Seed fix, b=0.1*randn
fix, Seeds 110..119 (frisch), f64, Masse identisch T3/T7/T10.
  REF: Uw = 0.5*randn(H,2) — SANITY (P_lernt >= 0.5, sonst UNENTSCHEIDEN,
    Serie-Post-Mortem, KEINE weitere Paritaets-Reparatur).
  CHANSEP: REF-Struktur, aber ohne Kanal-Overlap: v0 = Zufall nur in den ersten
    24 Koordinaten, v1 = Zufall nur in den letzten 24 (iid Gauss, je normiert)
    — testet KANAL-OVERLAP als Schluessel.
  CHANOVLP: maximaler Kanal-Overlap mit reduzierter Traeger-Dimension:
    v0, v1 = zwei unabhaengige iid Gauss-Vektoren auf den ERSTEN 24
    Koordinaten (beide Kanaele im selben 24-dim Unterraum)
    — testet TRAEGER-DIMENSION vs. OVERLAP getrennt von CHANSEP.
  HALFRAND: v0 = iid Gauss auf ersten 24 Koordinaten, v1 = iid Gauss auf
    letzten 24 — identisch mit CHANSEP... NEIN: HALFRAND faellt weg,
    CHANSEP deckt es ab. Stattdessen als vierter Arm:
  ENTROP: deterministische, aber "typische" Koordinaten-Richtungen:
    v0 = sign(randn)/sqrt(H) (iid Vorzeichen, Entropie maximal unter
    deterministischer Konstruktion), v1 = unabhaengig ebenso — testet, ob
    iid VORZEICHEN (statt Gauss-Amplituden) genuegen (Entropie-Knarp).
REGISTRIERUNG (GENAGELT):
  SANITY: P_lernt(REF) >= 0.5, sonst UNENTSCHEIDEN (Serie-Post-Mortem).
  PASS (OVERLAP-THESE): Sanity UND P_lernt(CHANOVLP) >= 0.5 UND
    P_lernt(CHANSEP) = 0 — Kanal-Overlap ist der Schluessel, nicht die
    Traeger-Dimension (24 reicht, wenn beide Kanaele denselben Raum teilen).
  KILL-A (TRAEGERTHESE): Sanity UND P_lernt(CHANSEP) >= 0.5 — auch ohne
    Overlap lernbar: Dann war die Traeger-Dimension/Trennung nie der Punkt;
    die zusaetzliche Rest-Variable ist die Amplituden-/Vorzeichen-Verteilung.
    (CHANOVLP dann berichtet als sekundaer.)
  KILL-B (VORZEICHEN-THESE): Sanity UND P_lernt(ENTROP) >= 0.5 UND
    MOD... (hier: CHANSEP und CHANOVLP beide = 0) — iid Vorzeichen genuegen:
    Der Schluessel ist die FEINVERTEILUNG (Entropie) der Antriebsmatrix,
    nicht Geometrie: Eine iid-Gleichverteilung ueber die Einheiten macht
    trainierbar, jede konzentrierte/strukturierte Amplitudenverteilung nicht.
  KILL-C (NICHTS DAVON): Sanity UND alle drei Zerlegungs-Arme < 0.5 — auch
    die Kanal-Struktur-Zerlegung scheitert. Dann ist die verbleibende
    Variable die Gauss-Amplituden-Verteilung selbst (nur REF-artige
    Amplituden-Spektren lernen) — T12 muesste die Amplituden-Verteilung
    drehen (z.B. Randomisierte Amplituden gegen festes Richtungsbild).
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen, DM-Verletzung.
  SEKUNDAER: MC je Arm, median acc, DM, Laufzeit.
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.


---

# Nachtrag 12 (2026-11-08): T11 — UNENTSCHEIDEN durch MEINEN Instrument-Bug (unregistrierte Amplituden-Aenderung am REF). Nebenbefund: Die Antriebs-ENERGIE ist ein massiver eigener Knopf.

## T11 (REF / CHANSEP / CHANOVLP / ENTROP, Seeds 110..119, DM 13.4848 exakt konstant)
Alle 4 Arme P_lernt = 0.00 — auch REF. VERDICT: UNENTSCHEIDEN — Sanity gerissen.
Die T11-Kanal-Struktur-Frage ist damit NICHT beantwortet; registriert: KEINE weitere
Paritaets-Reparatur dieser Linie. T11-Linie tot.

## Ursache — Mea culpa, klares Fehlverhalten gegen die eigene Kernregel
Die T11-Registrierung nannte REF: "Uw = 0.5*randn(H,2)" (die bewaehrte
T6/T9b/T10-Konvention, UNNORMIERT, Spaltennorm ~ 0.5*sqrt(48) ~ 3.46). Ich habe im
T11-Code alle Spalten stillschweigend auf exakte Norm 0.5 normiert ("Energie-
Konvention vereinheitlichen") — ohne die Aenderung zu registrieren oder zu
dokumentieren. Damit hatte REF ~48x weniger Antriebsenergie als jede bewaehrte
Referenz und verhungerte wie jede andere Verhungerungs-Konfiguration. Ein Sanity-
Arm, den man nicht wortgetreu implementiert, misst nichts. Lehre (ergaenzend zur
Freeze-Regel): Nicht nur die THESE ist genagelt — jede Abweichung vom bewaehrten
Referenz-Code im Sanity-Arm ist ein Instrument-Bug, auch wenn sie "sauberer"
aussieht.

## Nebenbefund (ungeplant, Zwei-Punkt, kontrolliert durch T12):
  iid-Gauss-Spalten, Norm 0.5 (T11-REF): P_lernt 0.00.
  iid-Gauss-Spalten, Norm ~3.46 (T9b-REF 0.90, T10-REF 0.70): lernt.
Die Antriebs-ENERGIE ist ein massiver eigener Knopf der Trainierbarkeit — neben
der Richtungs-Struktur (T9b/T10: KILL-C/KILL-B). Wohlgemerkt: Bei T9/T9b/T10
war die Energie pro Kanal als "konstant 0.5" registriert und gemeint war die
kohaerente Konstruktion (Spaltennorm 0.5 dort: 0.5*v mit ||v||=1); REF war
stets unnormiert. Die zwei Energie-Regime lagen also seit T9b NEBENEINANDER im
selben Experiment: REF (E~12) lernte, KOH-d-Familie (E=0.25) verhungerte — ein
Teil des T9b/T10-Kontrasts KOENNTE die Energie sein, nicht nur die Richtung!
Das ist eine ernste, ehrliche Relativierung: Die Richtungs-Story (Nachtraege
10/11) ist mit einem Energie-Konfund belastet. T12 trennt beide Knarren.

## T12 (ENERGIE- UND KANAL-ZERLEGUNG, registriert VOR dem Lauf)
Sechs Arme, W T6-B60 pro Seed fix, b=0.1*randn fix, Seeds 120..129 (frisch),
f64, Masse identisch T3/T7/T10/T11. Spaltennorm-Konventionen GENAGELT:
  REF: 0.5*randn(H,2) UNNORMIERT (bewaehrte Konvention) — SANITY >= 0.5.
  AMP-NEG: REF-Spalten exakt auf Norm 0.5 skaliert (= T11-REF-Replikation) —
    registrierte Negativkontrolle: erwartet P_lernt = 0. Wenn AMP-NEG >= 0.5,
    ist die Energie-Story widerlegt und T11 war ein anderes Artefakt.
  KOH48E: kohaerente Moden-Summen (T9-Konstruktion), aber auf REF-Energie
    hochskaliert (Spaltennorm exakt 0.5*sqrt(48)): trennt, ob die T9/T9b/
    T10-Verhungerung der kohaerenten Arme die RICHTUNG war oder die ENERGIE.
    DAS ist der entscheidende Arm des Laufs.
  CHANSEP-E, CHANOVLP-E, ENTROP-E: die T11-Zerlegungs-Arme, jeweils auf
    Spaltennorm exakt 0.5*sqrt(48) hochskaliert (CHANSEP/CHANOVLP: iid Gauss
    auf getrennten/gemeinsamen 24er-Haelften; ENTROP: iid Vorzeichen / sqrt(H),
    skaliert auf volle Norm).
REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF) >= 0.5, sonst UNENTSCHEIDEN (Serie-Post-Mortem, Ende
    der Kanal-/Energie-Linie).
  AMP-NEG-NEGATIVKONTROLLE (VOR der Primaer-Auswertung, berichtet):
    erwartet P_lernt(AMP-NEG) = 0; wenn >= 0.5: ENERGIE-STORY WIDERLEGT
    (Berichtspflicht, veraendert die Interpretation von KOH48E).
  PRIMAER (P-ENERGIE, die eine These): P_lernt(KOH48E) >= 0.5 — die
    Verhungerung der kohaerenten Arme war die ENERGIE, nicht die Richtung.
    Dann sind Nachtraege 10/11 relativiert: Der Richtungs-Kontrast
    REF-vs-KOH war ein Energie-Konfund (ehrliche Selbstkorrektur).
  KILL-A (RICHTUNGS-STORY HAELT): P_lernt(KOH48E) = 0 UND P_lernt(KOH48) = 0
    (bekannt aus T9/T9b/T10) — die Energie-Substitution rettet die kohaerente
    Richtung nicht: Die Richtungs-Story von Nachtrag 10/11 bleibt stehen,
    jetzt ENERGIE-BEREINIGT. Die T11-Frage (CHANSEP/CHANOVLP/ENTROP-E)
    wird sekundaer berichtet.
  KILL-B (MISCHUNG): P_lernt(KOH48E) in (0, 0.5): Energie verschiebt, Richtung
    moduliert — beide Knarren tragen.
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen, DM-Verletzung.
  SEKUNDAER: P_lernt der CHANSEP/CHANOVLP/ENTROP-E-Arme (T11-Frage bei
    richtiger Energie), MC je Arm, median acc, DM, Laufzeit.
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.
