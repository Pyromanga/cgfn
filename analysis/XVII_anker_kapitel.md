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
(G8) i
st der sweet spot: mehr Kapazitaet ALS Zufall bei geringerer Streuung.

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

Nicht gefunden (also naechste Nachbarn): rotation
s-aquivariante Nichtlinearitaeten
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
Design-Reparatur G10 (einm
alig, wie G4->G5).

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
>= 0.8), P3 JA (
alt-Fehler 0.73 vs neu 0.085, knapp 2x auf NC aggregiert —
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
  BR_RND:  median 0.5215  std 0.2106   (bimodal: 0.
48..1.00)
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
  P1 (Effekt >=0.15): JA — Eff
ekt 0.4785, dreifache Schwelle.
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
Beide Arme R2 ~ 0.000, Effe
kt exakt 0, keine Divergenzen.
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
                Abstand; Sanity-Kl
ausel war defekt und hat das
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
   13.5->0.80, 5.5->0.30, 2.6->0.10
; B60 minimal ueber dem
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
  Der registrierte KILL-Te
xt behauptete "G8 war Entkopplung"
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
Der G8-
Mechanismus-F transferiert nicht nur den Rahmen, sondern
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
     Lotto 
ist eine Traektorien-Eigenschaft. (negativ, echt)
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
MC steigt STRICT MONOTON mit d: 0.7775 (d=2), 0.8976 (4), 1.3535 (8), 1.6526 (16), 1.7047 (48) — bei konstantem DM und konstanter Energie pro Kanal. Im Reservoir-Ridge-Regime ist breite Moden-Abdeckung ALSO GUT: P2 (MC(2) >= 0.90*MC(48)) verfehlt klar (0.78 < 1.53). Erste direkte Evidenz RICHTUNG KILL-B: Diffusion ist auch fuer die Ridge-Ernte gut — der G8-Konzentrations-Vorteil (U=Q[:,0
]) ist im weissen 2-Kanal-Antrieb nicht reproduzierbar.

## Instrument-Diagnose (der eigentliche Ertrag)
T9 vergass den T6-Referenzpunkt im selben Lauf: T6 lernt mit Koordinaten-Zufalls-Antrieb (Uw = 0.5*randn(H,2), beide Kanaele ueber alle 48 Einheiten, Phasen zufaellig) bei P_lernt 0.70-0.80. T9s Antrieb v_c = R[:,subset] @ ones/sqrt ist pro Kanal RANG-1 in die Eigenbasis, Kanaele mod-getrennt (Kanal 0 -> erste, Kanal 1 -> letzte d/2 Moden), Phasen kohaerent. Selbst d=48 ist KEINE Diffusion im T6-Sinn: keine Phasenmischung, kein Kanal-Overlap, keine Kopplung an den Koordinaten-Anker der tanh. Der Knopf drehte an der Moden-ZAHL, aber die Lernbarkeit haengt vermutlich an RICHTUNG/PHASE. Testbar in einem Lauf.

## T9b (Reparatur-Linie, registriert VOR dem Lauf)
Arme: REF (exaktes T6-Instrument: Uw = 0.5*randn(H,2)) als Lernbarkeits-Sanity + die T9-d-Familie UNVERAENDERT (W T6-B60 pro Seed fix, Spektrum fix, b = 0.1*randn fix, Energie pro Kanal 0.5). SANITY: P_lernt(REF) >= 0.5, sonst UNENTSCHEIDEN und Serie-Post-Mortem (KEINE weitere Paritaets-Reparatur). PRIMAER: P1 P_lernt(48)-P_lernt(2) >= 0.3; P2 MC(2) >= 0.90*MC(48). PASS = Sanity UND P1 UND P2 (Doppeldissoziation: Training braucht Moden-Abdeckung, Ridge braucht sie nicht). KILL-A: P_lernt(2) >= 0.5 (Konzentration trainiert, Verhungerungs-Story falsch). KILL-B: P1 UND nicht P2 (auch Reservoir leidet — Diffusion universell; T9-MC-Kurve deutet hierher). KILL-C (NEU, die Phasen-Wette): Sanity JA, P1 NEIN — Lernbarkeit ist KEINE Funktion der Moden-Abdeckung; ist zusaetzlich P_lernt(48) = 0, verhungert JEDE kohaerente Eigenmoden-Summe: Es ist die RICHTUNG, nicht die ZAHL (Verhungerungs-These verfeinert, T8/T8b vollstaendig erklaert). Sekundaer: MC(REF) vs MC(d)-Kurve, DM, Laufzeit. Seeds 90..99. Workflow-Gate erweitert: Import-Check via runpy (run_name != __main__) nach pip, schliesst die py_compile-Blindstelle (fehlende Imports).


---

# Nachtrag 10 (2026-11-08): T9b — KILL-C. Es ist die RICHTUNG, nicht die ZAHL. U
nd: MC ist eine Funktion der Moden-Abdeckung, Training eine der Richtungs-Struktur.

## T9b (REF-Arm + d-Familie, Seeds 90..99, W=T6-B60-Familie fix, DM 13.9842 exakt konstant, Laufzeit 10m47s)
SANITY: REF (exaktes T6-Instrument, Uw = 0.5*randn(H,2)) lernt P_lernt = 0.90, 9/10 perfekt (median acc 1.0000; nur Seed 94 versagt) — Instrument faehig, Vergleich legitim. ALLE fuenf d-Arme P_lernt = 0.00 (Chance 0.48-0.52), 0 Divergenzen, DM-Streuung exakt 0. VERDICT: KILL-C — die Phasen-Wette verloren.

## Der Befund (verfeinerte Verhungerungs-These)
Trainierbarkeit ist KEINE Funktion der Moden-Abdeckung: Selbst d=48 (beide Kanaele auf je 24 Eigenmoden, kohaerente Summen, Energie pro Kanal konstant 0.5) verhungert. JEDE kohaerente Eigenmoden-Summe toetet das Training, jede iid-Koordinaten-Antriebsmatrix (REF) lernt. Es ist die RICHTUNG (Phasenmischung / Basis-Bezug), nicht die Zahl der getroffenen Moden. T8/T8b damit endgueltig erklaert: Der G8-F-Transfer schleppt die Antriebs-RICHTUNG mit, und genau daran stirbt das Training — nicht am Rahmen.

## Die saubere Dissoziation der beiden Variablen (Nebenbefund, sekundaer, aber strukturbildend)
MC-Kurve: strikt monoton in d (0.8519, 0.9012, 1.3462, 1.6743, 1.6973). Und: MC(REF) = 1.6549 liegt AUF der Kurve beim 48-Moden-Wert — der Koordinaten-Zufalls-Antrieb regt in der Eigenbasis saemtliche 48 Moden mit zufaelligen Amplituden an. Damit: MC haengt an der ZAHL (Spektral-Verteilung) der angeregten Moden und ist gegenueber der Richtungs-Struktur ROBUST; Training haengt an der RICHTUNG und ist gegenueber der Moden-Zahl FRAGIL. P2 verfehlt (MC(2) < 0.90*MC(48)): Der G8-Konzentrations-Vorteil ist ein Artefakt des Ein-Moden-Ein-Kanal-Designs, kein Regime-Gesetz. Der Regime-Kontrast von Nachtrag 8 (konzentriert gut fuer Ridge, verteilt gut fuer Training) kollabiert teilweise: Fuer die Ridge-Ernte ist Verteilung ebensogut.

## Regime-Karte v3
  RESERVOIR-RIDGE-REGIME (fixes W, Ridge, MC):
    - MC folgt der Moden-Abdeckung (T9/T9b-Monot
onie), richtungs-robust (REF auf der Kurve).
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
  KILL-B (ZERLEGUN
G SCHEITERT): MODRAND UND KOORDKOH beide < 0.5: Keine der
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
haben je eine Rest-Struktur: Kanal-
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
    deterministischer Konstruktion), v1 = unabhaeng
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


---

# Nachtrag 13 (2026-11-08): T12 — PASS (P-ENERGIE). Die Verhungerungs-Story war die ENERGIE. Groesste Selbstkorrektur der Serie. Literatur-Einordnung. T13-Registrierung (Energie-Kante).

## T12 (REF / AMPNEG / KOH48E / CHANSEP-E / CHANOVLP-E / ENTROP-E, Seeds 120..129, DM 13.3764 exakt konstant)
SANITY: REF 0.80. AMPNEG (E=0.25): 0.00 — Negativkontrolle sitzt exakt. KOH48E
(kohaerente 48-Moden-Konstruktion auf REF-Energie E~12): P_lernt 0.80 — DIE
KOHAERENTE RICHTUNG LERNT. CHANSEP-E 0.70, CHANOVLP-E 1.00, ENTROP-E 0.60:
Bei voller Energie lernen ALLE Geometrien. VERDICT: PASS — P-ENERGIE.

## Was faellt und was bleibt
- GEFALLEN: Die Richtungs-Story von Nachtrag 10/11 (T9b KILL-C, T10 KILL-B) ist
  als Energie-Konfund entlarvt (E=0.25 vs E=12 lief seit T9b ungewo
  nebeneinander). T8/T8b (G8-Transfer, 0/50) sind jetzt Ebenfalls
  energie-verdaechtig (deren Antrieb hatte ebenfalls Spaltennorm 0.5).
- UNANGREIFBAR: (1) MC-Moden-Abdeckungs-Kurve (T9/T9b: Energie dort KONSTANT
  0.25 ueber alle d-Arme — die Monotonie 0.78->1.70 ist echt). (2) MC ist
  energie-robust (T12: AMPNEG-MC 1.67 vs REF-MC 1.69 bei 48x Energie-
  unterschied) UND richtungs-robust (T10: Spread < 3%). MC zaehlt Moden, egal
  wie hart man sie treibt. (3) DM-Knopf (T6), Budget-Kurve (T6),
  Symmetrie-Theorem (T2b), Bias-Lotto (T2b). (4) Neue Karte: TRAINING haengt
  an der Antriebs-ENERGIE (Kante zwischen tot und lernbar, Faktor ~7 in der
  Norm gemessen: 0.5 tot, 3.46 lernt), Geometrie oberhalb der Kante fast egal.

## Mechanismus-Hypothese (testbar, verknuepft mit eigenem T2b-Theorem)
Bei kleiner Antriebsenergie bleibt tanh(z+b) im linearen Regime -> das Netz
ist effektiv linear -> mit linearem Readout ist Paritaet provabel unloesbar
(T2b-Symmetrie-Theorem). Die Energie-Kante ist vermutlich die Kante zwischen
linearem und nichtlinearem Betrieb (tanh-Saettigung ab ~z=1...3).

## Literatur-Einordnung (Recherche 2026-11-08, ehrliche Einbettung)
- Input-Scaling ist bekanntes Folklore im Reservoir-Computing: Lukosevicius
  (Practical Guide to ESNs) nennt Input-Skalierung als wichtigen getunten
  Hyperparameter; Hyperparameter-Studien (Matzner 2022) behandeln input
  scale als wirksamen Knopf.
- Die Echo-State-Eigenschaft ist an die Input-Staerke gekoppelt: Manjunath &
  Jaeger (Neural Computation 25(3), 2013) leiten ein 0-1-Gesetz her, das die
  ESP direkt an statistische Eigenschaften des treibenden Inputs bindet.
- MC haengt an der Input-Skala (Barancok & Farkas, ICANN 2014).
- NICHT gefunden (und damit unser eigener Zusatz): die quantitative
  Dissoziation "scharfe Trainierbarkeits-Kante bei konstantem W, Geometrie
  oberhalb der Kante gleichgueltig (kohaerent vs iid vs Vorzeichen)" und die
  V
  Einordnung: Quantisierung und Verschaerfung bekannten Folklores ("drive
  muss die Nichtlinearitaet wecken"), kein grundsaetzlich neues Gesetz —
  aber die Geometrie-Freiheit oberhalb der Kante ist ein eigenstaendiger
  scharfer Befund.
- Verwandte grosse Literatur-Linie ausserhalb des Reservoir-Kontexts:
  lazy/kernel- vs feature-learning-Regime (muP, Yang et al.): Skalen-
  Kanten zwischen "nichts lernen" und "Struktur lernen" sind auch dort
  zentral. Unsere Kante ist ein dynamisches Pendant auf der Antriebsseite.

## T13 (ENERGIE-KANTE, registriert VOR dem Lauf)
Sieben Arme, W T6-B60 pro Seed fix (DM exakt konstant), b=0.1*randn fix,
Seeds 130..139 (frisch), f64, Masse identisch T3/T7/T12.
  REF: 0.5*randn(H,2) UNNORMIERT — SANITY (P_lernt >= 0.5, sonst
    UNENTSCHEIDEN, Serie-Post-Mortem).
  Sechs Sweep-Arme: Spaltennorm s in {1.0, 2.0, 3.5} x Geometrie
  {REFGEOM (iid Gauss, exakt auf s normiert), KOHGEOM (kohaerente
    48-Moden-Summen, exakt auf s normiert)}.
REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  P1 (KANTE): P_lernt(REFGEOM, s=1.0) = 0 UND P_lernt(REFGEOM, s=3.5) >= 0.5
    (Schwellverhalten in der iid-Familie; Randpunkte aus T12 erwartet).
  P2 (GEOMETRIE-FREI): |P_lernt(KOHGEOM, s) - P_lernt(REFGEOM, s)| <= 0.3
    fuer ALLE s in {1.0, 2.0, 3.5}.
  PASS (ENERGIE-KANTE): Sanity UND P1 UND P2: Eine gemeinsame, scharfe Kante;
    Geometrie verschiebt sie nicht (Energie ist DER Trainings-Knopf).
  KILL-A (2D-KARTE NOETIG): P2 verletzt — die Geometrie verschiebt die Kante.
  KILL-B (KEINE KANTE): P1 verletzt ausserhalb der erwarteten Randpunkte
    (z.B. REFGEOM bei s=3.5 < 0.5, Serie-Inkonsistenz) ODER der Uebergang
    ist weich: P_lernt(s=2.0) fuer BEIDE Geometrien im 0.2-0.5-Zwischen-
    bereich ohne scharfen Sprung — dann ist es eine Kurve, kein Schalter.
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen, D
  SEKUNDAER: MC je Arm (erwartet flach ~1.6-1.8 — MC bleibt energie-robust),
    median acc, DM, Laufzeit.
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.


---

  # Nachtrag 14 (2026-11-08): T13 — PASS (ENERGIE-KANTE). Die Kante liegt zwischen Norm 2.0 und 3.5, ist scharf und geometrie-frei. REF sitzt knapp oberhalb. T14-Registrierung (Mechanismus-Beweis).

  ## T13 (REF + 6 Sweep-Arme s in {1.0, 2.0, 3.5} x {REFGEOM iid, KOHGEOM kohaerent}, Seeds 130..139, DM 13.8110 exakt konstant)
  P_lernt: REFGEOM 0.00 / 0.00 / 0.70; KOHGEOM 0.00 / 0.10 / 0.50; REF (unnnormiert
  ~3.46) 0.50. P1 (Kante) JA, P2 (Geometrie-frei, max Diff 0.20 <= 0.3) JA,
  Sanity JA (knapp). VERDICT: PASS — ENERGIE-KANTE BEWIESEN.

  ## Der Befund
  1. Die Kante ist SCHARF: 0 -> 0.5..0.7 innerhalb Faktor 1.75 in der Norm
     (E=4 -> E=12). Kein gleitender Uebergang, ein Regime-Schalter.
  2. Die Geometrie verschiebt die Kante NICHT (kleiner Trend: kohaerent im
     Uebergang leicht zurueck, aber unter der Registrierungsschwelle 0.3).
  3. REF SELBST SITZT KNAPP OBERHALB DER KANTE (unnnormiert ~3.46, P 0.50 vs
     REFGEOM exakt-normiert 3.5 mit 0.70). Retroaktive Aufklaerung: Die REF-
     Lotterei der Serie (0.50 in T9b/T10... 0.70-0.90) ist Kanten-Balancieren.
  4. MC flach (1.69-1.75) ueber alle Normen: Das Reservoir ist energie-robust,
     das Training lebt von der Energie. Dissoziation dreifach: Energie
     (T12/13), Richtung (T10), Norm-Sweep (T13).

  ## Karte nach T13 (Regime-Karte v4)
    TRAINING (BPTT, Adam, konstantes W-Regime):
      - PRIMAER-KNOPF: Antriebs-ENERGIE mit scharfer Kante zwischen Norm 2.0
        und 3.5 (E=4..12). Oberhalb: lernt (0.5-1.0). Unterhalb: tot (0.0-0.1).
      - Geometrie des Antriebs: oberhalb der Kante fast gleichgueltig
        (iid/kohaerent/Vorzeichen/kohaerent-Moden alle 0.5-1.0 in T12).
      - DM-Knopf (T6) bleibt: Er moduliert INNERHALB des lernbaren Regimes.
    RESERVOIR-RIDGE: MC = Funktion der Moden-Abdeckung (T9/T9b, energie-
      konstant); energie-robust (T12/13) UND richtungs-robust (T10).
    GEFALLEN (endgueltig): Die Richtungs-Verhungerungs-Story (Nachtraege 8/10/
    11) war ein Energie-Konfund. T8/T8b sind energie-verdaechtig.

  ## Mechanismus-Hypothese (seit Nachtrag 13, jetzt entscheidbar)
  Sub-Kante bleibt tanh(z+b) im linearen Regime -> effektives lineares Netz ->
  mit linearem Readout ist Paritaet nicht repraesentierbar (T2b-Linie) -> tot.
  Die Kante = Eintritt ins nichtlineare Regime (z+Betraege muessen ~1..3
  erreichen, tanh-Saettigung ab ~1).

  ## T14 (MECHANISMUS-BEWIES-VERSUCH, registriert VOR dem Lauf)
  Sechs Arme, W T6-B60 pro Seed fix (nur in den ID/RELU-Armen wird die
  AKTIVIERUNG geaendert, W bleibt trainierbar wie immer), b=0.1*randn fix,
  Seeds 140..149 (frisch), f64, Masse identisch T3/T12/T13.
    REF: 0.5*randn(H,2) UNNORMIERT, tanh — SANITY (P_lernt >= 0.5).
    T3.5: REFGEOM exakt Norm 3.5, tanh — Positiv-Kontrolle (erwartet >= 0.5).
    T1.0: REFGEOM exakt Norm 1.0, tanh — Negativ-Kontrolle (erwartet 0).
    ID3.5: Aktiviierung IDENTITY (tanh -> identity), REFGEOM Norm 3.5 —
      KERN-ARM: rein lineares RNN, W trainierbar. Erwartung: 0 bei jeder
      Energie, wenn die Linearitaets-Story stimmt.
    ID10: IDENTITY, REFGEOM Norm 10 — teste, dass selbst brute-force Energie
      ein lineares Netz nicht rettet.
    R3.5: RELU-Aktiviierung (tanh -> relu), REFGEOM Norm 3.5 — nichtlineare
      Kontrolle: ReLU ist nicht-ungerade; wenn die Nichtlinearitaet (nicht
      die tanh-Form) zaehlt, lernt auch R3.5.
  REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
    SANITY: P_lernt(REF) >= 0.5 UND P_lernt(T3.5) >= 0.5 UND
      P_lernt(T1.0) = 0, sonst UNENTSCHEIDEN (Instrument inkonsistent zur
      T12/T13-Basis, Serie-Post-Mortem).
    PASS (P-MECH, die eine These): P_lernt(ID3.5) = 0 UND P_lernt(ID10) = 0
      UND P_lernt(R3.5) >= 0.5: Die Nichtlinearitaet ist NOTWENDIG (Identity
      tot bei jeder Energie) UND die tanh-Form nicht (ReLU lernt) — die
      Kante ist der Eintritt ins nichtlineare Regime. Die Energie-Story
      ist MECHANISTISCH geschlossen: sub-Kante = linear = unloesbar (T2b).
    KILL-A (IDENTITY LERNT): P_lernt(ID3.5) >= 0.3 ODER P_lernt(ID10) >= 0.3:
      Ein lineares RNN loest die Paritaet — die Linearitaets-Story ist
      FALSCH, die Kante ist etwas anderes (z.B. Signal-Rausch-Verhaeltnis).
    KILL-B (RELU TOT): ID-Arme wie erwartet, ABER P_lernt(R3.5) = 0: Nicht-
      linearitaet allein reicht nicht; die Form der Nichtlinearitaet
      (Saettigung, Befaelligung) traegt.
    UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen, DM-Verletzung.
    SEKUNDAER: tanh-Auslastungs-Statistik (Anteil der Zustaende mit
      &#124;z+b&#124; > 1 waehrend der Test-Rolle, gemittelt) je Arm — zeigt, dass die
      Kante mit dem Nichtlinearitaets-Eintritt zusammenfaellt; MC je Arm;
      median acc, DM, Laufzeit.
    Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.


---

# Nachtrag 15 (2026-11-08): T14 — PASS (P-MECH). Die Kette ist mechanistisch geschlossen: Energie -> Nichtlinearitaets-Eintritt -> Lernbarkeit. T15-Registrierung (Skalierungsgesetz der Kante).

## T14 (REF / T3.5 / T1.0 / ID3.5 / ID10 / R3.5, Seeds 140..149, DM 13.9473 exakt konstant)
SANITY-Kette perfekt: REF 1.00, T3.5 0.60, T1.0 0.00 — konsistent mit T12/T13.
KERN: ID3.5 = 0.00, ID10 = 0.00 — das rein lineare RNN lernt die Paritaet bei
KEINER Antriebs-Energie (W trainierbar, 0 Divergenzen). R3.5 = 1.00 — ReLU
lernt perfekt (10/10). VERDICT: PASS — P-MECH.

## Das geschlossene Gesetz der Serie
Die Energie-Kante ist der EINTRITT INS NICHTLINEARE REGIME:
  - Sub-Kante: tanh bleibt linear -> effektives lineares Netz -> Paritaet
    mit linearem Readout nicht loesbar (T2b-Linie) -> tot (ID-Arme beweisen
    den Endpunkt: selbst lineare Dynamik mit brute-force Energi
  - Oberhalb: Nichtlinearitaet nimmt am Gradienten teil -> lernbar,
    GEOMETRIE-ALLES-GLICH (T12: iid/kohaerent/Vorzeichen/kanaele 0.5-1.0).
  - tanh-Form nicht spezifisch (ReLU 1.00).
  - NL-Auslastungs-Diagnostik (Anteil |z+b| > 1): T1.0: 0.000 (tot),
    T3.5: 0.083 (lernt 0.60), REF: 0.231 (lernt 1.00) — die Kante
    korreliert exakt mit dem Nichtlinearitaets-Eintritt. Schon ~8 Prozent
    Auslastung reichen.
DIE SERIE HAT DAMIT IHREN ERSTEN VOLLSTAENDIG MECHANISTISCH GESCHLOSSENEN
BEFUND: Antriebs-Energie (Kante, scharf, geometrie-frei, T12/T13) ->
Nichtlinearitaets-Beteiligung (T14) -> Lernbarkeit. Sub-Kante = linear =
unloesbar ist ein THEORETISCH getragenes Gesetz (T2b), experimentell
bestaetigt.

## Faellige Storys (endgueltige Bilanz der Serie-Korrekturen)
- Richtungs-/Verhungerungs-Story (Nachtraege 8/10/11): Energie-Konfund,
  gefallen in T12. T8/T8b energie-verdaechtig.
- G8-Regime-Kontrast (konzentriert gut fuer Ridge): im 2-Kanal-Regime nicht
  reproduziert (T9/T9b P2 verfehlt); MC ist geometrie- UND energie-robust
  und zaehlt nur Moden-Abdeckung.
- Ueberlebt: Energie-Kante-Gesetz (geschlossen), MC-Moden-Abdeckungs-Kurve
  (T9/T9b, energie-konstant), DM-Knopf (T6, moduliert innerhalb des
  lernbaren Regimes), Budget-Kurve (T6), Symmetrie-Theorem (T2b),
  Bias-Lotto (T2b), All-or-nothing-Optimierungs-Loch (T4b).

## Auffaelliger Zufall: Die Kante liegt bei 0.5*sqrt(H)
T13: Kante zwischen Norm 2.0 und 3.5 bei H=48. Und 0.5*sqrt(48) = 3.46 —
EXAKT die REF-Konvention (0.5*randn(H,2) hat Spaltennorm ~ 0.5*sqrt(H)).
Die bewaehrte Konvention sitzt MOEGLICHERWEISE AUF der Kante — das wuerde
erklaeren, warum REF in der Serie zwischen 0.50 und 1.00 lottert
(Kanten-Balancieren). Wenn die Kante systematisch bei 0.5*sqrt(H) liegt,
gilt das Skalen-Gesetz ||v_c||-Kante = 0.5*sqrt(H) — ein direktes
Design-Rezept (Antriebsnorm >= 0.5*sqrt(H), sonst funktional tot).
T15 macht das entsc

## T15 (KANTEN-SKALIERUNGSGESETZ, registriert VOR dem Lauf)
Acht Arme: H in {24, 96} x REFGEOM-Normen s in {2.0, 3.5, 4.9} plus je ein
unnnormierter REF-Arm (SANITY). W-B60-Familie bei jeweiligem H, b=0.1*randn
fix, Seeds 150..159 (frisch), f64, Masse identisch T3/T12/T13/T14
(Paritaet, P_lernt = Fraktion(acc >= 0.99); MC wie T7 nur sekundaer bei
H=48-Standard). DM-Konstanz wird INNERHALB jedes H geprueft (DM ist
H-abhaengig, nicht arm-abhaengig).
  REF24 / REF96: 0.5*randn(H,2) unnnormiert — SANITY je H (>= 0.5).
  R24_2.0 / R24_3.5 / R24_4.9: H=24, REFGEOM exakt normiert.
  R96_2.0 / R96_3.5 / R96_4.9: H=96, REFGEOM exakt normiert.
REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF24) >= 0.5 UND P_lernt(REF96) >= 0.5, sonst
    UNENTSCHEIDEN (Serie-Post-Mortem, keine weitere Skalierungs-Reparatur).
  PASS (P-SQRT, die eine These: Kante = 0.5*sqrt(H)):
    P_lernt(R24_2.0) = 0 UND P_lernt(R24_3.5) >= 0.5
    UND P_lernt(R96_3.5) = 0 UND P_lernt(R96_4.9) >= 0.5.
    (Kante bei 2.45 fuer H=24, bei 4.90 fuer H=96 — die Vorhersage
    prueft beide Seiten des Gesetzes an beiden H.)
  KILL-A (ABSOLUTE KANTE): P_lernt(R96_3.5) >= 0.5 — die Kante bleibt bei
    ~3.5 auch fuer H=96: absolute Amplituden-Schwelle (pro-Einheit-
    Betrag), kein sqrt(H)-Gesetz.
  KILL-B (KANTE BEI H=24 HOEHER ALS VORHERGESAGT): P_lernt(R24_3.5) = 0
    UND P_lernt(R24_4.9) >= 0.5 (Kante zwischen 3.5 und 4.9 auch bei H=24):
    Die Kante skaliert schwaecher oder gar invers — Skalierungs-Gesetz
    unklar, feiner Sweep noetig.
  SONST (keine der Kanten-Konstellationen): GEMISCHT — Bericht, keine
    der registrierten Formen sitzt; Kanten-Karte bei 2D (H x s) noetig.
  UNENTSCHEIDEN: Sanity gerissen, >3/10 Divergenzen in einem Arm,
    DM-Verletzung INNERHALB eines H.
  SEKUNDAER: median acc, DM je H, Laufzeit.
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.


---

# Nachtrag 16 (2026-11-08): T15 Run-1 — UNENTSCHEIDEN (Crash-Klasse): Instrument-Bug im AUSWERTUNGS-READOUT (REF-Doppelschluessel). Mea culpa. Rerun mit unveraendertem Design gefeuert.

## T15 Run-1 (8 Arme, Seeds 150..159, Lauf 18:32-18:58 UTC, 0 Divergenzen)
VERDICT (Run-1): UNENTSCHEIDEN (Crash-Klasse) — gemeldet: DM-Konstanz
innerhalb eines H verletzt. FEHLALARM aus dem READOUT: stats[] ist nach
Arm-NAMEN geschlossen; REF24 und REF96 teilen den Schluessel 'REF',
REF96 ueberschreibt REF24 (laeuft in ARMS nach H=24). Die Auswertung liest
fuer H=24 DM=20.6678 (den H=96-Wert) und rennt gegen die R24-Arme mit
DM=7.5793 -> scheinbare Verletzung. Die ARM-Sektionen sind sauber:
REF24 DM=7.5793 exakt wie alle R24-Arme, REF96 DM=20.6678 exakt wie alle
R96-Arme — DM-Konstanz INNERHALB jedes H sitzt per Konstruktion. Der
Doppeleintrag war im Vorfeld als kleine Unsauberheit notiert und hat
zugeschlagen. Lehre von T11 in der zwei
bewaehrten Referenz-Muster ist ein Instrument-Bug — hier nicht im
Sanity-Arm, sondern im Auswertungs-Readout.

## Run-1-Arm-Daten (gueltige Messungen; deterministische Seeds -> im Rerun identisch erwartet)
  REF24: P_lernt 0.80 | REF96: P_lernt 1.00 — SANITY je H sitzt.
  H=24: s=2.0: 0.70, s=3.5: 0.90, s=4.9: 1.00.
  H=96: s=2.0: 0.10, s=3.5: 0.60, s=4.9: 0.80.
  0 Divergenzen in allen 8 Armen; DM je H exakt konstant.
Vorblick (kein Verdikt, nur Lesart der Messungen): Der Uebergang ist an
H=24/96 eine RAMPE, kein Schalter (H=24 lernt schon bei s=2.0 mit 0.70 —
die scharfe T13-Kante bei H=48 repliziert sich NICHT als scharfe Null
unterhalb 0.5*sqrt(H)). Aber die Ordnung P(H=24) > P(H=96) bei JEDEM
festen s (0.70/0.10, 0.90/0.60, 1.00/0.80) traegt die sqrt(H)-RICHTUNG:
hoeheres H braucht mehr Norm. Formales Verdikt erst in Nachtrag 17 nach
dem Rerun (registrierte Kriterien, keine Nachverhandlung).

## Rerun (registriert: Design UNVERAENDERT, nur Readout-Reparatur)
Mess-Arme, Seeds, Masse, Registrierung und VERDICT-Logik unangetastet.
Einzige Aenderungen: REF-Arme werden getrennt verschluesselt (REF24/
REF96); der redundante s_ref24-Patz entfernt. Deterministische Seeds ->
identische Arm-Zahlen erwartet, nur der Readout korrigiert. Bewahrung:
Rerun nur bei Crash — Crash-Klasse lag vor (Instrument-Bug).


---

# Nachtrag 17 (2026-11-08): T15 Rerun — VERDICT: KILL-A (formal). P-SQRT gefallen. Ehrliche Lesart: Weder scharfes Skalen-Gesetz noch absolute Kante — der Uebergang ist eine RAMPE, und die registrierten binaeren Kriterien fassen sie nicht.

## Rerun (Run-2, Start 19:04 UTC, fertig 19:31 UTC, Readout repariert)
Arm-Zahlen identisch zu Run-1 (deterministische Seeds), Readout korrekt:
SANITY REF24=0.80, REF96=1.00 — beide sitzen. DM-Konstanz innerhalb beider
H OK (7.5793 / 20.6678), 0 Divergenzen in allen 8 Armen.
Formaler Durchgang: PASS nein (R24_2.0=0.70 statt 0; R96_3.5=0.60 statt 0),
KILL-B nein, KILL-A-Bedingung R96_3.5 >= 0.5 erfuellt (0.60).
VERDICT: KILL-A. Freeze: kein Rerun, keine Nachverhandlung am Etikett.

## Warum das KILL-A-Etikett di
Der KILL-A-Text behauptet eine absolute Kante ~3.5 unabhaengig von H.
Die Daten widersprechen dem in BEIDE Richtungen:
  H=24: s=2.0 lernt schon 0.70 — weit unter 3.5.
  H=96: s=4.9 nur 0.80 — kein saettigendes All-or-nothing oberhalb 3.5.
Die Kriterien waren auf eine SCHARFE Kante gebaut (T13-Format bei H=48:
0.00 -> 0.70 in einem Norm-Schritt). An H=24/96 ist der Uebergang eine
RAMPE: P_lernt steigt glatt mit s, und die ganze Rampe verschiebt sich
mit H. Registrierungs-Lektion: Ein binaeres Kanten-Kriterium (tot vs
lernt) ist gegen Rampen blind — die Vorblick-Warnung aus Nachtrag 16 war
berechtigt, das formale Etikett ist streng genommen ueber-keywordet.

## Das Gesetz-Material (berichtet, kein neues Verdikt)
1. RICHTUNG: P(H=24) > P(H=96) bei jedem festen s (0.70/0.10, 0.90/0.60,
   1.00/0.80) — groessere Netze brauchen mehr Antriebsnorm. Aber die
   Ordnung ist nicht sauber ueber alle H: bei s=2.0 liegt H=48 aus T13
   mit 0.00 ZWISCHEN H=24 (0.70) und H=96 (0.10) — Rauschen bei n=10 oder
   echte Nicht-Monotonie, unentscheidbar mit dem vorhandenen Material.
2. KEIN KOLLAPS: Weder s/sqrt(H) noch s/H legt die drei H-Kurven
   aufeinander (H=24 lernt bei skaliert 0.41 schon 0.70; H=96 bei
   skaliert 0.50 erst 0.80). Ein einfaches Ein-Parameter-Skalierungs-
   Gesetz ist mit drei H nicht im Material — die 2D-Karte (H x s) ist
   der naechste Stein, NEU zu registrieren.
3. T13-REFRAMING: Die "scharfe Kante" bei H=48 war vermutlich ein
   Rampen-Punkt mit Zufallsrest (nur zwei Rasterpunkte unterhalb, 10
   Seeds). REF (unnnormiert ~3.46) sass am Rampenanstieg — die REF-
   Lotterei der Serie (0.50-1.00) ist MITTLERE-RAMPEN-Streuung, kein
   Kanten-Balancieren eines scharfen Gesetzes. Die T15-Daten stuetzen
   das: REF24 (~2.45) sitzt auf der Rampe (0.80), REF96 (~4.9) am
   oberen Ende (1.00).
4. Design-Rezept (pragmatisch, konservativ): Antriebsnorm >= ~1.0*sqrt(
   wahlen (oberhalb der Rampe), niemals unter ~2 (dort kein verlaessliches
   Lernen bei keinem getesteten H). Das 0.5*sqrt(H)-Rezept aus Nachtrag
   15 ist als SCHARFES Gesetz gefallen; als grobe untere Orientierung
   bleibt es brauchbar, aber mit Rampe, nicht mit Kante.

## Serie-Stand: Die Energie-Kette bleibt, verliert die Schaerfe
Die geschlossene Kette (T12/T13/T14: Energie -> NL-Eintritt -> Lernbarkeit,
geometrie-frei) ist unangetastet — sub-Kante linear/tot, oberhalb lernbar.
Neu durch T15: Der Uebergang ist breiter und H-abhaengiger als bei H=48
geschaetzt; "scharf" war eine Ueberschaetzung aus zwei Rasterpunkten.
Ueberlebt: Energie-Gesetz (Rampe, Richtung mit H), MC-Moden-Kurve, DM-
Knopf, Budget-Kurve, Symmetrie-Theorem (T2b), Bias-Lotto (T2b),
All-or-nothing (T4b), Mechanismus-Beweis (T14). Gefallen: P-SQRT als
scharfes Skalen-Gesetz 0.5*sqrt(H).


---

# Nachtrag 18 (2026-11-08): W1-REGISTRIERUNG — LAB-KERNEL, der Experiment-Compiler (Werkzeug-Stein, P0 Design)

## Warum ein Werkzeug-Stein (die Kostenrechnung)
Zwei Laeufe des heutigen Tages (T11 einst, T15 Run-1 heute) sind an der
Disziplin-Luecke zwischen Registrierung und Code verbrannt — zusammen
~52 Runner-Minuten, das volle Tagesbudget. Beide Bug-Klassen
(handgeschriebene Abweichung vom bewaehrten Referenz-Arm; handgeschriebene
Auswertung abseits der eigenen Registrierung) sind strukturell gleich: Es
gab zwei Quellen der Wahrheit. W1 eliminiert die zweite.

## W1 (LAB-KERNEL, registriert VOR dem Bau — Design in analysis/W1_lab_kernel.md)
THESE: Registrierung, Lauf, Auswertung und Verdikt aus EINER deklarativen
Quelle generiert -> die Bug-Klasse wird unmoeglich, nicht bloss unwahrscheinlich.
Fuenf Module, drei Phasen: C-Contract (Generator), A-Invarianten-Audit
(misst Konventionen zur Laufzeit statt ihnen zu vertrauen), D-Determinismus-
Diff (Rerun-Diff gegen letzten Result-Commit), K-Kollaps-Gate (Pre-Flight:
unpruefbare Begriffe, doppelte Keys, unregistrierte Masse), N-Auto-Nachtrag
(Zahlen-Skelett mit Lueckenliste). P0 (dieser Stein): nur Design. P1: C+A,
Abnahme AUSSCHLIESSLICH am Historien-Test (known-answer):
  W1-PASS: (a) T15-Contract reproduziert Run-2 exakt (KILL-A, Commit 4d9bd4b),
  (b) historische T11-Abweichung wird als Audit-Alarm angezeigt,
  (c) T15-Run-1-Doppelkey wird verweigert/angemeckert.
  W1-KILL: nur mit Hardcodes erreichbar -> Contract ist Schaum; oder (b)/(c)
  unpruefbar -> These falsch.
  W1-UNENTSCHEIDEN: T15 nicht reproduzierbar; EINE Reparatur, dann tot.
Bewahrung: Nach Abnahme keine Nachjustierung; erste Anwendung auf dem
Stone nach T15. Kosten: 0 Runner-Minuten (Offline-Historien-Test).


---

# Nachtrag 19 (2026-11-08): W1-P1 — PASS. Der Contract-Kernel besteht den Historien-Test: T15 exakt wiedergegeben, T11-Bug angezeigt, T15-Run1-Bug verweigert.

## W1-P1 (tools/lab_kernel.py, Lauf 22:29-22:32 UTC, reines Python, keine Runner-Minuten verbrannt nach Gate-Abbruch)
VERDICT: W1-PASS. (a) Der Kernel generiert aus dem T15-Contract die
Auswertung: KILL-A — ABSOLUTE KANTE, alle acht Arm-Zahlen identisch mit
Commit 4d9bd4b. (b) Der T11-Contract (REF deklariert unnnormiert ~3.46) plus
die historische Messung (0.5) schlaegt als INVARIANTEN-VERLETZUNG Alarm —
die stille Normierung von T11 haette diesen Kernel nie passiert. (c) Die
T15-Run-1-Struktur (doppelte REF-Keys) wird mit VERWEIGERT abgewiesen.
Die Wette von W1 gilt als gehalten: Verdikt und Auswertung aus EINER
deklarativen Quelle, die zwei historischen Bu
toetbar.

## Bug-Protokoll des Werkzeug-Steins (ehrlich)
- Run-1 (12s): Schreib-Artefakt — beim Push wurden \n-Zeichen im
  String-Literal als echte Zeilenumbrueche aufgeloest (unbeendeter
  String). Die py_compile/runpy-Gates fingen es ab Null-Kosten. Lehre:
  Der Push-Weg ist selbst ein Instrument und braucht einen Selbst-Audit
  (der jetzt als Vor-Push-Probe eingebaut ist).
- Run-2 (laufend): _eval_pred rief einen bereits berechneten bool auf
  (TypeError). Das war DIE EINE registrierte Reparatur der W1-Linie
  (Abschnitt 4: UNENTSCHEIDEN -> EINE Reparatur, dann tot). Run-3 bestand.
- Ironie-Bilanz: Der Bug-Fang-Kernel brauchte drei Versuche, weil ihn
  zwei Bugs genau der Klassen trafen, gegen die er gebaut ist. Aber:
  Beide wurden von Gates und Logbuch gefangen, keines ist still durch-
  gegangen. Das System funktioniert — auch dort, wo es gegen mich
  arbeitet.

## Architektur-Schuld (registriert fuer P2, vor erster Kernel-Anwendung einzuloesen)
P1 ist absichtlich Ein-Datei-Minimalbeweis: Engine, Contracts, Fixtures
und Test-Runner liegen in tools/lab_kernel.py zusammen. Sauber ist nur die
Trennung: Engine (tools/lab_kernel.py), Contracts (contracts/, reine
Daten), Historien-Fixtures + Runner (tests/). Diese Bereinigung ist der
erste P2-Schritt, BEVOR der Kernel ein echtes Experiment generiert —
sonst bauen wir die Zwei-Quellen-Versuchung in die Ordnerstruktur
zurueck.

## Naechste Schritte (registriert in W1, Abschnitt 3)
P2: Architektur-Split (Engine/Contracts/Tests), D-Determinismus-Diff
(Rerun-Diff gegen letzten Result-Commit), K-Kollaps-Gate (Pre-Flight),
N-Auto-Nachtrag (Zahlen-Skelett mit Lueckenliste). Danach: erste echte
Kernel-Anwendung auf dem naechsten Physik-Stein.


---

# Nachtrag 20 (2026-11-08): W1-P2-REGISTRIERUNG — Vollausbau D/K/N, Architektur-Schuld einzuloesen VOR jeder Kernel-Anwendung (genagelt VOR dem Lauf)

## P2-Umfang (registriert, KEINE NACHVERHANDLUNG)
1. ARCHITEKTUR-SPLIT (die Schuld aus Nachtrag 19): Engine ->
   tools/lab_kernel.py; Contracts -> contracts/ (t15.py, t11.py, reine
   Daten); Historien-Fixtures + Runner -> tests/ (fixtures.py,
   run_historien_test.py). Die P1-Regression a/b/c bleibt Teil der
   Abnahme — jetzt gegen die getrennten Bausteine.
2. D — DETERMINISMUS-DIFF: determinismus_diff(rerun, commit). Leer bei
   identischem Rerun; JEDE Abweichung ein Flag (Umwelt-Bug-Verdacht:
   Runner-Image, torch-Version). Geflaggt wird, nicht verwurstet.
3. K — KOLLAPS-GATE, zwei Ebenen:
   preflight_gate(contract) verweigert den Start bei (i) doppelten
   Arm-Keys (T15-Run1-Klasse), (ii) Praedikaten auf unbekannte Arme,
   (iii) Praedikaten auf UNREGISTRIERTE Masse — die operationalisierte
   T15-"scharf"-Klasse: Eine Registrierung, die eine Messung verlangt,
   die kein Gate liefern kann, startet nicht.
   Laufzeit: evaluate schlaegt NICHT-GEMESSEN-Alarm, wenn eine
   registrierte Groesse im Mess-Input fehlt — kein stilles Loch.
   EHRLICHE GRENZE (registriert): Freitext-Begriffe ohne messbare
   Operationalisierung prueft keine Maschine. K prueft die
   datenfassbare Variante; der Rest bleibt Menschenpflicht.
4. N — AUTO-NACHTRAG: auto_nachtrag(contract, result) — Zahlen-Skelett
   mit Verdikt und LUECKENLISTE (welche registrierte Groesse im Output
   fehlt). XVII-Urteile (Lesart, Selbsteinordnung) bleiben
   handgeschrieben; die Zahlen nicht.

## P2-Abnahme (genagelt, alles Historien-/Known-Answer-Test, 0 neue Physik)
  P2-PASS: Regression (a) T15-Wiedergabe KILL-A mit identischen Zahlen,
  (b) T11-stille-Normierung -> Audit-Alarm, (c) T15-Run1-Doppelkey ->
  VERWEIGERT — UND neu:
  (d) D-DIFF: diff(Run2, Run2) LEER; diff(Run2, Run2-mutiert) flaggt
      GENAU die mutierte Arm-Groesse (REF24.pf).
  (k1) K-PRE-FLIGHT: Contract mit Praedikat auf unregistriertes Mass
      ("schaerfe") -> VERWEIGERT mit UNREGISTRIERT-Alarm.
  (k2) K-LAUFZEIT: Mess-Input ohne einen Arm -> NICHT-GEMESSEN-Alarm,
      Verdikt UNENTSCHEIDEN (Crash-Klasse), kein stiller Durchlauf.
  (n) N-SKELETT: T15-Result -> Skelett mit KILL-A, allen acht Arm-Zahlen
      (z.B. R96_3.5.pf = 0.6) und LUECKENLISTE: KEINE; unvollstaendiger
      Result -> LUECKENLISTE nennt die fehlende Groesse (R96_4.9.pf).
  P2-KILL: Ein Modul nur mit Hardcodes/test-spezifischen Sonderfaellen
      erreichbar — dann ist der Contract-Schaum-Vorwurf zurueck.
  P2-UNENTSCHEIDEN: Ein Test faellt; EINE Reparatur erlaubt, dann
      P2-Linie tot (wie P1).
  Bewahrung: kein Rerun ohne Crash; Kostenziel 0 Runner-Minuten netto
  (reines Python, Sekunden, Kontingent unberuehrt).


---

# Nachtrag 21 (2026-11-08): W1-P2 — PASS (im ERSTEN Versuch). Vollausbau D/K/N abgenommen, Architektur-Schuld eingeloest.

## W1-P2 (Lauf 22:44-22:45 UTC, Bot-Commit 7af90d2, reines Python, Sekunden)
VERDICT: W1-P2-PASS — alle sieben Tests bestanden:
  REGRESSION (P1, gegen die getrennten Bausteine):
  (a) T15-Wiedergabe: KILL-A — ABSOLUTE KANTE, alle acht Arm-Zahlen
      identisch mit Commit 4d9bd4b.
  (b) T11-stille-Normierung: INVARIANTEN-VERLETZUNG-Alarm
      (gemessen 0.5, deklariert ~3.46).
  (c) T15-Run1-Doppelkey: VERWEIGERT (plus zwei zusaetzliche
      UNBEKANNTER-ARM-Alarme durch das schaerfere K-Gate).
  NEU (P2):
  (d) D-DIFF: identisch -> 0 Flags; mutiert -> genau das geflaggte
      REF24.pf (Rerun 0.7 vs Commit 0.8).
  (k1) K-PRE-FLIGHT: Praedikat auf unregistriertes Mass ("schaerfe")
      -> VERWEIGERT (die operationalisierte T15-"scharf"-Klasse —
      genau dieser Bug-Typ machte T15-PASS unmoeglich).
  (k2) K-LAUFZEIT: fehlender Arm -> fuenf Alarme, Verdikt
      UNENTSCHEIDEN (Crash-Klasse), kein stilles Loch.
  (n) N-SKELETT: KILL-A + alle acht Arm-Zahlen + LUECKENLISTE: KEINE;
      bei fehlendem Arm nennt die LUECKENLISTE die fehlende Groesse.

## Protokoll (ehrlich)
Erster Versuch, null Reparaturen, null verbrannte Runner-Minuten.
Der Vor-Push-Selbst-Audit (Umlaut-Probe + \n-Literal-Probe aus der
W1-Run-1-Lehre) lief als harter Gate vor dem Push — und fing beim
Nachtrag-21-Push zweimal einen Umlaut im eigenen Protokoll-Text,
bevor er das Repo erreichte (Null-Kosten, Korrektur im Sandbox-
Auditorium). Die Lehre aus dem eigenen Bug-Protokoll zahlt sich aus.
Die Architektur-Schuld aus Nachtrag 19 ist EINGELOEST, bevor der
Kernel erstmals scharf schiesst: Engine (tools/lab_kernel.py),
Contracts (contracts/, reine Daten), Fixtures + Runner (tests/).

## Stand der W-Linie
W1 ist damit VOLLSTAENDIG (P0 Design, P1 C+A, P2 D/K/N + Split).
Alle fuenf Module des registrierten Designs sind gebaut und am
Historien-Test abgenommen. Der Kernel ist
 SCHARF: Die naechste
registrierte These (z.B. T16, 2D-Karte H x s) laeuft erstmals durch
den Contract-Generator — Verdikt generiert, Audit misst, K-Gate
verweigert vor dem Start, D-Diff gegen den letzten Result-Commit,
N schreibt das Zahlen-Skelett. Danach P3 (XVII als abfragbare
Wissensstruktur) und P4 (These-Generator).


---

# Nachtrag 22 (2026-11-08): T16-REGISTRIERUNG — 2D-KARTE (H x s), ERSTER KERNEL-GENERIERTER STEIN (genagelt VOR dem Lauf)

## Warum T16 (und nicht P3/P4 zuerst)
Der W1-Kernel ist abgenommen (Nachtrag 21) und im Echtbetrieb UNGE-
TESTET. Der erste scharfe Schuss muss ein Physik-Stein mit bekanntem
Instrument sein (T15-Wortlaut), nicht eine neue Werkzeug-Ebene auf
ungetestetem Werkzeug. T16 ist der registrierte Folge-Stein aus
Nachtrag 17 ("2D-Karte H x s, NEU zu registrieren").

## T16-THESE (P-RAMPE-SHIFT-SUBH, die eine These)
Die Lernbarkeits-Rampe P(s|H) verschiebt sich mit H — die Rampen-Mitte
s50 waechst mit H —, ABER sub-proportional zu H:
ratio50 = s50(H=96)/s50(H=24) liegt strikt in (1, 4) (4 = H-Verhaeltnis).

## Architektur (Kernel-Anwendung, genagelt)
Contract: contracts/t16.py (reine Daten — Arme, Seeds, s-Raster,
Task-Konstanten, Invarianten, Sanity, Verdikt-Logik). Instrument:
experiments/t16_2d_karte.py (Konstruktion, Training, Selbst-Messung —
liest Arme/Seeds/Raster/Konstanten AUSSCHLIESSLICH aus dem Contract).
VERDICT generiert tools/lab_kernel.py (W1-P2). K-Pre-Flight vor dem
ersten Trainingsschritt (Runner-Minuten bleiben bei Verweigerung
unberuehrt). Neu nach A-Standard: Der Lauf MISST selbst Spaltennorm
(T11-Klasse), dmdev (DM-Konstanz innerhalb H, Gruppen-Relativ-
abweichung), ndiv, s50, ratio50.

ARME (14): REF24, REF96 (unnnormierte Konvention, SANITY je H) +
R24_s / R96_s fuer s in {1.5, 2.0, 2.8, 3.5, 4.4, 5.5} (REFGEOM,
exakt normiert). Seeds 160..169 (frisch). H in {24, 96}. Paritaet
wortgetreu wie T15 (Delayed-XOR, T=24, RK2, tau=1.0, dt=0.1,
substeps=5, Adam 600, Batch 32, lr 5e-3, 1024 Test-Samples, Test-Seed
777, FIXED_P0=4242, WB=0.6, LAM_MAX=0.6, b=0.1*randn fix).
P_lernt = Fraktion(acc >= 0.99).

s50-DEFINITION (genagelt): Erste 0.5-Durchquerung der nach s
sortierten P_lernt-Kurve der NORMIERTEN Arme eines H, linear
interpoliert zwischen Rasterpunkten. Divergenz-Arme (pf NaN) zaehlen
als nicht gelernt (0 in der Kurve). Nicht
 einklammerbar (pf am
unteren Rand >= 0.5 oder nirgends >= 0.5) -> s50 NICHT GEMESSEN ->
Kernel-Kollaps-Alarm -> UNENTSCHEIDEN (Aufloesung), EINE Reparatur
erlaubt, dann T16-Linie tot.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG — der Contract IST
diese Registrierung in ausfuehrbarer Form):
  SANITY: P_lernt(REF24) >= 0.5 UND P_lernt(REF96) >= 0.5.
  PASS (P-RAMPE-SHIFT-SUBH): 1.0 < ratio50 < 4.0 — Rampe verschiebt
    sich mit H, sub-proportional zu H.
  KILL-A (KEIN/INVERSER SHIFT): ratio50 <= 1.0 — die T15-Richtung
    (groessere Netze brauchen mehr Norm) ist auf Rampen-Ebene falsch.
  KILL-B (H-PROPORTIONAL): ratio50 >= 4.0 — schlimmster Fall: Rezepte
    muessen mit H skalieren; sqrt-artige Regeln reichen nicht aus.
  UNENTSCHEIDEN (Crash-Klasse): s50 nicht einklammerbar, Sanity
    gerissen, ndiv > 3 in einem Arm, dmdev-Verletzung (DM-Konstanz
    innerhalb H) oder colnorm-Verletzung (Normierung fehlgeschlagen).
    Rerun nur bei Crash; EINE Reparatur.
  SEKUNDAER: median acc, DM je H, s50/sqrt(H) je H (sqrt-Kollaps-
    Frage nur BERICHT, kein Verdikt — dafuer ist das Raster zu grob),
    Laufzeit, D-Diff-Snapshot (results/t16_measurements.json, ma-
    schinenlesbar) fuer kuenftige Reruns (erster Lauf ohne Vorgaenger).
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.
  KOSTEN-SCHAETZUNG: ~45-55 Runner-Minuten (14 Arme x 10 Seeds).

## Ehrliche Risiken (vor dem Lauf genannt)
1. pf(1.5|H=24) koennte >= 0.5 sein (T15: pf(2.0|24)=0.70; T13:
   pf(1.0|48)=0.00) -> s50 nicht einklammerbar -> UNENTSCHEIDEN.
   Risiko akzeptiert; die EINE Reparatur waere ein versetztes/
   feineres Raster.
2. n=10 Seeds je Arm: pf-Rauschen +-0.15. Genau deshalb liegen die
   registrierten Schwellen weit auseinander (1.0 und 4.0).
3. 6-Punkt-Interpolation ist grob: ratio50 ist eine GROBE Schaetzung;
   registriert ist der weite Bereich (1, 4), keine genaue Zahl.


---

# Nachtrag 23 (2026-11-08): T16 — PASS (P-RAMPE-SHIFT-SUBH), erster kernel-generierter Stein. Die Rampe verschiebt sich mit H, sub-proportional zu H. Und: Der Kernel im Echtbetrieb — sauber.

## T16 (Lauf 22:50-23:23 UTC, ~33 min, Bot-Commit 731933f, Seeds 160..169, 14 Arme, 0 Divergenzen)
VERDICT (GENERIERT vom Kernel): PASS — P-RAMPE-SHIFT-SUBH.
  ratio50 = s50(96)/s50(24) = 3.5/2.0 = 1.75 — strikt in (1, 4).
  RAMPE H=24: pf = 0.20 / 0.50 / 0.80 / 1.00 / 1.00 / 0.90
       (s = 1.5 / 2.0 / 2.8 / 3.5 / 4.4 / 5.5)
  RAMPE H=96: pf = 0.00 / 0.10 / 0.30 / 0.50 / 0.80 / 0.90
  Beide Kurven erstmals der Serie: SAUBER MONOTON (bis auf n=10-
  Rauschen an der Saettigung), keine Kante, kein Kliff — die
  T15-Rampen-Lesart ist voll bestaetigt.

## Lesart (handgeschrieben, wie es das N-Modul verlangt)
1. DIE EINE THESE HAELT: Die Rampe verschiebt sich mit H
   (2.0 -> 3.5), sub-proportional zu H (1.75 << 4). KILL-B
   (H-proportional) ist weit weg; KILL-A (kein Shift) auch.
2. SPAERER BEFUND (nur Material, nicht registriert): ratio50 = 1.75
   liegt sogar UNTER sqrt(4) = 2 — s50/sqrt(H) FAELLT mit H
   (0.408 -> 0.357). Vorsicht: s50 sitzt exakt auf Rasterpunkten
   (pf ist k/10 diskret, 0.50-Rasterzier), das echte s50 liegt
   je einen halben Rasterpunkt unsicher. Moegliche Ordnung:
   Der Shift ist SUB-SQRT — groessere Netze brauchen weniger
   sqrt-skalierte Norm als vermutet. Wenn das haelt, ist das
   Design-Rezept GUENSTIGER als gedacht: 0.5*sqrt(H) liegt
   OBERHALB der Rampe bei beiden H (0.41/0.36 gemessen).
   Naechster Stein dafuer: H=192-Punkt (kostet ~8 min) als
   Extrapolations-Test der s50(H)-Kurve.
3. DESIGN-REZEPT (Stand T16, konservativ): Antriebsnorm >=
   1.0*sqrt(H) bleibt die sichere Wahl; die Rampe liegt bei
   ~0.4*sqrt(H) (Mitte), Ref-10%-Untergrenze unverletzt.
4. T15-KONSISTENZ: R96_3.5 pf 0.50 (T15: 0.60, andere Seeds/W-
   Ziehung) — im Rauschen. REF-Lotterei: REF24 0.90, REF96 1.00.

## Kernel im Echtbetrieb (der eigentliche Werkzeug-Be
fund)
- K-PRE-FLIGHT: OK vor dem ersten Trainingsschritt (0 verbrannte
  Minuten bei Verweigerung — dieses Mal nicht noetig).
- SELBST-MESSUNG: colnorm exakt (max Abweichung 0.0), dmdev ~1e-16,
  ndiv=0 ueberall. Die T11-Blindstelle ist geschlossen: haette das
  Instrument still normiert, haette der Audit-Alarm geschlagen.
- VERDICT generiert, LUECKENLISTE: KEINE, N-Skelett maschinell.
- BUG-PROTOKOLL: LEER. Erster Stein der Serie ohne jeglichen Bug-
  Eintrag, null Reparaturen, Laufzeit 33 min (unter der 45-55-
  Schaetzung). D-Diff-Snapshot liegt (results/t16_measurements.json)
  fuer den Crash-Fall.
- Ehrliche Einordnung: Ein sauberer Lauf ist EIN Datenpunkt, kein
  Beweis. Aber der Stein lief end-to-end (Contract -> Pre-Flight ->
  Selbst-Messung -> generiertes Verdikt -> Skelett) ohne jede
  menschliche Auswertungshand.

## Serie-Stand nach T16
Energie-Gesetz-Kette komplett: Energie -> NL-Eintritt -> Lernbarkeit
(T12/T13/T14), Uebergang = RAMPE (T15/T16), Rampen-Mitte s50(H)
steigt mit H, sub-proportional (T16), groessenordnungsmassig
s50 ~ 0.4*sqrt(H) im getesteten H-Bereich {24, 96}.
Offen als naechster Stein: s50(H)-Kurve um H=192 erweitern
(Extrapolationstest der Sub-Sqrt-Ordnung) — NEU zu registrieren.
Danach P3 (XVII als abfragbare Wissensstruktur), P4 (These-
Generator).


---

# Nachtrag 24 (2026-11-08): T17-REGISTRIERUNG — s50-EXTRAPOLATION BEI H=192 (genagelt VOR dem Lauf). Plus: Nutzer-These zur Research-Taste als offene Design-Frage registriert.

## Nutzer-These (registriert als offene Design-Frage, nicht als Beschuss-These)
Der Kapitaen haelt "Research Taste bleibt beim Menschen" fuer den
wahrscheinlichen FLASCHENHALS, nicht fuer ein Feature (Widerlegung
meiner Einordnung aus der Recon-Diskussion). Registriert fuer P4/L5:
TASTE AUTOMATISIEREN ist ein spaeter eigener Stein (Generator-Regeln
fuer "was ist eine wertvolle Frage"), nicht ein Tabu. Die Recon
unterstuetzt die Sorge: Die Literatur nennt research taste als eine
der drei offenen Kernluecken autonomer Wissenschaft. Vorbehalt
meinerseits, ehrlich: Taste ohne Verlust-Verdikt-Signal ist nicht
trainierbar — genau das liefert aber unser Stack ab P3. Offen.

## T17-THESE (P-SUBSQRT-EXTRAPOLATION, die eine These)
  Die s50(H)-Kurve wachst SUB-SQRT WEITER: ratio2 = s50(192)/s50(96)
  liegt strikt in (1.0, 1.4142) (1.4142 = sqrt(2), das H-Verhaeltnis
  2 waere sqrt-Skalierung). Basis: T16 s50(24)=2.0, s50(96)=3.5
  (ratio50=1.75 < sqrt(4)=2, Commit 731933f).

## INSTRUMENT (wortgetreu T16, Kernel-generiert wie T16)
  Contract contracts/t17.py (reine Daten), Instrument experiments/
  t17_s50_extrapolation.py, Verdikt generiert. H=192 (neu), Seeds
  170..179 (frisch), Paritaet wie T15/T16. ARME (6): REF192
  (unnnormiert, SANITY) + R192_s fuer s in {2.8, 3.5, 4.4, 5.5, 7.0}
  (REFGEOM exakt normiert). s50-Definition wie T16 genagelt (erste
  0.5-Durchquerung, linear interpoliert; nicht einklammerbar ->
  NICHT GEMESSEN -> Kernel-Alarm -> UNENTSCHEIDEN, EINE Reparatur).
  ratio2 = s50(192)/3.5 mit T16-Anker s50(96)=3.5 als registrierte
  Konstante im Contract.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF192) >= 0.5.
  PASS (P-SUBSQRT-EXTRAPOLATION): 1.0 < ratio2 < 1.4142.
  KILL-A (SQRT-ARTIG ODER STAERKER): ratio2 >= 1.4142 — die Sub-Sqrt-
    Ordnung von
 T16 extrapoliert nicht; H-Skalierung naeher an sqrt(H).
  KILL-B (KEIN WEITERER SHIFT): ratio2 <= 1.0 — die Rampe stoppt.
  UNENTSCHEIDEN (Crash): s50 nicht einklammerbar (Raster verfehlt die
    Rampe: alles < 0.5 oder unterster Rand >= 0.5), Sanity gerissen,
    ndiv > 3, dmdev- oder colnorm-Verletzung. Rerun nur bei Crash;
    EINE Reparatur, dann T17-Linie tot.
  SEKUNDAER: median acc, DM, s50/sqrt(192), Laufzeit, D-Diff-Snapshot.
  Bewahrung: kein Arm wird nachjustiert.
  KOSTEN-SCHAETZUNG: ~40-55 Runner-Minuten (H=192 ist teuer: 6 Arme x
    10 Seeds; Ehrlichkeit gegenueber der frueheren "~10 min"-Schaetzung
    des Kapitaens-Chat: das war zu optimistisch, H skaliert quadratisch
    im Rollout).
  RISIKEN (vor dem Lauf): (1) s50(192) > 7.0 -> Rampe verlaesst das
  Raster nach oben -> UNENTSCHEIDEN (Richtung KILL-A, unmesbar).
  (2) s50(192) < 2.8 -> ebenso nach unten (Richtung KILL-B). (3)
  n=10-Diskretisierung: s50 unsicher +-1 Rasterpunkt, die Schwellen
  1.0/1.4142 liegen dicht — dieses Mal bewusst scharf, denn es ist
  ein EXTRAPOLATIONSTEST, kein breites Karten-Gate.


---

# Nachtrag 25 (2026-11-09): T17 — PASS (P-SUBSQRT-EXTRAPOLATION). Drei s50-Punkte, ein Kandidaten-Gesetz: s50 ~ 0.55 * H^0.41. Der Theory-Core hat sein erstes Material.

## T17 (Lauf 23:43-00:06 UTC, 23 min, Bot-Commit 7e6da59, Seeds 170..179, 6 Arme, 0 Divergenzen, Bug-Protokoll leer)
VERDICT (GENERIERT): PASS — P-SUBSQRT-EXTRAPOLATION.
  s50(192) = 4.675 | Anker s50(96) = 3.5 (T16) | ratio2 = 1.3357,
  strikt in (1.0, 1.4142) — knapp unter sqrt(2), Sub-Sqrt haelt.
  RAMPE H=192: pf = 0.00 / 0.30 / 0.40 / 0.80 / 0.80
       (s = 2.8 / 3.5 / 4.4 / 5.5 / 7.0)
  SANITY REF192: 0.90. colnorm exakt, dmdev ~1e-16, ndiv = 0.

## Die drei s50-Punkte (T16/T17, konsistent gemessen)
  H= 24: s50 = 2.000  (T16)
  H= 96: s50 = 3.500  (T16)
  H=192: s50 = 4.675  (T17)

## Kandidaten-Gesetz (MATERIAL, unregistriert — der Theory-Core-Fit)
Log-Log-Fit ueber alle drei Punkte:
  s50(H) ~ 0.547 * H^0.408
  Probe: H=24 -> 2.00 (exakt), H=96 -> 3.52 (vs 3.50), H=192 ->
  4.67 (vs 4.675). Alle drei Punkte innerhalb 1 Prozent.
Der EXPONENT ~0.41 ist stabil ueber beide H-Schritte (0.40 im
ersten, 0.42 im zweiten Doppel-Schritt) — kein Drift. s50/sqrt(H)
faellt weiter (0.408 / 0.357 / 0.337), konsistent mit Exponent <
0.5. EHRliche Grenzen: (1) drei Punkte, n=10-Seeds, s50-Raster-
unsicherheit +-1 halber Rasterpunkt; (2) nur EINE Aufgabenfamilie
(Paritaet), EINE W-Familie (T6-B60), EINE Antriebsgeometrie
(REFGEOM); (3) der Fit ist deskriptiv, KEIN registriertes Verdikt.

## Was das Gesetz bedeutet (wenn es haelt)
- Design-Rezept GUENSTIGER als jede fruehere Version: Antriebsnorm
  >= ~1.5 * s50(H) ~ 0.8 * H^0.41 waechst deutlich langsamer als
  sqrt(H) — grosse Netze sind ANTRIEBS-BILLIGER als die sqrt-Regel
  behauptete.
- Die NL-Eintritts-Kette (T14) bekommt eine quantitative Spitze:
  Der benoetigte Antrieb waechst sub-thermisch mit der Breite.

## DER NAECCHSTE SCHRITT: der erste THEORY-CORE-PRUEFSTEIN (T18)
Der Stack bekommt seinen ersten echten Vorhersage-Test: Das gefittete
Ge
setz sagt EINE Zahl voraus, ein neuer Stein prueft sie am Kernel.
  T18 (registriert VOR dem Lauf, NEU): H=12 (billigste Breite, keine
  Messung existiert). VORHERSAGE aus dem Fit: s50(12) ~ 0.547 *
  12^0.408 ~ 1.51. Design: REF12 + R12_s-Raster um die Vorhersage
  (s in {0.8, 1.1, 1.5, 1.9, 2.4}), Seeds 180..189. Registriertes
  Verdikt: PASS wenn das interpolierte s50(12) in (1.1, 1.9) liegt
  (Vorhersage +-25 Prozent, grosszuegig gegen n=10-Rauschen und
  Rastergroebheit); KILL wenn ausserhalb — dann faellt das
  Potenz-Gesetz als Einzelgesetz und der Fit war Zufall.
  Kosten: H=12 ist der billigste Stein der Serie (~10-15 min).
Das ist der geschlossene Kreislauf in Miniatur: Weltmodell (Fit)
sagt voraus -> Contract -> Lauf -> generiertes Verdikt -> Graph.
GENAuso wird Research-Taste messbar (Nutzer-These, Nachtrag 24):
Jede Vorhersage des Modells ist ein Trainingsbeispiel.

## Kernel-Protokoll (Echtbetrieb, Stein 2 von 2)
Zweiter kernel-generierter Stein, wieder: K-Pre-Flight OK, Selbst-
Messung sauber, Verdikt generiert, LUECKENLISTE: KEINE, Bug-
Protokoll: LEER, Laufzeit 23 min (unter der 40-55-Schaetzung).
Echtbetriebs-Bilanz des Kernels: 2 Steine, 0 Bugs, 0 Reparaturen.


---

# Nachtrag 26 (2026-11-09): EXPONENTEN-ANALYSE (ehrlich: KEIN exaktes Pinning aus 3 Punkten) + T18-REGISTRIERUNG — Theory-Core-Vorhersage-Stein (genagelt VOR dem Lauf)

## Exponenten-Analyse (Antwort auf die Kapitaens-Frage)
Der 1%-Fit s50 ~ 0.546 * H^0.4083 ist real, ABER: Die Raster-
unsicherheit (n=10, pf=k/10, s50 +- ~0.25) blaest den Exponenten auf
das Band [0.382, 0.433]. Darin leben ALLE Kandidaten:
  2/5 = 0.400 | ln(3/2) = 0.405 | 1/sqrt(6) = 0.408 (0.02% Abweichung
  — VERDAECHTIG exakt, aber Numerologie-Gefahr: 3 Punkte pinnen keine
  Konstante) | 5/12 = 0.417. KEIN Kandidat ist ausschliessbar; <1%-
  Abweichung beweist hier KEINE genaue Formel, nur eine gute
  Naeherung. Schritt-Exponenten (0.404 / 0.418) zeigen keinen Drift.
ROBUST ueber alle Kandidaten: die Vorhersage s50(12) in [1.47, 1.52]
(e=0.40: 1.523 | e=0.408: 1.498 | e=5/12: 1.471).
PINNING-PLAN (registriert): Wenn T18 PASS -> T19 als Pinning-Stein
(H in {12, 48} mit feinem Raster und 20 Seeds an den Klammer-
punkten; Ziel: Exponenten-Band +-0.01) ODER analytische Herleitung
(Warum H^alpha mit alpha ~ 0.41? Kandidat-Mechanismus: DM/Spektrum-
Kopplung der T6-B60-Familie). VERALLGEMEINERUNG (offen, eigene
Steine): Das Gesetz ist bisher an EINER Familie gemessen (Paritaet,
T6-B60-W, REFGEOM); allgemein gueltig heisst: Familie variieren.

## T18 — THE-CORE-VORHERSAGE-TEST (P-GESCHLOSSENE-VORHERSAGE)
ERSTER VORHERSAGE-STEIN der Serie: Das gefittete Weltmodell (Nachtrag
25) sagt eine Zahl voraus; der Kernel verdiktet sie. Das ist der
geschlossene Kreislauf in Miniatur und der erste messbare
Research-Taste-Datenpunkt (Nutzer-These, Nachtrag 24).

VORHERSAGE (aus s50 ~ 0.546 * H^0.408, Nachtrag 25):
  s50(H=12) ~ 1.51 (Kandidaten-Band [1.47, 1.52]).

INSTRUMENT (kernel-generiert wie T16/T17): Contract contracts/t18.py,
Instrument experiments/t18_core_vorhersage.py, Verdikt generiert.
H=12 (neu, billigste Breite), Seeds 180..189, Paritaet wie T15/T16/
T17. ARME (7): REF12 (unnnormierte Konvention
, role ref — Bericht,
KEIN Sanity: 0.5*sqrt(12)=1.73 liegt VORHERGESAGT auf der Rampe —
T3-Lehre: kein Sanity-Arm auf einem marginalen Punkt), HIGH12
(REFGEOM exakt Norm 3.5, role anchor — SANITY: weit oberhalb der
Vorhersage-Rampe, dort muss das Instrument lernen), R12_s fuer s in
{0.8, 1.1, 1.5, 1.9, 2.4} (role sgrid, REFGEOM exakt normiert).

s50-Definition wie T16/T17 genagelt (erste 0.5-Durchquerung der
sgrid-Kurve, linear interpoliert; nicht einklammerbar -> NICHT
GEMESSEN -> Kernel-Alarm -> UNENTSCHEIDEN, EINE Reparatur).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(HIGH12) >= 0.5.
  PASS (P-GESCHLOSSENE-VORHERSAGE): s50(12) in [1.1, 1.9]
    (Vorhersage 1.51, Band +-27 Prozent — grosszuegig gegen n=10-
    Rauschen, scharf genug, um einen kaputten Fit zu toeten).
  KILL-A (VORHERSAGE ZU HOCH): s50(12) < 1.1 — das Netz lernt frueher
    als das Gesetz sagt; Fit war Zufall oder H^alpha falsch geformt.
  KILL-B (VORHERSAGE ZU NIEDRIG): s50(12) >= 1.9 — das Netz lernt
    spaeter; dito.
  UNENTSCHEIDEN (Crash): s50 nicht einklammerbar (Raster verfehlt),
    Sanity gerissen, ndiv > 3, dmdev-/colnorm-Verletzung. Rerun nur
    bei Crash; EINE Reparatur, dann T18-Linie tot.
  SEKUNDAER: REF12-pf (Konventions-Bericht), DM, Laufzeit,
    D-Diff-Snapshot, implizierter Exponent aus 4 Punkten (Bericht).
  Bewahrung: kein Arm wird nachjustiert.
  KOSTEN-SCHAETZUNG: ~10-15 Runner-Minuten (H=12 ist schnell).
  RISIKEN: pf(0.8) >= 0.5 -> s50 < 0.8 -> UNENTSCHEIDEN (haette
  KILL-A-Richtung); pf(2.4) < 0.5 -> UNENTSCHEIDEN (KILL-B-Richtung).


---

# Nachtrag 27 (2026-11-09): T18 — PASS formal, aber das reine Potenz-Gesetz KNICKT. Der Theory-Core lernt aus seinem ersten Vorhersage-Test: 14 Prozent daneben heisst nicht "bestaetigt", es heisst "fast".

## T18 (Lauf 08:03-08:20 UTC, 17 min, Bot-Commit b6cd8f4, Seeds 180..189, 7 Arme, 0 Divergenzen, Bug-Protokoll leer)
VERDICT (GENERIERT): PASS — P-GESCHLOSSENE-VORHERSAGE (s50(12) = 1.30
in [1.1, 1.9]). ABER die scharfe Vorhersage war 1.51 — das Messergebnis
liegt 14 Prozent DARUNTER, nah an der Bandunterkante. Formal PASS,
inhaltlich: Das 3-Punkt-Gesetz s50 ~ 0.546*H^0.408 ist als GLOBALES
Ein-Parameter-Gesetz GEFAELLEN.

## Die vier Punkte jetzt (T16/T17/T18)
  H= 12: s50 = 1.300 (T18)   H= 24: s50 = 2.000 (T16)
  H= 96: s50 = 3.500 (T16)   H=192: s50 = 4.675 (T17)
SCHRITT-EXPONENTEN: 12->24: 0.6215 | 24->96: 0.4037 | 96->192: 0.4176.
Der Exponent ist NICHT konstant: Die Kurve ist am kleinen H STEILER
und flacht auf ~0.41 ein. Ein Ein-Exponent war von Anfang an zu
einfach geglaubt (drei Punkte), jetzt ist es widerlegt (vier Punkte).

## Refit ueber alle vier Punkte (Material)
  (a) Reiner Potenz-Fit: s50 ~ 0.447 * H^0.450 — Residuen bis
      +-6.5 Prozent (H=24: -6.5, H=12: +5.3). Kein <1%-Gesetz.
  (b) AFFINE Form: s50 ~ 0.564 * H^0.41 - 0.176 — Residuen bis
      +-6.7 Prozent. Praktisch aequivalent zu (a) im Bereich
      [12, 192] (beide sagen z.B. H=48 bei ~2.55 voraus).
  EHRLEICHES FAZIT: Beide Formen treffen die vier Punkte auf ~5-6
  Prozent. Als VORHERSAGE-Werkzeug ist der Theory-Core brauchbar
  (Vorhersageband +-15 Prozent waere ehrlich); als EXAKTES Gesetz
  ist nichts gepinnt. Die <1%-Frage des Kapitaens (Nachtrag 26)
  ist damit negativ beantwortet: Es gibt (noch) keine genaue
  Formel, nur eine gute Naeherung mit Biegung am unteren H-Rand.

## Was der Theory-Core aus T18 gelernt hat (der eigentliche Ertrag)
1. DAS MODELL HAT EINEN VERLUST-SIGNAL-DATENPUNKT: Vorhersage 1.51,
   gemessen 1.30, Fehler 14 Prozent. Erste Zeile des kuenftigen
   Research-Taste-Trainings (Nutzer-These, Nachtrag 24): Ein Fit aus
   3 Punkten ist zu optimistisch, wenn er keine
   Unsicherheits-Schaetzung traegt. Lehre: kuenftige Vorhersagen mit
   explizitem Band UND mit Modellvergleich gegen Alternativ-Form.
2. KANDIDAT-KOVARIATE ENTDECKT (Material): DM wachst mit H (4.84 bei
   H=12 bis 29.43 bei H=192). Moeglicherweise ist s50 eine Funktion
   von DM (oder beides folgen dem Spektrum), nicht von H direkt —
   das wuerde die Biegung erklaeren und ist der analytische Angriffs-
   punkt fuer die WARUM-Frage.
3. Die Sanity-Architektur (HIGH12-Anker statt REF12-auf-der-Rampe,
   T3-Lehre) hat funktioniert: Sanity 0.60, sauber getrennt vom
   Rampen-Bericht.

## T19-REGISTRIERUNG (VORHERSAGE, genagelt VOR dem Lauf) — DER ZWEITE VORHERSAGE-TEST, diesals MIT BAND UND MODELLVERGLEICH
Design: H=48 (die Luecke in der Kurve, billig). s-Raster um die
beiden Form-Vorhersagen (beide ~2.55): R48_s fuer s in {1.9, 2.2,
2.55, 2.9, 3.3} (role sgrid), HIGH48 (s=5.5, Sanity-Anker), REF48
(Konventions-Bericht, role ref). Seeds 190..209 (20 Seeds — doppelt,
denn T18s Lehre ist Raster/Rauschen-empfindlichkeit). f64, Paritaet
wie immer, kernel-generiert (contracts/t19.py).
VORHERSAGEN (registriert, beide Formen aus Nachtrag 27):
  s50(48) ~ 2.55 (reiner Potenz-Fit: 2.55 | affin: 2.58 — die Formen
  unterscheiden sich hier praktisch nicht; der Test ist gegen die
  MESSUNG, nicht gegen eine Form gegen die andere).
REGISTRIERUNG (GENAGELT):
  SANITY: P_lernt(HIGH48) >= 0.5.
  PASS (P-CORE-VORHERSAGE-2): s50(48) in [2.05, 3.05] (+-20 Prozent
    um 2.55, breiter als T18 gelernt hat: 14%-Fehler-Plus-Rauschen).
  KILL-A (ZU KLEIN): s50(48) < 2.05 — beide Formen zu optimistisch,
    Kurve biegt staerker.
  KILL-B (ZU GROSS): s50(48) >= 3.05 — Kurve steiler als beide.
  UNENTSCHEIDEN (Crash): Raster verfehlt (alles < 0.5 oder Rand >=
    0.5), Sanity gerissen, ndiv > 6 von 20, dmdev-/colnorm-Verletzung.
    Rerun nur bei Crash; EINE Reparatur.
  SEKUNDAER: 4-Punkt-Refit mit H=48 (Bericht), REF48-pf, DM, Laufzeit.
  Bewahrung: kein Arm nachjustiert. KOSTEN: ~15-25 min (H=48,
    8 Arme x 20 Seeds).
  RISIKO (ehrlich): REF48 (0.5*sqrt(48)~3.46) sitzt wieder auf der
  Rampe — deshalb ist Sanity auf HIGH48, REF48 nur Bericht.


---

# Nachtrag 28 (2026-11-09): T19 UNENTSCHEIDEN (Crash — Raster verfehlt). Die eine erlaubte Reparatur: Raster nach UNTEN. Und ein Physik-Alarm, der wichtiger ist als das Verdikt.

## T19 Lauf 1 (gefeuert 16:51 UTC, Ergebnisse 17:21 UTC, ~30 min, Bot-Commit fe0f84e, Seeds 190..209, 7 Arme, 0 Divergenzen)
VERDICT (GENERIERT): UNENTSCHEIDEN (Crash-Klasse). Registriertes
Crash-Kriterium "Raster verfehlt" traf EXAKT zu: Die sgrid-Kurve
s=1.9:0.55, 2.2:0.75, 2.55:0.85, 2.9:0.90, 3.3:0.95 liegt beim
untersten Rasterpunkt bereits UEBER 0.5 — s50(48) sitzt UNTERHALB
des Rasters, s50_from_curve -> None, LUECKENLISTE voll (s50 als
Arm-Mass nie geliefert). Kein KILL, kein PASS: unentscheidbar.

## PHYSIK-ALARM (Vorzeichen, kein Verdikt): Moegliche Nicht-Monotonie
P_lert(s=1.9, H=48) = 0.55, fast auf der 0.5-Linie. Falls s50(48)
wirklich < 1.9, dann gilt s50(48) < 2.00 = s50(24) — die s50-Kurve
faellt zwischen H=24 und H=48. Das wuerde nicht nur KILL-A (Vor-
hersage 2.55 zu optimistisch) bedeuten, sondern das monophone
Potenz-Gesetz GANZ fallen lassen. Bisherige Reihe: 12:1.30,
24:2.00, 96:3.50, 192:4.675. Ehrlich: Rerun-1 kann auch Seed-Bimodalitaet
oder Raster-Quantisierung (20 Seeds -> 0.05-Schritte) uebertreiben —
genau deshalb R2 mit dichterem unteren Raster.

## Ursachen-Analyse (ehrlich, Bau-Fehler-Klasse: Prediction-Lock-in)
Das Raster wurde UM DIE VORHERSAGE 2.55 gelegt. Das ist ein
Design-Vorurteil: Wer das Raster auf seine eigene Vorhersage zentriert,
misst nur deren Bestaetigung und blendet das Gegenteil aus. T18
hatte genau davor gewarnt (14 Prozent daneben). Die Contract-Struktur
(s50 als Arm-Mass, Instrument setzt es kurvenweit) war intakt —
allein die Raster-Platzierung war falsch. Lehre fuer den Kernel:
Raser-Wahl ist Teil der Registrierung und gehoert GEGEN die
Vorhersame robustifiziert (Randpunkte beidseitig weit ausserhalb
des erwarteten Bands).

## REPARATUR R2 (die EINE erlaubte; danach ist die Linie tot)
NUR das Raster geaendert, alles andere GENAGELT wie Nachtrag 27:
  s-grid: {0.7, 0.9, 1.1, 1.3, 1.55, 1.9, 2.2} (7 Punkte, dicht unten,
    Obergrenze 2.2 reicht: P(2.2)=0.75 bekannt).
  Seeds 190..209 (20), H=48, f64, REF48/HIGH48 unveraendert.
  SANITY: P_lernt(HIGH48) >= 0.5.
  PASS: s50(48) in [2.05, 3.05) | KILL-A: < 2.05 | KILL-B: >= 3.05.
  UNENTSCHEIDEN: Raster erneut verfehlt (s50 < 0.7), Sanity gerissen,
    ndiv > 6 von 20, dmdev-/colnorm-Verletzung — DANN LINIE TOT.
  KEINE Nachverhandlung der Schwellen.
VORHERSAGE FUER R2 (Band, kein Verdikt, aus Lauf-1-Material):
  s50(48) in [1.2, 1.9] — falls real: KILL-A und das Potenz-Gesetz
  verliert die Monotonie. Falls s50 in [2.05, 3.05): PASS (dann war
  Lauf 1 ein reines Raster-Artefakt — unwahrscheinlich angesichts
  P(1.9)=0.55).
KOSTEN: ~40 min (9 Arme x 20 Seeds, H=48).


---

# Nachtrag 29 (2026-11-09): T19-R2 KILL-A. Die s50-Kurve hat ein EINBRUCH bei H=48. Das monophone Potenz-Gesetz in H ist GEFALLEN — und genau das waehlt den naechsten Stein: DM-Kovariate.

## T19-R2 Lauf 4 (gefeuert 20:47 UTC, Ergebnisse 21:26 UTC, ~39 min, Bot-Commit 43f4056, Seeds 190..209, 9 Arme, 0 Divergenzen, LUECKENLISTE LEER)
VERDICT (GENERIERT): KILL-A — ZU KLEIN. s50(48) = 1.83 < 2.05.
R2-Band-Vermutung [1.2, 1.9] aus Nachtrag 28: GETROFFEN (1.83).
Vorhersage 2.55 lag 39 Prozent daneben. P-CORE-VORHERSAGE-2 tot.

## Der BEFUND: Nicht-Monotonie
s50-Reihe komplett: 12:1.30, 24:2.00, 48:1.83, 96:3.50, 192:4.675.
s50(48) < s50(24): Die Kurve FAELLT zwischen 24 und 48 und steigt
dann steil (48->96 Exponent 0.9355, der hoechste der Serie).
Schritt-Exponenten gesamt: 12->24: 0.6215, 24->48: -0.1282,
48->96: 0.9355, 96->192: 0.4176. KEIN konstanter Exponent existiert,
die Schwankung ist nicht Rauschen um einen Mittelsatz, sondern
SYSTEMATISCH: erst hoch, dann negativ, dann fast 1, dann halb.

## Ehrliche Lesart (zwei Lagen, beide offen)
1. FALL DELLT-RECHT: Der Einbruch ist real. Dann ist H NICHT die
   kausale Variable — das Gesetz ist in etwas anderem verankert,
   das mit H nicht monoton waechst. Kandidat: DM (Nachtrag 27:
   DM waechst 4.84 -> 13.21 -> 29.43, aber moeglicherweise nicht
   glatt). Die Schritt-Exponenten-Schwingung riecht nach einer
   verkehrten x-Achse.
2. FALL SAATGUT-RAUSCHEN: Die Delle ist klein (2.00 -> 1.83 sind
   8.5 Prozent) und P ist auf 0.05 quantisiert (20 Seeds). T16
   mass s50(24)=2.00 mit EIGENEM Raster. Ob die Delle die
   Seed-Summen ueberlebt, ist offen — aber KILL-A ist FORMAL
   unangreifbar: Die Vorhersage 2.55 war auf KEINEM Blatt richtig.

## Konsequenz: T20 = DM-KOVARIATE (Nutzer-Entscheidung, direkt nach R2)
Warum dieser Stein jetzt: Er unterscheidet die zwei Faelle NICHT
durch noch eine Kurve in H (das waere die dritte Seed-Summe auf
derselben Achse), sondern durch KAUSAL-TEST: DM bei FESTEM H=48
variieren (Konjugation der Eigenbasis — Spektrum identisch,
Diagonalmasse anders) und s50 messen.
  - Wenn s50 auf DM hoert: H war nie die Variable; Delle erklaert
    sich aus DM(H)-Verlauf; WARUM-Frage hat einen Anker. KILL.
  - Wenn s50 bei DM-Variation bei ~1.83 bleibt: DM ist Kovariate
    ohne Kausalitaet; H/Spektrum behaelt die Rolle. Dann ist die
    Delle entweder Rauschen (Fall 2) oder verlangt einen dritten
    Kandidaten.
REGISTRIERUNG als Nachtrag 30, Design dort genagelt. EINE Reparatur
falls Crash, dann Linie tot.

## Kernel-Bilanz nach dem Stein (Echtbetrieb)
5 kernel-generierte Steine (T15..T19R2), 2 saubere Verdikte in Serie,
1 korrekt erkannter UNENTSCHEIDEN-Crash, 1 Reparaturlauf sauber,
0 Runner-Minuten auf kaputten Contracts verbrannt (Syntax-Gate +
Wrapper-Guard). Die Engine war in KEINEM einzigen Lauf der
Flaschenhals — jedes Mal war es Bau oder Design. Das ist die
L1-Aussage in Echtbetrieb: abgenommen.


---

# Nachtrag 30 (2026-11-09): T20-REGISTRIERUNG — DM-KOVARIATE (Kausal-Test). Nutzer-Entscheidung direkt nach T19-R2-KILL-A (Nachtrag 29).

## DIESE (P-DM-KAUSAL)
s50 haengt an DM (Diagonalmasse), nicht an H. Wenn DM bei festem
H=48 variiert wird — Spektrum identisch via Basis-Konjugation —
verschiebt sich s50 ueber die Stufen hinweg messbar.

## DESIGN (GENAGELT)
H=48 fix. W' = Q_a W Q_a^T, Q_a aus QR(I + a*G), G fix (Seed 2020),
a in {0, 1, 4, 12} — vier DM-Stufen; a=0 ist Identitaet
(Referenzstufe = T19-R2-Material, s50=1.83 erwartbar).
Pro Stufe s-Raster {1.3, 1.55, 1.83, 2.2} (dicht um 1.83).
16 sgrid-Arme + REF48 (Bericht) + HIGH48 (Sanity, s=5.5).
Seeds 190..209 (20), f64, Paritaet wie immer.

## VERDICT-MASS
s50spread = max(s50_stufe) - min(s50_stufe), an jedem Arm als
registriertes Mass 's50spread'.
  PASS — DM-KAUSAL:  s50spread >= 0.2 (s50 hoert auf DM).
  KILL — DM-NEBENSPUR: s50spread < 0.2 (s50 bleibt bei ~1.83,
    DM ist Kovariate ohne Kausalitaet — H/Spektrum behaelt Rolle).
  UNENTSCHEIDEN (Crash): DM-Bereich zu schmal
    ((max DM - min DM)/mean DM < 10 Prozent ueber Stufen),
    Stufen-Raster verfehlt (Stufen-s50 nicht interpolierbar),
    Sanity gerissen, ndiv > 6 von 20, colnorm-Verletzung.
    EINE Reparatur, dann Linie tot.

## SEKUNDAER (Bericht)
DM pro Stufe, s50 pro Stufe, Stufen-Kurven, implizite DM(H)-Reihe.

## KOSTEN
18 Arme x 20 Seeds, H=48 ≈ 70-90 min (Timeout 150).

## RISIKO (ehrlich)
Der erreichbare DM-Bereich der Konjugation ist vorher NICHT
bekannt. Falls a=12 DM nicht um >= 10 Prozent bewegt: Crash als
UNENTSCHEIDEN, NICHT als KILL — ein schmaler Bereich beweist nichts.


---

# Nachtrag 31 (2026-11-10): T20 UNENTSCHEIDEN (Stufen-Raster verfehlt) — aber das schwerste Sekundaer-Signal der Serie. R2: WB-Treppe statt Konjugation, zensierungsfreies Mass. Und: Die Delle bleibt unerklaert, auch gegen DM.

## T20 Lauf 1 (gefeuert 07:21 UTC, Ergebnisse 09:07 UTC, ~106 min, Bot-Commit bfb1f59, 18 Arme, 0 Divergenzen)
VERDICT (GENERIERT): UNENTSCHEIDEN — registriertes Kriterium
"Stufen-Raster verfehlt". Nur die Identitaetsstufe (a=0) liefert
ein interpolierbares s50=1.89 (reproduziert T19-R2 1.83 auf 3
Prozent — Rasterdifferenz). Die Konjugationsstufen kollabieren:
DM faellt 13.21 -> 1.40/1.34/1.58 (Faktor ~9, Spektrum identisch),
P_lernt(2.2) bricht 0.75 -> 0.05/0.15/0.05 ein, s50 dort nicht
interpolierbar -> LUECKENLISTE voll.

## DAS SEKUNDAER-SIGNAL (kein Verdikt, aber der Geruch)
KEIN Kontinuum: Lern-Kollaps fast auf Null bei DM-Kollaps.
Lesart (Geruch): DM ist kein Kovariat, sondern ein ORDNUNGSPARAMETER.
Diagonallastige Dynamik = H entkoppelte Kanaele, eine Mode ein
Drehknopf, Energie gebuendelt, NL-Eintritt gelingt. Gemischte
Basis = jede Parameter-Aenderung streut ueber alle Moden, Energie
verschmiert, NL-Eintritt verfehlt. Das ist das fehlende Glied der
Energie-Kette: DM entscheidet, WOHIN die Eingangsenergie geht.

## GERICHT: Die naive "falsche-x-Achse"-Version ist WIDERLEGT
DM(H) komplett (D-Diff-Snapshots, 0 Runner-Minuten): 12:4.84,
24:8.29, 48:13.21, 96:20.55, 192:29.43. s50 gegen DM aufgetragen:
1.30, 2.00, 1.83, 3.50, 4.675 — die Delle sitzt bei DM=13.21
GENAUSO. DM als blosse Ersatzachse loest die Delle NICHT auf.
Sie bleibt unerklaert: Seed-Rauschen (8.5 Prozent, ein 20er-Satz
je H) oder zweites Phaenomen. Ehrlich: offen.

## R2-REGISTRIERUNG (die EINE Reparatur von T20; danach Linie tot)
Konjugation war ein Schalter, kein Regler (QR einer Zufallsmatrix
gibt immer denselben gemischten DM-Bereich — Treppe wurde Kante;
Bau-Fehlerklasse: Instrument-Dial-Blindheit). R2 nimmt den GLATTEN
Regler: WB in eigenbasis() steuert die Diagonallast direkt
(th = WB*rand; WB->0 macht R zur Identitaet, DM->1).
  DESIGN (GENAGELT): H=48 fix, Spektrum fix (genS unveraendert),
  WB-Stufen {0.6, 0.45, 0.3, 0.15}, je Stufe s-Raster
  {1.83, 2.2, 2.9} (T20-a=0-Kurve: 1.83:0.45, 2.2:0.75 — Crossing
  liegt prognostiziert im Raster). 12 Stufen-Arme + HIGH48
  (Sanity, s=5.5, WB=0.6). KEIN REF48 (identisch zur WB0.6-Stufe
  bereits gemessen — Ressourcen schonend, Nutzer-Auflage).
  Seeds 190..209 (20), f64.
  PRIMAER (VERDIKT): pfspread220 = max-min von P_lernt(s=2.2)
  ueber die WB-Stufen, registriert an jedem Arm.
    PASS — DM-TREPPE-KAUSAL: pfspread220 >= 0.4.
    KILL — DM-NEBENSPUR:     pfspread220 < 0.4.
  SEKUNDAER (Bericht, KEIN Crash): s50rep je Stufe — interpolierbar
  oder -1.0 = ZENSIERT (Lektion aus T20: Zensierung ist
  Messergebnis, kein Instrumentenfehler; der -1-Code haelt sie aus
  der LUECKENLISTE raus).
  UNENTSCHEIDEN (Crash): DM-Bereich < 10 Prozent relativ ueber
  Stufen, Sanity gerissen, ndiv > 6 von 20, colnorm-Verletzung.
  DANACH LINIE TOT.
  KOSTEN: 13 Arme x 20 Seeds ≈ 60-75 min (Timeout 120).
  RISIKO (ehrlich): Falls die WB-Stufen ebenfalls springen
  (Treppe wieder Kante): Crash, Linie tot, DM bleibt Geruch ohne
  Verdikt.


---

# Nachtrag 32 (2026-11-10): T20-R2 KILL — DM-NEBENSPUR. Der Schalter trug T20s Kollaps, nicht die Diagonallast. DM-Linie TOT.

## T20-R2 Lauf 1 (gefeuert 11:07 UTC, Ergebnisse 12:01 UTC, ~54 min, Bot-Commit a6381cc, 13 Arme, 0 Divergenzen, LUECKENLISTE LEER)
VERDICT (GENERIERT): KILL — DM-NEBENSPUR. pfspread220 = 0.0500
(Schwelle 0.4). Registriert in Nachtrag 31; keine Nachverhandlung;
keine Reparatur (Regel greift nur bei UNENTSCHEIDEN). DM-Linie TOT.

## Zahlen
WB-Treppe {0.6, 0.45, 0.3, 0.15} bei H=48, Spektrum fix,
Seeds 190..209. DM lief glatt mit: 13.2123 / 16.9006 / 22.2136 /
30.8182 (rel Bereich 0.8470, Guard 0.10 locker bestanden — der
Dial ist ein echter Regler, T20s Kanten-Defekt behoben).
P_lernt(2.2) je Stufe: 0.75 / 0.80 / 0.75 / 0.75. DM um Faktor
2.3 gedreht — das Lernen bewegte sich um 0.05. Keine Richtung,
keine Treppe: Rauschen um 0.75. Alle Arme n_ok=20/20, ndiv=0,
Sanity HIGH48 sauber (pf=1.0). s50rep nur bei wb=0.6 messbar
(1.8917, reproduziert T19-R2 1.83 und T20-a=0 1.89); darunter
ZENSIERT (-1, Bericht statt Crash — Zensierungs-Lektion zahlt
sich aus, die -1-Kodierung hielt die LUECKENLISTE leer).

## Lesart: Der Schalter, nicht die Last
T20s spektakulaerer Doppelkollaps (DM 13.21 -> ~1.4, P(2.2)
0.75 -> 0.05) war KEIN Diagonallast-Effekt. Es war der
Strukturbruch: die harte Konjugation Q_a W Q_a^T ist ein
Schalter, und der BRUCH brachte das Lernen um — nicht die
verschobene Last. Der glatte WB-Dial bewegt DM kontinuierlich
durch den gesamten frueheren Bereich und weit darueber hinaus:
nichts passiert. DM ist damit als Nebenpur GEMESSEN; die
Ordnungsparameter-/Phasenuebergang-These aus Nachtrag 31 ist
KILL. Ehrliche Bilanz: zwei Achsen getestet (H in T19-R2, DM in
T20/R2), zwei KILLs. Die Delle bei H=48 (1.83 gegen interpoliert
~2.4) bleibt unerklaert; als Hypothese uebrig: Seed-Rauschen
(8.5 Prozent, 20er-Satz).

## Kernel-Bilanz (Echtbetrieb)
6 kernel-generierte Steine (T15..T20R2), 3 saubere Verdikte in
Serie, 1 UNENTSCHEIDEN + 1 zensierungsfreie Reparatur, 0
Runner-Minuten auf kaputten Contracts. Bau war wieder der
Flaschenhals (diesmal: Rueckgabe-Shape von get_file_contents,
zwei Proben-Tode bevor die Form stand), nie die Engine.

## Naechster Stein (Nutzer-Entscheidung, DM-Linie tot)
(a) Delle-Statistik-Stein: mehr Seeds gezielt an H=24/48, direkt
    Rauschen-vs-Realitaet, kleinster Compute, schliesst die
    T19-Frage.
(b) P3-Befund-Graph L2: 0 Runner-Minuten, Instrument-Ausbau.
(c) T20-Verallgemeinerung auf andere Familie (nur falls das
    Schalter-Phaenomen selbst zur Linie wird).


---

# Nachtrag 33 (2026-11-10): P3-BEFUND-GRAPH v1 (L2) + T21-REGISTRIERUNG — DELLE-STATISTIK. Nutzer-Entscheidung nach T20-R2-KILL (Nachtrag 32).

## P3-BEFUND-GRAPH v1 (L2, 0 Runner-Minuten)
analysis/P3_befund_graph.md: alle Steine G1-G10, T1-T20-R2, W1 als
maschinenlesbare Knoten (Kern-Frage, Verdikt, Kern-Zahl, XVII-Anker)
plus Kanten (Folge-Logik) und Offene-Fragen-Liste. Zweck: Fundament
fuer L3/L5 — ein Vorschlags-System kann nur auf einem Graphen
arbeiten, nicht auf 105k Zeichen Prosa. v1 hand-autoriert aus XVII
(Verdiktkette gegen Nachtrag-Header verifiziert). Update-Pflicht:
jeder neue Stein bekommt seinen Knoten im selben Commit wie sein
Nachtrag.

## T21 — DIESE (P-DELLE-REAL)
Die Delle bei H=48 (s50-Reihe 12:1.30, 24:2.00, 48:1.83, 96:3.50,
192:4.675; Delle = 8.5 Prozent gegen H=24) ist REAL, nicht
Seed-Rauschen. H und DM sind als Achsen gefallen (T19-R2 KILL-A,
T20-R2 KILL); dies ist die letzte offene Messfrage der T-Linie.

## DESIGN (GENAGELT)
Frischer Seed-Block 210..309 (100 Seeds, DISJUNKT zu allen frueheren
Bloecken 190..209 — kein Seed-Anchoring; P quantisiert auf 0.01,
SE(P) ~ 0.05, SE(d183) ~ 0.07).
H in {24, 48} x s in {1.55, 1.83, 2.2} (6 sgrid-Arme) + HIGH48
(Sanity, s=5.5, wb=0.6). wb=0.6 Standard ueberall.
  PRIMAER (VERDIKT): d183 = P_lernt(48, 1.83) - P_lernt(24, 1.83),
  registriert an jedem Arm.
    PASS — DELLE-RECHT: d183 >= +0.15 (H=48 lernt FRUEHER — Delle
      real, zweites Phaenomen, neue Linie noetig).
    KILL — RAUSCHEN: -0.15 < d183 < +0.15 (Kurven identisch
      innerhalb 2 Sigma — die Delle war Seed-Lotto).
    KEHRUNG: d183 <= -0.15 (H=24 lernt frueher — Vorzeichen
      ueberraschend, Bericht + neue Frage).
  SEKUNDAER (Bericht, KEIN Crash): s50rep je H aus 3-Punkt-Kurve
    (-1 = ZENSIERT, Lektion aus T20/R2), median acc, volle Kurven.
  UNENTSCHEIDEN (Crash): Sanity gerissen, ndiv > 6 von 100,
    colnorm-Verletzung.
KOSTEN: 7 Arme x 100 Seeds ≈ 100-120 min (Timeout 200).
RISIKO (ehrlich): 3-Punkt-Kurven geben grobes s50rep; primaer
traegt allein d183. Erwartung (Band, kein Verdikt, aus T16/T19/T20-
Kurven): d183 in [+0.05, +0.20] — die Schwelle 0.15 sitzt im
Erwartungsband, der Ausgang ist offen und genau deshalb ein guter
Stein.


---

# Nachtrag 34 (2026-11-10): T21 RUN-1 CRASH — TIMEOUT. Mea culpa: Kostenmodell. Rerun mit GENAGELTEM Design (nur Timeout 300). Vorab-Zahlen BERICHTET, KEIN Verdikt.

## T21 Lauf 1 (gefeuert 17:20 UTC, abgebrochen ~20:40 UTC, ~200 min = Timeout-Limit, Bot-Commit 5332f46, 6/7 Arme vollstaendig, H48_s2.2 fehlt, keine generierte Auswertung)
CRASH-URSACHE: TIMEOUT. Als Crash registriert waren: Sanity gerissen,
ndiv > 6, colnorm — TIMEOUT stand NICHT in der Liste.
Instrumenten-Versagen, kein Physik-Ausgang. KEIN VERDIKT.

## MEA CULPA (Kostenmodell, ehrlich)
Schaetzung war 100-120 min; real brauchten ~6.9 Arme bereits 200 min
(100 arm-runs ≈ 29 min). Zwei Fehler: (1) Der Anchor HIGH48 mit 100
Seeds wurde im Kostenmodell wie ein sgrid-Arm behandelt — ist aber
teuer. (2) H=24-Runs sind nicht halb so billig wie H=48: Fixkosten
pro Run (600 Steps, Batch 32, Rollout 24 Frames) dominieren, H ist
nur ein Faktor. Lehre fuer den Kernel: Kostenmodelle muessen
Anchor-Seed-Zahl und Fixkosten pro Run ausweisen. Infra-Fix im
Repair: Timeout 300.

## VORAB-ZAHLEN (Bericht, KEIN Verdikt — H48_s2.2 fehlt)
H24:  P(1.55)=0.49 | P(1.83)=0.67 | P(2.2)=0.78
H48:  P(1.55)=0.13 | P(1.83)=0.29
HIGH48 Sanity: pf=0.97 (haelt), 0 Divergenzen, alle Arme n_ok=100/100.
AUFFAELLIG (nur Geruch, kein Verdikt): d183-Vorab = 0.29 - 0.67 =
-0.38 — KEHRUNG-Richtung, weit jenseits der ±0.15-Schwelle. UND:
Der frische Block 210..309 verschiebt BEIDE Kurven gegen den alten
Block 190..209 (P48(1.83): 0.45 alt vs 0.29 frisch; P24(1.83) alt
implizit ~0.45 aus s50(24)=2.00 vs 0.67 frisch). Moegliche Lesart,
erst ein Geruch: Block-Effekte (Seed-Familien) sind von derselben
Groessenordnung wie die Delle selbst — vielleicht war die ganze
s50-Reihe ein Block-Artefakt auf Traegerschultern. Der Rerun
entscheidet mit vollstaendiger Auswertung.

## REPARATUR (die EINE erlaubte; danach Linie tot)
Design UNVERAENDERT wie Nachtrag 33 (Seeds 210..309, 7 Arme,
d183-Schwelle ±0.15, alle Verdikte). EINZIGE Aenderung:
Workflow-Timeout 200 -> 300 Minuten (reine Infrastruktur, kein
Design-Parameter; Praezedenz T15: Rerun mit unveraendertem Design).
Erwartete Laufzeit ~230-240 min. Danach: generierte Auswertung,
Lesart als Nachtrag 35, T21-Knoten im P3-Graph im selben Commit.


---

# Nachtrag 35 (2026-11-11): T21 RUN-2 KOMPLETT — aber VERDIKT GEBLOCKT DURCH MEINEN READOUT-BUG (Anchor s50rep nie gesetzt). Reparatur: Neu-Auswertung der GEFRORENEN Run-2-Daten. Vorab: d183 = -0.38.

## Run 2 (gefeuert 20:45 UTC, Ergebnisse 00:02 UTC, 196 min, Bot-Commit be35ddf, 7/7 Arme, 0 Divergenzen)
FORMAL: UNENTSCHEIDEN (Crash-Klasse) — Kollaps-Gate: an HIGH48 ist
die registrierte Groesse s50rep NICHT GEMESSEN. Ursache: BAU-BUG,
mea culpa. Ich registrierte s50rep als Mass an JEDEM Arm, setzte es
im Instrument aber nur fuer sgrid-Arme; der Anchor blieb None.
Fehlerklasse: Readout-Vollstaendigkeit (verwandt mit T15-Run1:
Auswertungs-Readout; derselbe Lehrbereich).

## Die eingefrorenen Run-2-Zahlen (Bericht)
H24:  P(1.55)=0.49 | P(1.83)=0.67 | P(2.2)=0.78 | s50rep=1.5656
H48:  P(1.55)=0.13 | P(1.83)=0.29 | P(2.2)=0.43 | s50rep=ZENSIERT
HIGH48 Sanity: pf=0.97 (haelt), 0 Divergenzen, alle Arme n_ok=100/100.
GEMESSEN: d183 = -0.3800 (Schwelle ±0.15).

## REPARATUR (die EINE erlaubte; danach Linie tot)
KEIN Rerun der Physik: Das Experiment ist deterministisch (fixe
Seeds) — ein Rerun wuerde BIT-IDENTISCHE Zahlen produzieren und 200
Runner-Minuten verbrennen. Reparatur stattdessen als Neu-Auswertung
der GEFRORENEN Messungen (results/t21_measurements.json, Bot-Commit
be35ddf): Readout-Fix (Anchor-Kurve ist Einpunkt -> s50rep = -1
ZENSIERT, exakt die sgrid-Regel aus Nachtrag 33), danach
lab_kernel.evaluate auf den UNVERAENDERTEN Daten. Instrument:
experiments/t21_repair_readout.py, ~2 Runner-Minuten, keine neue
Physik, keine Nachverhandlung von Schwellen. Ausgang: formales
Verdikt aus Run-2-Daten; Lesart + T21-Knoten im P3-Graph als
Nachtrag 36 im selben Zug.

## Vorab-Einordnung (Bericht, KEIN Verdikt)
d183 = -0.38 liegt 5+ Sigma jenseits der Schwelle — KEHRUNG-Richtung.
Und schaerfer: Der frische Block 210..309 verschiebt BEIDE Kurven
gegen den alten Block 190..209 (H48: P(2.2) 0.75 alt vs 0.43 frisch;
P(1.83) 0.45 vs 0.29. H24: s50rep 1.57 frisch vs 2.00 alt aus T16).
Moegliche Lesart, falls KEHRUNG formal wird: Die s50-Reihe T16-T19
war BLOCK-KONFUNDIERT — Seed-Block-Effekte sind von der Groessen-
ordnung der gesamten H-Trend-These. Das Seed-Lotto aus dem G-Kapitel
schlaegt im T-Bau zurueck, mit verspaeteter Fuellung. Nachtrag 36
traegt die Lesart nach dem formalen Verdikt.
