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


---

# Nachtrag 36 (2026-11-11): T21 KEHRUNG — Vorzeichenumkehr im frischen Block. Die Delle ist tot, aber NICHT als Rauschen: Der H-Trend selbst ist block-empfindlich.

## Verdikt-Weg (formal)
Run 2 komplett (Bot-Commit be35ddf); Verdict durch meinen Readout-Bug
formal geblockt (Nachtrag 35); die EINE Reparatur als Neu-Auswertung
der gefrorenen Daten (Bot-Commit 44f3b0d, ~2 Runner-Minuten):
VERDICT: KEHRUNG. d183 = -0.3800 (Schwelle ±0.15), LUECKENLISTE
LEER, Sanity pf=0.97, 0 Divergenzen. T21-Linie nach der Reparatur
TOT (Regel: EINE Reparatur bei UNENTSCHEIDEN).

## Lesart
1. DIE DELLE IST TOT, ABER NICHT ALS RAUSCHEN. RAUSCHEN haette
   identische Kurven verlangt. Gemessen: H24 liegt an ALLEN drei
   Rasterpunkten ueber H48 (0.49/0.67/0.78 gegen 0.13/0.29/0.43).
   Die Delle existiert im Block 210..309 UEBERHAUPT NICHT — das
   Vorzeichen des H-Effekts ist gedreht. s50rep(24)=1.5656;
   s50rep(48) ZENSIERT (Kurve erreicht 0.5 erst oberhalb 2.2).
2. DIE H-TREND-THESE IST BLOCK-EMPFINDLICH. Block 190..209
   (20 Seeds, T16-T20): H48 lernt bei 1.83/2.2 gut (0.45/0.75) —
   daraus wuchsen die s50-Reihe, das Sub-Sqrt-Gesetz (T17) und
   der Delle-Befund (T19-R2). Block 210..309 (100 Seeds): H48
   durchgaengig SCHWAECHER als H24. Gleicher Bauplan, gleiches
   Instrument — zwei Bloecke, entgegengesetzte H-Ordnungen.
   Signifikanz des Block-Effekts: P48(2.2) 0.75 vs 0.43 ≈ 2.9
   Sigma; P48(1.83) 0.45 vs 0.29 ≈ 1.3 Sigma (alleinstehend
   schwach). Ehrlich: sauber belegt ist EIN Rasterpunkt plus die
   durchgaengige KEHRUNG im frischen Block.
3. ANSCHLUSS AN DAS G-KAPITEL. G2/G5: Das Seed-Lotto sitzt in
   der Basis-Rotation gegen die tanh-Achsen. T21 zeigt: Im
   T-Bau ist das Lotto so gross, dass 20er-Bloecke ganze
   Skalierungsgesetze tragen koennen, die ein 100er-Block
   umdreht. Damit ist der zweite Allgemein-Befund des Anker-
   Kapitels im T-Bau angekommen: Der Bias-Confound-Transfer
   gelang als Hebel (T12), der Seed-Lotto-Transfer jetzt als
   FALLE FUER DIE EIGENE T-SERIE. 20er-Bloecke sind KEINE
   vertraegliche Basis fuer H-Trend-Aussagen.
4. VORHERSAGE-BILANZ (ehrlich). Registriertes Band: d183 in
   [+0.05, +0.20]. Gemessen: -0.38. Wieder ein Vorzeichenfehler
   (wie T19: 2.55 gegen 1.83). Der Theory-Core lernt: Rueckgrat
   jeder kuenftigen Registrierung ist die Block-Frage.

## Naechste Linie (Nutzer-Entscheidung; T21-, DM-, H-Linien tot)
(a) T22 BLOCK-ROBUSTHEIT: EINE Zelle (H=48, s=2.2), K disjunkte
    20er-Bloecke — Verteilung der Block-P_lernt. Scharfe Frage:
    Ist std(P_Block) von der Groessenordnung der T19-Delle und
    des T16-Trend-Offsets? Kosten: K x 20 Runs (K=10 ≈ 1 h).
(b) 0-Runner-Reanalyse: alle vorhandenen T21/T20-R2-Einzelseed-
    Messungen als Block-Stichproben sammeln (D-Diff-Praxis).
(c) L5 ERSTER DURCHLAUF: Der P3-Graph steht — ein LLM schlaegt
    Stein 27 vor, registriert VOR dem Lauf. Research Taste wird
    Messobjekt (Nutzer-These Nachtrag 24).


---

# Nachtrag 37 (2026-11-11): (b) 0-RUNNER-REANALYSE — BLOCK-VERTEILUNG aus T21-Einzelseed-Daten. Zwei Befunde: Block-Streuung GROSS, KEHRUNG block-robust.

## Methode
results/t21_delle_statistik_table.csv (700 Einzelseed-Werte, Bot-
Commit be35ddf): Die 100 frischen Seeds sind 5 disjunkte 20er-
Bloecke. Block-P_lernt je Arm, 0 Runner-Minuten (D-Diff-Praxis).
Datei: analysis/P_block_reanalyse.md.

## BEFUND 1: BLOCK-STREUUNG IST GROSS
std(P_Block) je Zelle: H24 0.058/0.081/0.133; H48 0.075/0.058/0.081.
Groessenordnung 0.06-0.13 — vergleichbar mit der T19-Delle
(8.5 Prozent) und den T16-Trend-Offsets. Ein EINZELNER 20er-Block
kann H-Unterschiede dieser Groesse vortaeuschen UND verstecken.
T21-Lesart Nr. 2 ist jetzt zahlenfest: 20er-Bloecke sind als
einzige Basis fuer solche Aussagen untauglich.

## BEFUND 2: DIE KEHRUNG IST BLOCK-ROBUST (gepaart)
d183 je Block: -0.50/-0.45/-0.20/-0.25/-0.50 — KEHRUNG in 5 von 5
Bloecken. d220 je Block: ebenfalls 5 von 5 negativ. Das gepaarte
Design (dieselben Seeds an H24 UND H48) eliminiert die Block-
Varianz: Der H-Effekt im frischen Block ist KEIN Block-Artefakt.
Offen bleibt die Versoehnung mit dem ALTEN Block 190..209 — dort
gibt es nur ungepaartes Material (P48(2.2)=0.75 vs frisch 0.43,
≈ 2.9 Sigma zwischen Block-Mitteln).

## KONSEQUENZ FUER (a)
Der Stein verschaerft sich und wird BILLIGER: nicht "gibt es
Block-Streuung" (beantwortet: JA), sondern "war der ALTE Block ein
AUSREISSER?" — EINE Zelle (H=48, s=2.2), 5 neue disjunkte
20er-Bloecke (310..409), z-Mass fuer 0.75. ~120 Runs ≈ 60 min
statt 200. Registrierung als L5-Vorschlag in Nachtrag 38.


---

# Nachtrag 38 (2026-11-11): L5-DURCHLAUF 1 — ERSTER LLM-VORSCHLAG (GLM 5.3 via Vibe), registriert VOR dem Lauf. Nutzer-Reihenfolge b -> c -> a: dies ist (c) und nagelt (a) fest.

## L5-PROTOKOLL
Vorschlag generiert NUR aus analysis/P3_befund_graph.md +
Nachtraegen 36/37 — kein Nutzer-Geruch, keine zusaetzliche
Information. Registrierung VOR dem Feuer, inklusive Vorhersage-
band. Spaeter misst das Labor, ob das Band hielt: Research Taste
wird Messobjekt (Nutzer-These Nachtrag 24: "Taste bleibt beim
Mensch" ist der Bug — erster Testpunkt).

## LLM-VORSCHLAG: T22 BLOCK-ROBUSTHEIT (ALT vs NEU)
THESE: Der alte Seed-Block 190..209 war ein AUSREISSER: P48(2.2)
= 0.75 ist ein Extremwert der Block-Verteilung dieser Zelle, nicht
ihr Vertreter. Wenn ja: Die T16-T19-Kette (Sub-Sqrt-Gesetz, Delle)
beruhte auf einem Block-Unfall — und die Serie endet nicht mit
einem toten Gesetz, sondern mit einem methodischen GESETZ.

DESIGN (GENAGELT):
EINE Zelle (H=48, s=2.2, wb=0.6). 5 NEUE disjunkte 20er-Bloecke:
310-329, 330-349, 350-369, 370-389, 390-409. Anchor HIGH48
(s=5.5, 20 frische Seeds 410..429). 120 Runs ≈ 60 min (Timeout 90).
  PRIMAER (VERDIKT): zalt = (0.75 - mean(P_neu5)) / std(P_neu5),
  registriert an jedem Arm.
    PASS — ALT-AUSREISSER: zalt >= 2.5. Der alte Block ist
      Extremwert; T16-T19-Kette ein Block-Unfall. Konsequenz:
      H-Trend-Aussagen nur noch als Block-Mittel mit Angabe von
      std(P_Block).
    KILL — KONTINUITAET: zalt < 2.5. Der Wert 0.75 liegt innerhalb
      der Block-Verteilung; H48(2.2) streut so breit, dass
      Einzelblock-Trends bedeutungslos sind — dieselbe Konsequenz
      wie PASS, nur schwaecher signifikant.
    UNENTSCHEIDEN (Crash): std(P_neu5) < 0.02 (degenerierte
      Streuung, z-Mass sinnlos), Sanity gerissen, ndiv > 6 je
      Block, colnorm-Verletzung.
  SEKUNDAER (Bericht): Block-P je Block; 10-Block-Gesamtverteilung
    (5 aus Nachtrag-37-Reanalyse + 5 neu); zalt-10-Variante als
    Bericht.
  VORHERSAGEBAND (L5, KEIN Verdikt): mean(P_neu5) in [0.30, 0.56];
    std(P_neu5) in [0.05, 0.13]; zalt in [2.0, 6.0].

## BEWERTUNGS-METRIK (L5-Bilanz, spaeter nachtragen)
Teilband-Treffer von 3. Band generiert ohne Blick auf die neuen
Daten (existieren noch nicht). Falls das Band haelt: erster L5-
Punkt. Falls nicht: erste messbare Fehl-Calibration des Vorschlags-
Modells.



---

# Nachtrag 39 (2026-11-11): T22 VERDICT — PASS: ALT-AUSREISSER. Der alte Block war ein Unfall, kein Gesetz. L5-Bilanz 3/3.

## Lauf
Run 42984601545 (08:07-08:46 UTC, 39 min), Bot-Commit b4b0f91
("results: T22-block-robustheit 1"). 120/120 Runs, Lueckenliste LEER,
Vertrag exakt wie in Nachtrag 38 genagelt (kein Design-Drift).

## Zahlen (Bot, unveraendert uebernommen)
- Block-P (310..409): B0 0.35, B1 0.40, B2 0.50, B3 0.40, B4 0.65.
- mean(P_neu5) = 0.46, std(P_neu5) = 0.107.
- zalt = 2.72 (Schwelle 2.5) -> PASS.
- zalt-10 (Bericht, gegen alle 10 Bloecke inkl. Reanalyse-Konstanten
  [0.35, 0.40, 0.55, 0.50, 0.35]) = 3.18.
- 10-Block-Verteilung: mean 0.445, std 0.096.

## Lesart
1. DER ALTE BLOCK WAR EIN AUSREISSER. P48(2.2)=0.75 (Block 190..209)
   ist ein Extremwert der Block-Verteilung, nicht ihr Vertreter. Die
   T16-T19-Kette (Sub-Sqrt-Gesetz, Delle) beruhte auf einem
   Block-Unfall. Die Serie endet nicht mit einem toten Gesetz,
   sondern mit einem METHODISCHEN GESETZ: Einzelblock-Trends der
   Groesse ~0.1 sind in dieser Simulationsklasse nicht
   interpretierbar; H-Aussagen nur noch als Block-Mittel mit
   std(P_Block) (Pass-Konsequenz aus Nachtrag 38, jetzt bindend).
2. B4 = 0.65 (median acc 1.0): Der Ausreisser-Mechanismus ist
   REPRODUZIERBAR — 1 von 5 neuen Bloecken knallt nach oben. Das
   Seed-Lotto des G-Kapitels (G2/G5) schlaegt im BLOCK-Massstab,
   nicht nur pro Einzel-Seed. Der Extremwert ist kein einmaliger
   Betriebsunfall, sondern ein ~20-Prozent-Ereignis pro Block.
3. VERSOEHNUNG MIT NACHTRAG 37. Die 2.9-Sigma-Diskrepanz
   (0.75 alt vs 0.43 frisch) ist aufgeloest: 0.43 war selbst ein
   BLOCKWERT am unteren Rand, 0.75 einer am oberen. Beide sind
   keine Zell-Wahrheit; die Zell-Wahrheit ist die Verteilung
   (mean 0.445, std ~0.10 ueber 10 Bloecke). Die Reanalyse-std
   (0.06-0.13) und die T22-std (0.107) sind konsistent —
   Nachtrag 37 haelt stand.
4. KONSEQUENZ FUER DIE T-SERIE. Der H-Trend ist als
   Einzelblock-Gesetz endgueltig tot (T21 KEHRUNG + T22 PASS
   zusammen). T16-T19-Zahlen duerfen kuenftig nur mit
   Block-Fehlerbalken zitiert werden. Offene Folge-Fragen:
   BLOCK-MITTEL-GESETZ (gibt es einen echten Trend im
   Block-Mittel ueber H?) oder SCHALTER-Linie (T20-Nebenbefund:
   harte Konjugation als Strukturbruch).

## L5-Bilanz (Durchlauf 1, Research Taste als Messobjekt)
Registriertes Band (Nachtrag 38, VOR dem Lauf):
- mean in [0.30, 0.56] — gemessen 0.46 — TREFFER.
- std in [0.05, 0.13] — gemessen 0.107 — TREFFER.
- zalt in [2.0, 6.0] — gemessen 2.72 — TREFFER.
3/3 Teilband-Treffer. Erster Messpunkt fuer die Nutzer-These
(Nachtrag 24: Research Taste ist messbar). Ehrliche Einschraenkung:
n=1, und das Band war breit — ein Treffer ist schwache Evidenz fuer
Kalibration; eine Fehl-Calibration wurde noch NICHT beobachtet
(das waere der informativere Fall). Der Loop Taste -> Vorschlag ->
Kernel -> Verdikt -> Gedaechtnis hat einen vollen Zyklus
durchlaufen: Vorschlag aus P3-Graph allein, Registrierung vor dem
Lauf, Verdikt ohne Nachverhandlung, Nachtrag im selben Commit wie
der Graph-Knoten.

## Kernel-Bilanz (Stand nach T22)
8 kernel-generierte Steine (T15..T22). T22 ist der erste Stein auf
LLM-Vorschlag — im ERSTEN Versuch, 0 Crash, 39 Runner-Minuten,
kein Rerun. Verdikt-Serie: 5 saubere Verdikte in Serie
(T19-R2, T20-R2, T21, T22 + Reparatur-Verdikt). Der Bau blieb der
einzige Flaschenhals der Saison; die Engine hat seit T19-R2 keinen
einzigen Fehler produziert. Wrapper-Guard und Syntax-Gate in allen
Workflows: 0 Runner-Minuten auf kaputten Contracts seit Rollout.


---

# Nachtrag 40 (2026-11-11): P4-GENERATOR v1/v1.1 (L3 IN ECHTBETRIEB) — erster Maschinen-Vorschlag, erster Maschinen-Defekt, erste R7-Lehre. T23-REGISTRIERUNG.

## L3 in Echtbetrieb
P4-Generator v1 gebaut und gelauft (~2 Runner-Minuten, kein
Torch): deterministische Regeln-Maschine auf eingefrorenem
P3-Snapshot, Kritiker-Regeln R1-R6, dokumentiertes Scoring
(Ueberraschung 0.5, Kostenquote 0.3, Konsequenz-Scheidung 0.2).
Bot-Commit d87cf6c (v1.0-Proposal), v1.1-Regeneration analog.

## Der erste Maschinen-Vorschlag und sein DEFEKT (ehrlich)
v1.0 schlug T23 mit Raster a=[0,4,12,24] vor. Kernel-Kritiker-
Cross-Check gegen die gefrorenen T20-Einzelmessungen riss ihn
VOR dem Feuer (0 Runner-Minuten Verlust):
1. Die T20-Dosis-Kurve zeigt den Kollaps BEREITS bei a=1
   (P 0.75 -> 0.05; a=4: 0.15; a=12: 0.05). Das Schalter-Fenster
   liegt zwischen a=0 und a=1 — ausserhalb des v1.0-Rasters.
2. Folge: vorhersagbares UNENTSCHEIDEN (drop4 gross, kein
   Folge-Abfall; beide Verdikt-Praedikate konnten nicht feuern).
   56 Runner-Minuten waeren verbrannt.
3. Das v1.0-Band widersprach seiner eigenen Quelle (drop4 in
   [-0.10, 0.25] gegen T20-drop4 ~ 0.60): Der Snapshot enthielt
   Aggregate ohne Dosis-Kurve.
4. R1-R6 pruefen Prozess-Hygiene, nicht Raster-Adaequanz —
   deshalb NEUE Regel R7 (UEBERGANGSFENSTER-ABDECKUNG): Ein
   Raster muss die Klammer-Stufen des steilsten beobachteten
   Uebergangs enthalten. R7 haette v1.0 abgelehnt (Beleg im
   v1.1-Proposal).
LEHRE: Der Kritiker muss die QUELL-Daten pruefen, nicht nur den
Prozess. Der Loop hat den ersten Maschinenfehler in der
Registrierungsphase gefangen — das Pre-Mortem-Gat funktioniert,
diesmal mit Menschen als Kritiker; ab v1.1 ist die Lehre Regel
im Generator.

## v1.1-Regeneration
Snapshot um T20-Dosis-Kurve erweitert, R7 ergaenzt, neu
generiert: T23 SCHALTER-DOSIS v1.1 (analysis/P4_proposal_v1_1.md,
R1-R7 alle ERFUELLT). T24 BLOCK-MITTEL-GESETZ vertagt (ueber Cap,
konfirmatorisch).

## T23-REGISTRIERUNG (GENAGELT, keine Nachverhandlung)
FRAGE: Repliziert der Lern-Kollaps bei Konjugationstiefe a=1
(T20, alter Ausreisser-Block 190..209) auf frischen Bloecken?
DESIGN: EINE Zelle (H=48, s=2.2, wb=0.6, T20-Basisparameter).
Stufen a in {0, 1, 4, 24} (Konjugation Q_a W Q_a^T,
Q_SEED=2020 wie T20). Je Stufe 2 disjunkte 20er-Bloecke,
frische Seeds 430..589; Anchor HIGH48 (s=5.5, 590..609).
180 Runs, Kostenmodell R3: Fix ~5 min + 17 s/Run x 180 =
56 min (Cap 60), Timeout 75.
PRIMAER (VERDIKT): drop1 = P(0) - P(1), registriert an jedem
Arm. SE(Stufen-Mittel)=0.068, SE(drop1)=0.096.
  PASS — SCHALTER-BESTAETIGT: drop1 >= 0.30 (>= 3.1 SE).
    Konsequenz: eigene Linie SCHALTER; erste Anschlussfrage:
    Ist die top-Singulaerrichtung die kritische? (Gegenprobe:
    Konjugation entlang Zufallsrichtung a=1.)
  KILL — KEIN-KOLLAPS: drop1 < 0.30.
    Konsequenz: T20-Doppelkollaps war Seed-Lotto am
    Ausreisser-Block; Schalter-Linie TOT.
  UNENTSCHEIDEN (Crash): Sanity (HIGH48 pf < 0.5), ndiv > 6 je
    Block, std(P_Block) < 0.02 je Stufe, colnorm. EINE
    Reparatur (R5), dann Linie tot.
SEKUNDAER (Bericht): Schalter-FORM: drop124 = P(1) - P(24);
< 0.15 = Plateau nach Kipp (BINAER-Form), fortlaufender Abfall
= Dosis-Form. Kein Verdikt.
VORHERSAGEBAND (mechanisch aus Block-Statistik, KEIN
Hand-Tuning — erster Maschinen-Band der Serie):
  P(0) in [0.31, 0.58] | P(1) in [0.00, 0.25] |
  P(4) in [0.00, 0.30] | P(24) in [0.00, 0.25] |
  drop1 in [0.05, 0.55] | drop124 in [-0.10, 0.20].

## L5-Bilanz-Status (Durchlauf 2)
Vorschlagsquelle erstmals MASCHINE (P4-Generator v1.1) statt
LLM. Band-Mechanik: Schwellen und Grenzen aus gemessener
Block-Statistik (T22 + Nachtrag 37), nicht aus Gefuehl —
damit ist eine Fehl-Calibration, falls sie eintritt, auf den
Generator zurueckfuehrbar. Bewertung: 6 Teilband-Treffer
(P(0), P(1), P(4), P(24), drop1, drop124); der informative
Fall bleibt die Verfehlung (Nachtrag-39-Einschraenkung).

## Kernel-Bilanz (Stand)
Der Loop Taste -> Vorschlag -> Kernel -> Verdikt ->
Gedaechtnis laeuft jetzt mit ZWEI Vorschlags-Quellen (LLM:
T22; Maschine: T23) und EINEM Kritiker-Gat, das seinen ersten
Fund in der Registrierungsphase machte (0 Runner-Minuten).
Ab T23: 9 kernel-generierte Steine; L3 offiziell in
Echtbetrieb — AGI-Loop-These: 3.5 von 4 Bausteinen.


---

# Nachtrag 41 (2026-11-11): T23 RUN-1 — SCHALTER REPLIZIERT (drop1=0.55, ~5.7 SE), Verdict geblockt durch Readout-Bug (mea culpa). R5-REPARATUR REGISTRIERT: Pfad (b).

## Lauf
Run 43074001615 (18:02-18:56 UTC, 54 min), Bot-Commit 184c7c1.
180/180 Runs, Sanity clean (HIGH48 pf=1.0), keine Divergenz,
colnorm exakt. Vertrag wie Nachtrag 40 genagelt.

## Zahlen (Bot, unveraendert)
- Stufen-Block-Mittel: a=0: 0.575 | a=1: 0.025 | a=4: 0.10 |
  a=24: 0.15. std(P_Block): 0.075 / 0.025 / 0.05 / 0.00.
- drop1 = 0.55 (Schwelle 0.30) — das waeren ~5.7 SE, weit jenseits
  der registrierten 3.1-SE-Schwelle.
- drop124 = -0.125 (leichte ERHOLUNG nach Kipp: a=24 > a=1).

## Verdict: UNENTSCHEIDEN (Crash) — zwei Ursachen
1. READOUT-BUG (mea culpa, maximal peinlich): HIGH48.pstage wurde
   vom Instrument nie gesetzt — die T21-Lektion (registrierte
   Masse an JEDEM Arm, Anchor inklusive) hatte ich als R2-Regel
   selbst in den Contract geschrieben und im Instrument verletzt.
   Exakt der T21-Crash, nochmal, vom selben Autor.
2. std(P_Block) = 0.0 bei a=24 (0.15/0.15) < 0.02: Die aus T22
   uebernommene Crash-Klausel feuert. Sie schuetzte dort den
   Nenner des z-Masses; T23 hat keinen Nenner.

## R5-REPARATUR (die EINE erlaubte; registriert VOR dem Rerun)
Pfad (b), Nutzer-Entscheidung. Deterministische Neu-Auswertung
der gefrorenen Daten (results/t23_measurements.json, Run 1),
KEIN Rerun, kein Torch, ~2 Runner-Minuten. Zwei Eingriffe:
- Reparatur 1 (READOUT, T21-Praezedenzklasse): HIGH48.pstage =
  HIGH48.pf (Ein-Arm-Stufenmittel des Anchor; Definition hier
  dokumentiert). Kein Messwert wird veraendert.
- Klausel-Dokumentation (KEINE Daten-Aenderung): Die std-Klausel
  wird auf ihren dokumentierten Ursprungszweck zurueckgefuehrt —
  Schutz von Nenner-fuehrenden Primaer-Massen (T22 zalt). T23s
  Primaermass drop1 hat keinen Nenner; die verdikts-entscheidende
  Stufe a=1 hat std=0.025 >= 0.02 und ist clean. a=24 (std=0.0)
  ist Boden-Stabilitaet, als Form-Befund dokumentiert. drop1 und
  Schwelle 0.30 UNBERUEHRT — keine Nachverhandlung des
  Verdikt-Masses. Erwartetes Ergebnis: PASS (drop1=0.55 >= 0.30).
PRAEZEDENZ, ehrlich benannt: Erster Fall einer rueckgefuehrten
Crash-Klausel. Grenze gezogen: Verdikt-Mass, Schwelle, Band und
Konsequenzen sind unantastbar; nur der Geltungsbereich einer
Klausel wird auf ihren dokumentierten Zweck begrenzt.

## Lesart (der gefrorenen Zahlen — unabhaengig vom Reparatur-Ausgang)
1. DER SCHALTER IST REAL UND EXTREM SCHARF. Schon EINE
   konjugierte Dimension kippt P_lernt auf frischen Bloecken
   von 0.575 auf 0.025. Der T20-Befund war KEIN Block-Artefakt —
   im Gegenteil: Der alte Ausreisser-Block hat den Schalter
   eher UNTERtreibt (0.75 war der Block-Oberwert; der frische
   Block-Mittelwert 0.575 ergibt drop1=0.55 statt 0.70).
2. KEINE DOIS-KURVE: Nach dem Kipp Plateau (0.10-0.15) mit
   leichter Erholung — kein monotoner Abfall. Der GLEITEND-
   Ausgang ist damit empirisch praktisch ausgeschlossen.
3. ANSCHLUSS (bei PASS): eigene Linie SCHALTER. Erste Frage:
   Ist die top-Singulaerrichtung die kritische? Gegenprobe
   Konjugation entlang Zufallsrichtung a=1. Danach: Ordnungs-
   parameter des Kipppunkts (DM parallel kollabiert, T20).

## L5-Bilanz (Durchlauf 2, Vorschlagsquelle MASCHINE)
- P(0) [0.31,0.58]: 0.575 TREFFER | P(1) [0.00,0.25]: 0.025
  TREFFER | P(4) [0.00,0.30]: 0.10 TREFFER | P(24) [0.00,0.25]:
  0.15 TREFFER | drop1 [0.05,0.55]: 0.55 TREFFER.
- drop124 [−0.10,0.20]: −0.125 **NEIN** — DIE ERSTE VERFEHLUNG
  DER SERIE, und sie sitzt genau an der duennen Quelle (T20
  n=1, alter Block): Die leichte Erholung bei a=24 war aus den
  Daten nicht vorhersagbar. Das Band war nicht trivial-breit —
  es verfehlte ehrenwert an der richtigen Stelle.
- LEHRE FUER DEN GENERATOR (v1.2, Anpassung an das Instrument,
  kein Rerun noetig): Band-Grenzen aus n=1-Quellen bekommen
  zusaetzliche Weite (z.B. 50 Prozent) oder ein Auszeichnungs-
  Merkmal "duenne Quelle" im Proposal.
- SERIEN-STAND Research Taste: LLM 3/3 (breites Band) |
  Maschine 5/6 (mechanisches Band, mit Verfehlung). Erste
  kalibrierbare Differenz zwischen zwei Vorschlags-Quellen.

## Kernel-Bilanz (ehrlich)
9 kernel-generierte Steine (T15..T23). T23: erster Maschinen-
Vorschlag, erster Lauf, erster Readout-Crash — WIEDER im
Readout (T21-Parallelfall, gleicher Urheber). Lehre fuer den
Kernel v2: Preflight muss pruefen, dass JEDES registrierte Mass
an JEDEM Arm platzfindet (R2-Check maschinell, nicht per
Selbstkontrolle des Bauenden). Bau blieb Flaschenhals; die
Engine hat auch diesen Lauf fehlerfrei durchgezogen.


---

# Nachtrag 42 (2026-11-11): T23 REPARATUR-VERDICT — PASS: SCHALTER-BESTAETIGT. Neue Linie SCHALTER eroeffnet. Recon-Positionierung.

## Verdict
Reparatur-Run (Bot-Commit 19:46 UTC, 2 Runner-Minuten): Readout-Fix
NUR HIGH48.pstage = pf (Ein-Arm-Stufenmittel, n41-Definition);
geforderte Kontrolle: gefrorene Werte unveraendert (drop1 = 0.55,
drop124 = -0.125, kein Messwert angefasst). Klausel-Protokoll:
std(a=1) = 0.025 >= 0.02 — verdikts-entscheidende Stufe CLEAN;
std(a=24) = 0.0 Boden-Stabilitaet, Form-Befund; Klausel-Ursprung
(T22 Nenner-Schutz) dokumentiert. Lueckenliste LEER.
VERDICT: PASS — SCHALTER-BESTAETIGT (drop1 = 0.55 >= 0.30, ~5.7 SE).

## Die neue Linie SCHALTER (Kopf-Stein T23)
- Befund: Eine einzige konjugierte Dimension (a=1) kippt P_lernt
  auf frischen Bloecken von 0.575 auf 0.025. Kein Block-Artefakt
  (T22-Lehre angewendet — der alte Ausreisser-Block hat den
  Schalter eher UNTERtrieben). Form: Plateau mit Mini-Erholung
  (0.10-0.15), KEINE Dosis-Kurve. GLEITEND ist empirisch tot.
- PRAEZEDENZ-PROTOKOLL (ehrlich): Erste rueckgefuehrte Crash-
  Klausel der Serie (Pfad (b), Nutzer-Entscheidung). Grenze
  gezogen: Verdikt-Mass, Schwelle, Band UNANTASTBAR; nur der
  Geltungsbereich einer Klausel wurde auf ihren dokumentierten
  Ursprung (Nenner-Schutz) begrenzt. Kuenftige Registrierungen:
  std-Klauseln NUR an Nenner-fuehrenden Primaer-Massen.

## Recon-Positionierung (2026-11-11, vor jeder Unbekannt-Behauptung)
1. KONTRAST ZUR ROTATIONSINVARIANZ: Rekurrente-Kernel- und
   Mean-Field-Theorie sind rotationsinvariant (Reservoir Computing
   meets Recurrent Kernels and Structured Transforms, NeurIPS
   2020; Jaeger/Lukosevicius 2009 empfehlen sogar orthogonale
   Matrizen als Topologie). Unser Befund ist der Gegensatz bei
   endlichem H mit elementweiser tanh: Die Invarianz bricht
   nicht statistisch, sondern als SCHALTER — eine Dimension
   Basis-Rotation gegen den tanh-Anker toetet das Aufgabenlernen
   (Anschluss G5/G6 des Anker-Kapitels: der Anker ist real und
   scharf).
2. 'WHY LLMS AREN'T SCIENTISTS YET' (Trehan/Chopara, arXiv
   2601.03315): Drei von vier autonomen LLM-Forschungspipelines
   scheitern daran, stillschweigendes Domains-Wissen in
   glaubwuerdige Hypothesenformulierung zu operationalisieren.
   Unsere Gegen-Serie (L3/L5): Still schweigendes Wissen als
   REGELN kodieren (R1-R7), Vorschlag mit Band VOR dem Lauf,
   Kernel-Verdikt ohne Nachverhandlung, messbare Band-Quote.
   Erste Datenpunkte: LLM 3/3 (breites Band) | Maschine 5/6
   (mechanisches Band, Verfehlung an duenner Quelle).

## L5-Bilanz nach Durchlauf 2 (final)
Maschinen-Band: 5/6 Treffer (P(0) 0.575, P(1) 0.025, P(4) 0.10,
P(24) 0.15, drop1 0.55 — alle im Band; drop124 -0.125 gegen
[-0.10, 0.20] NEIN). Die einzige Verfehlung sitzt an der einzigen
duennen Quelle (T20 n=1, alter Block) — das Band war nicht
trivial-breit, es verfehlte ehrenwert an der richtigen Stelle.
Generator-Lehre v1.2: Band-Grenzen aus n=1-Quellen bekommen 50
Prozent zusaetzliche Weite. Research Taste ist jetzt eine Serie
mit ZWEI Quellen und einer Verfehlung — messbar, wie gefordert
(Nutzer-These Nachtrag 24).

## Naechster Stein (Generator v1.2, L3)
T24-Kandidat SCHALTER-RICHTUNG: Kippt JEDE 1-dim-Konjugation
(RICHTUNGS-AGNOSTISCH — Symmetriebruch-Mechanismus, Anschluss
G6) oder nur bestimmte Rotationsachsen (RICHTUNGS-SPEZIFISCH —
Achsen-Lotto)? Vorschlag generiert vom P4-Generator v1.2 (T23-
Dosis-Kurve frisch im Snapshot), Registrierung als Nachtrag 43.

## Kernel-Bilanz (Stand)
9 kernel-generierte Steine (T15..T23). 6 saubere Verdikte in
Serie. Zwei Reparaturen der Saison (T21, T23) — beide in der
Readout-Klasse, beide 2 Runner-Minuten, beide mit Praezedenz-
Dokumentation. Lehre fuer den Kernel v2 notiert: Preflight muss
R2 maschinell pruefen (jedes registrierte Mass an jedem Arm).
Bau blieb Flaschenhals; die Engine ist weiter fehlerfrei.



---

# Nachtrag 43 (2026-11-11): T24 SCHALTER-RICHTUNG REGISTRIERT (L5-Durchlauf 3, Quelle Maschine). Band und Schwellen VOR dem Lauf eingefroren.

## Registrierung
Vorschlag: P4-Generator v1.2 (Bot-Commit 20:09 UTC, Proposal v1.2,
Score 0.78 — Ueberraschung 1.0, Kostenquote 0.74, Scheidung 1.0).
Erster Test der Generator-Lehre R8 (BAND-QUELLEN-HAERTUNG) im
Echtbetrieb. Registrierung NACH Maschinen-Vorschlag, VOR Bau und
Lauf — Freeze-Regel gilt, keine Nachverhandlung.

## Frage
Kippt JEDE 1-dim-Konjugation das Lernen (RICHTUNGS-AGNOSTISCH,
Symmetriebruch-Mechanismus, Anschluss G6) oder nur bestimmte
Rotationsachsen (RICHTUNGS-SPEZIFISCH, Achsen-Lotto)?

## Design (eingefroren)
EINE Zelle (H=48, s=2.2, wb=0.6, T20-Basisparameter:
t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600, batch=32,
lr=5e-3, lam_max=0.6, fixed_p0=4242, test_seed=777, hold=3).
Konjugationstiefe a=1, Q_a = QR(I + 1*G_q), 5 Achsen via q_seed
in [2020, 5150, 8080, 1111, 4242], je 20 frische Seeds:
  Q2020: 610..629 | Q5150: 630..649 | Q8080: 650..669
  Q1111: 670..689 | Q4242: 690..709
Control a=0: 710..729 | Anchor HIGH48 (s=5.5, a=0): 730..749.
Total 140 Runs, Fix 5.0 min + variabel 39.7 min = 44.7 min
(Cap 60). q_seed=2020 ist die Replikations-Achse (R7: T23 bei
a=1 mass 0.025 — Vorab-Replikation im selben Lauf).

## Masse (R2: an JEDEM Arm inkl. Anchor)
pf, colnorm, ndiv, P_Q (Block-P je Achse), pqmax = max(P_Q).

## Verdict-Schwellen (unantastbar)
- PASS — RICHTUNGS-AGNOSTISCH: pqmax <= 0.15. Jede getestete
  Achse kippt. Konsequenz: Symmetriebruch-Mechanismus bestaetigt
  (Anschluss G6), naechster Stein: Ordnungsparameter des
  Kipppunkts.
- KILL — RICHTUNGS-SPEZIFISCH: pqmax >= 0.30. Mindestens eine
  Achse laesst Lernen zu. Konsequenz: Achsen-Stein — top-
  Singulaerrichtung vs Zufallsrichtung wird naechste Frage.
- UNENTSCHEIDEN: 0.15 < pqmax < 0.30 oder Crash (Sanity
  Anchor pf < 0.5, ndiv > 6 je Block, colnorm-Verletzung).
  R5: EINE Reparatur, dann Linie tot. R4 erfuellt (PASS und
  KILL fuehren zu verschiedenen naechsten Steinen).

## Vorhersageband (mechanisch, R8-Haertung)
- P_Q je Achse (5 Werte) in [0.00, 0.22]
  (Quelle: T23 a=1 frisch 0.025, n=1 Achse, R8: +50% Weite)
- pqmax in [0.05, 0.25] (Union beider Hypothesen)
- P(0)-Control in [0.25, 0.64]
  (Quelle: 10-Block-Statistik T22/n37: 0.445 +- 2*0.096)

## L5-Serien-Zettel (Durchlauf 3)
Quelle: Maschine (Generator v1.2). Band-Treffer werden nach dem
Lauf an Nachtrag 44 fortgeschrieben; die Verfehlung von
drop124 (Durchlauf 2) bleibt der einzige Maschinen-Missstand —
R8 ist genau dafuer zum ersten Mal im Spiel.

## Kernel-Bilanz (Stand, unveraendert bis T24-Verdict)
9 kernel-generierte Steine (T15..T23). 6 saubere Verdikte,
2 Readout-Reparaturen. T24 ist der erste Stein mit R8-Band und
dem Achsen-Schnitt-Verdikt (pqmax).



---

# Nachtrag 44 (2026-11-11): T24 VERDICT — PASS: RICHTUNGS-AGNOSTISCH. Sauberster Stein der Saison (keine Reparatur, Lueckenliste LEER). Ordnungsparameter ist die naechste Frage.

## Lauf
Run 43109271510 (22:09-22:38 UTC, 29 Runner-Minuten), Bot-Commit
55e027f. 140/140 Runs, Sanity clean (HIGH48 pf=1.00), ndiv=0
ueberall, colnorm exakt, R2-Readout: pq/pqmax an JEDEM Arm
gesetzt — erster kernel-generierter Stein OHNE jede
Intervention (kein Readout-Crash, keine Reparatur).

## Zahlen (Bot, gefroren)
Achsen-P_Q (a=1, je 20 frische Seeds):
  q_seed=2020: 0.05 | 5150: 0.05 | 8080: 0.05 | 1111: 0.10
  | 4242: 0.00
pqmax = 0.10 (Schwelle PASS <= 0.15). Control a=0: P(0) = 0.60.
Anchor HIGH48: 1.00. L5-BAND (R8-Erstfahrt): 7/7 TREFFER.

## Lesart
1. JEDE ACHSE KIPPT. Alle fuenf Rotationsrichtungen bringen
   P_lernt von 0.60 auf 0.00-0.10. Der Schalter von T23 ist
   kein Achsen-Artefakt: RICHTUNGS-AGNOSTISCH bestaetigt.
2. Kein Achsen-Lotto: Die Streuung ueber Achsen (0.00..0.10)
   liegt innerhalb des Block-Rauschens (SE(P_Block)=0.068,
   T22). Q4242=0.00 und Q1111=0.10 sind nicht trennbar —
   Achsen-Unterschiede sind bisher untestbar klein.
3. REPLIKATION: Q2020 frisch 0.05 vs T23 0.025 — stabil am
   Boden, kein Erholungstrend zwischen Bloecken.
4. MECHANISMUS-STUFE: Symmetriebruch. Eine einzige gegen den
   tanh-Anker rotierte Dimension toetet das Aufgabenlernen,
   egal in welche Richtung. Anschluss G6 (volle Symmetrie
   toetet Kapazitaet) jetzt als SCHALTER-Phaenomen bei
   endlichem H gefasst: Der Kontrast zur rotationsinvarianten
   Reservoir-Theorie (NeurIPS 2020, Jaeger/Lukosevicius 2009)
   ist zweiseitig scharf — Invarianz ist bei endlichem H
   kein Kontinuum, sondern ein Kliff (G7-Parallelfall).
5. R8-Erstfahrt: 7/7 — und das Band war nicht trivial-breit:
   P(0)=0.60 sass nahe der Oberkante 0.64, pqmax=0.10 weit
   unten. Die +50%-Haertung hat die n=1-Baender der Achsen
   gerade nicht zur Freifahrt gemacht.

## L5-Serien-Zettel (Durchlauf 3, Quelle MASCHINE)
Maschine: 7/7 in diesem Lauf; Serie Maschine gesamt 12/13.
LLM: 3/3. Gesamtkalibration: 15/16 — die eine Verfehlung
bleibt drop124 (Durchlauf 2, duenne Quelle). Research Taste:
beide Quellen liefern brauchbare Baender; die Maschine wird
mit R8 erstmals ohne Verfehlung getestet.

## Naechster Stein (SCHALTER-Linie, Generator-Rotation steht an)
ORDNUNGSPARAMETER DES KIPPPUNKTS: Was unterscheidet kollabierte
von intakten Netzen MESSBAR vor dem Verdikt? T20-Kandidat: der
DM-Kollaps (13.21 -> 1.4) parallel zum Schalter. Frage: Ist DM
ein Ordnungsparameter des Schalters (steigt/springt er mit a),
und sagt er die Kipp-Zugehoerigkeit je Seed voraus?
Generierungs-Pfad: P4-Generator v1.3 mit T24-Frischdaten im
Snapshot.

## Kernel-Bilanz (ehrlich)
10 kernel-generierte Steine (T15..T24). T24: erster komplett
unproblematischer Lauf der Serie (Bau, Lauf, Readout, Verdikt
in einem Durchgang, 29 Minuten, 0 Alarme). Lehre bestaetigt:
R2 als maschinelle Readout-Schleife (nicht als
Bauenden-Selbstkontrolle) hat der T21/T23-Fehlerklasse die
Grundlage entzogen. Bau blieb Flaschenhals der Saison, aber
die Fehlerklasse ist bei T24 erstmals leer.



---

# Nachtrag 45 (2026-11-12): T25 ORDNUNGSPARAMETER REGISTRIERT (L5-Durchlauf 4, Quelle Maschine). Band und Schwellen VOR dem Lauf eingefroren.

## Registrierung
Vorschlag: P4-Generator v1.3 (Bot-Commit bae6c64, 07:11 UTC,
Proposal v1.3, Score 0.80 — Ueberraschung 1.0, Kostenquote
0.56, Scheidung 1.0). Erster Maschinen-Vorschlag auf Basis der
T24-Frischdaten. Recon-Positionierung (2026-11-12, vor der
These): Die Recherche findet KEIN Doppel des Schalter-
Befunds; naechste Nachbarn sind (1) Geometric-Continuity-
Arbeit arXiv 2605.04971 (symmetriebrechende Nichtlinearitaeten
pinnen Schichten an ein gemeinsames Koordinatensystem — MLP/
Transformer, Kontinuum, kein binarer Schalter, nicht
rekurrent), (2) Warmuth, Downside of Rotation Invariance
(untere Schranken rotationsinvarianter Algorithmen, Theorie
ohne empirischen Schalter), (3) Vorontsov 2017 (Orthogonalitaet
als RNN-Empfehlung — genau der Knaul, an dem wir den Schalter
messen), (4) Seed-Folklore (torch.manual_seed(3407), mean+-std-
Bericht). Prioritaets-Fenster als endlich eingeschaetzt.

## Frage
Ist der DM-Kollaps (T20-Beifang 13.21 -> 1.40, parallel zum
Schalter) ein ORDNUNGSPARAMETER des Kipppunkts — trennt DM
gekippte von intakten Netzen VOR dem Verdikt — oder ein
Begleitprodukt (konsistent mit T20-R2 KILL DM-NEBENSPUR)?

## Design (eingefroren)
EINE Zelle (H=48, s=2.2, wb=0.6, T20-Basisparameter).
Stufen a in {0, 1, 4, 24} (Konjugation Q_a W Q_a^T,
Q_SEED=2020), je 1 frischer 20er-Block:
  a=0: 750..769 | a=1: 770..789 | a=4: 790..809 | a=24: 810..829
Anchor HIGH48 (s=5.5, a=0): 830..849. DM-Definition
unveraendert aus dem T20-Instrument. Total 100 Runs,
Fix 5.0 + variabel 28.3 = 33.3 min (Cap 60). R7: a=1 ist
die Replikations-Stufe (T23), a=0 die Klammer (T24-P(0)).

## Masse (R2: an JEDEM Arm inkl. Anchor)
pf, colnorm, ndiv, DM je Seed; DM-Mittel je Stufe;
dmdrop = DM(0) - DM(1).

## Verdict-Schwellen (unantastbar)
- PASS — ORDNUNGSPARAMETER-BESTAETIGT: dmdrop >= 5.0 UND
  dmdrop >= 3 * SE(dmdrop, im Lauf gemessen) UND Sanity
  clean. Konsequenz: DM wird Vorhersage-Instrument; naechster
  Stein: DM-Schwellwert-Kippregel pro Seed.
- KILL — DM-NEBENPRODUKT: dmdrop < 5.0 oder < 3 * SE.
  Konsequenz: DM als Diagnostikum gefallen; Ordnungsparameter-
  Suche geht zu Gain-Verteilung.
- UNENTSCHEIDEN: Crash (Sanity Anchor pf < 0.5, ndiv > 6 je
  Block, colnorm-Verletzung). R5: EINE Reparatur, dann Linie
  tot. R4 erfuellt.

## Vorhersageband (mechanisch, R8-Haertung)
- DM(0) in [8.80, 19.80] (T20 n=1: 13.21, +50% Weite)
- DM(1) in [0.90, 2.10] (T20 n=1: 1.40, +50% Weite)
- dmdrop in [5.90, 17.70] (T20-Effekt 11.81, n=1)
- P(0) in [0.25, 0.64] | P(1) in [0.00, 0.22] |
  P(4) in [0.00, 0.30] | P(24) in [0.00, 0.25]
  (T23 frisch, n=1 je Stufe — R8; P(0): T22 10-Block)

## L5-Serien-Zettel (Durchlauf 4)
Quelle: Maschine (Generator v1.3). Zweiter R8-Band-Einsatz;
erste Doppelklausel der Serie (Schwelle + 3-SE gegen im Lauf
gemessene Streuung, weil DM-SE ungetestet ist — R1-Luecke
dokumentiert, nicht vertuscht).

## Kernel-Bilanz (Stand, unveraendert bis T25-Verdict)
10 kernel-generierte Steine (T15..T24). T25 ist der erste
Stein der Linie SCHALTER mit einem diagnostischen (nicht
kausalen) Verdikt-Mass — der Uebergang von Phaenomen- zu
Werkzeug-Frage.



---

# Nachtrag 46 (2026-11-12): T25 VERDICT — PASS: ORDNUNGSPARAMETER-BESTAETIGT (dmdrop=11.65, ~55 SE). Mit Ehrlichkeits-Befund: DM trennt EBENEN, nicht einzelne Seeds. Zweiter interventionsfreier Stein in Serie.

## Lauf
Run 08:07-08:29 UTC (22 Runner-Minuten), Bot-Commit 50656f2.
100/100 Runs, Sanity clean (HIGH48 pf=1.00), ndiv=0 ueberall,
colnorm exakt, keine Reparatur — nach T24 der ZWEITE komplett
interventionsfreie Stein in Serie. R2: dm/dmdrop/dmeff an
JEDEM Arm, Lueckenliste LEER.

## Zahlen (Bot, gefroren)
DM-Mittel je Stufe (je 20 frische Seeds):
  a=0: 13.0614 (SE 0.211) | a=1: 1.4126 (SE 0.012)
  a=4: 1.3333 (SE 0.011) | a=24: 1.5898 (SE 0.015)
dmdrop = 11.6488 | SE(dmdrop) = 0.2113 | 3*SE = 0.6338
Doppelklausel erfuellt (>= 5.0 UND >= 3*SE mit Faktor ~55).
dmeff = 11.6488 >= 5.0 -> PASS. P-Stufen: 0.55/0.05/0.05/0.10.
L5-BAND (R8): 7/7 TREFFER.

## Lesart
1. DM KOLLABIERT EXAKT MIT DEM SCHALTER. a=0: 13.06, ab a=1:
   1.33-1.59 — Plateau, gekoppelt an die Konjugationstiefe,
   nicht an eine Dosis (identische Form wie die P-Kurve in
   T23). Der Ordnungsparameter faellt mit dem Schalter
   ZUSAMMEN, auf ~55 SE. DM ist damit als EBENEN-Marker des
   Symmetriebruchs bestaetigt: Diagonal-Dominanz = wie stark
   die Basis an den tanh-Achsen klebt.
2. EHRLICHKEITS-BEYFANG (Per-Seed-Bericht, registriert als
   Diagnostik ohne Verdikts-Gewicht): DM trennt innerhalb
   des a=0-Blocks NICHT: DM gelernt 13.133 (n=11) vs DM
   gescheitert 12.974 (n=9) — Differenz 0.16 < SE 0.21.
   Der Per-Seed-Teil der registrierten Frage ist damit
   VORBEIGEANTWORTET, NEGATIV: DM ist ein makroskopischer
   Ordnungsparameter (unterscheidet konjugiert vs intakt),
   KEIN Seed-Praediktor. Der in Nachtrag 45 projektierte
   Folge-Stein 'DM-Schwellwert-Kippregel pro Seed' entfaellt
   aus eigener Kraft — die Frage 'per Seed' braucht ein
   anderes Mass.
3. MECHANISMUS-ANSCHLUSS: DM bleibt keine Ursache (T20-R2
   KILL DM-NEBENSPUR unangetastet) — aber die diagnostische
   Seite ist jetzt bestanden: Der Schalter hat ein
   messbares, billiges Vorhersage-Instrument auf Ebenen-
   Niveau (DM aus W allein, vor jedem Training).
4. SITUATION DER LINIE SCHALTER: Existenz (T23), Richtung
   (T24), Ordnungsparameter Ebene (T25) — drei bestaetigte
   Stufen. Offen: Per-Seed-Ordnung (gibt es UEBERHAUPT eine
   Groesse, die innerhalb eines Blocks gelernt vs
   gescheitert trennt — Kandidaten: Gain-Verteilung,
   DM-Varianten, oder ist das Intra-Block-Lotto rein
   unveraenderlich?).

## L5-Serien-Zettel (Durchlauf 4, Quelle MASCHINE)
Maschine 7/7 in diesem Lauf; Serie Maschine gesamt 19/20
(T23 5/6, T24 7/7, T25 7/7). LLM 3/3. Gesamtkalibration
22/23 — die einzige Verfehlung bleibt drop124 (duenne
Quelle). R8-Bilanz nach zwei Einsaetzen: 14/14.

## Kernel-Bilanz (ehrlich)
11 kernel-generierte Steine (T15..T25). 8 saubere Verdikte.
Zwei interventionsfreie Steine in Serie (T24, T25) — die
R2-Readout-Schleife als Maschinen-Pflicht laesst die
T21/T23-Fehlerklasse dauerhaft leer. Die Doppelklausel
(dmeff) hat sich als Encoding bewaehrt: Verdict-Mass,
Schwelle und SE-Klausel in einem einzigen Kernel-Praedikat,
ohne Nachverhandlungsspielraum.


---

# Nachtrag 47 (2026-11-12): T26 PER-SEED-ORDNUNG REGISTRIERT (L5-Durchlauf 5, Quelle Maschine/Generator v1.4). Band und Schwellen VOR dem Lauf eingefroren.

## Registrierung
Vorschlag: P4-Generator v1.4 (Proposal v1.4, Bot-Commit 575cf17,
12:24 UTC, Score 0.78 — Ueberraschung 1.0, Kostenquote 0.37,
Scheidung 1.0). Erster Generator-Vorschlag auf Basis der
T25-Frischdaten inkl. Ehrlichkeits-Beyfang. Generator-
Historie des Tages, ehrlich: ZWEI Gate-Faenge vor dem ersten
erfolgreichen Lauf (R1-Text: fehlendes Anfuehrungszeichen +
falsch gesetztes Prozent-Format; dann these-String: fehlende
Quote hinter T27). Beide vom py_compile-Gate abgefangen, BEVOR
der Generator lief — genau der Sinn des Gates (Kontingent-
Schonung), aber zwei toete Runs sind zwei toete Runs.

## Frage
Trennt UEBERHAUPT eine messbare Groesse gelernte von
gescheiterten Seeds INNERHALB eines Blocks? T25-Beyfang:
DM gelernt 13.133 (n=11) vs gescheitert 12.974 (n=9),
Differenz 0.159 < SE 0.21 — DM ist kein Trenner. Kandidaten:
DM, Initial-Loss, Gain-Mittel, Gain-Std, Weight-Norm, colnorm.

## Design (eingefroren)
EINE Zelle (H=48, s=2.2, wb=0.6, T20-Basisparameter).
Stufen a in {0, 1} (Konjugation Q_a W Q_a^T, Q_SEED=2020),
je 1 frischer 20er-Block:
  a=0: 850..869 | a=1: 870..889
Anchor HIGH48 (s=5.5, a=0): 890..909. Total 60 Runs,
Fix 5.0 + variabel 17.0 = 22.0 min (Cap 60). R7: a=1 ist die
Replikations-Stufe (Richtungs-Konstanz des Markers), a=0 die
Klammer (T24-P(0)).

## Masse (R2: an JEDEM Arm inkl. Anchor)
pf, colnorm, ndiv, alle Kandidaten je Seed; Trennung gelernt
vs gescheitert je Kandidat je Stufe. Schwellwerte je Kandidat
WERDEN VOR DEM READOUT fixiert (Doppelklausel).

## Verdict-Schwellen (unantastbar)
- PASS — INTRA-BLOCK-MARKER: mindestens EINE Kandidat-Groesse
  trennt gelernt vs gescheitert im a=0-Block mit
  >= 3 * SE(im Lauf gemessen), Schwellwert VOR dem Readout
  fixiert, UND gleiche Richtung im a=1-Block UND Sanity clean.
  Konsequenz: T27 Kippregel pro Seed (Vorhersage vor dem
  Training).
- KILL — INTRA-BLOCK-LOTTO: keine Kandidat-Groesse trennt
  >= 3 * SE. Konsequenz: Per-Seed-Frage NEGATIV geschlossen;
  DM bleibt Ebenen-Marker; Linie SCHALTER: neue offene Frage
  aus Geometrie/Replikation (Gain-Verteilung als Ebenen-
  Kandidat).
- UNENTSCHEIDEN: Crash (Sanity Anchor pf < 0.5, ndiv > 6 je
  Block, colnorm-Verletzung). R5: EINE Reparatur, dann Linie
  tot. R4 erfuellt.

## Vorhersageband (mechanisch, R8-Haertung an duennen Quellen)
- DM gelernt in [8.80, 19.70] (T25 n=11, +50% Weite)
- DM gescheitert in [8.60, 19.50] (T25 n=9, +50% Weite)
- Differenz DM in [0.11, 0.24] (T25-Beyfang 0.159, nicht
  trennend — das Band ist bewusst ENG um NULL: Erwartung ist
  kein Trenner)
- P(0) in [0.25, 0.64] | P(1) in [0.00, 0.22]
  (T22 10-Block / T23 frisch n=1 — R8)

## L5-Serien-Zettel (Durchlauf 5)
Quelle: Maschine (Generator v1.4). Erster Stein, dessen Frage
direkt aus einem Ehrlichkeits-Beyfang (T25 Per-Seed-NEGATIV)
geboren wurde — die Maschine hat den negativen Befund
verarbeitet, statt ihn zu verbergen.

## Kernel-Bilanz (Stand)
11 kernel-generierte Steine (T15..T25), 8 saubere Verdikte.
Neu in der Bilanz: die Generator-Reparaturkette (2 Gate-Faenge
am selben Tag) — das Gate hat beide vor Runner-Kosten gefangen,
die Fehlerklasse "Syntax in generiertem Code" bleibt damit im
Kernel sichtbar und teuer-im-Kontingent-nicht.


---

# Nachtrag 48 (2026-11-12): LINIE TRAIN eroeffnet — T27 SCHALTER-TRAIN REGISTRIERT (erster Gradienten-Stein). Band und Schwellen VOR dem Lauf eingefroren.

## Motivation (ehrlich)
Der Kernbefund der Saison (Seed-Lotto = Koordinaten-Artefakt
des tanh-Ankers; Schalter bei einer rotierten Dimension; DM
als Ebenen-Marker) ist bisher NUR an eingefrorenen Reservoirs
bewiesen. Die offene Bruecke: Existiert der Schalter auch in
einem GRADIENTEN-TRAINIERTEN Netz? Das ist die Designluecke
der T-Serie — und sie ist im 40-EUR-Rahmen schliessbar:
H=48-MLP, 600 Steps, CPU-Rechenminuten.

## Recon-Positionierung (2026-11-12, vor der These)
Kein Doppel gefunden. Naechste Nachbarn: (1) Omnigrok (ICLR
2023) — Init-SKALA/Norm als Grokking-Ursache, nicht die
Relation Basis-vs-tanh-Achsen; (2) Noether-Learning-Dynamics —
Skalen-Symmetrien in der Lern-Dynamik, kein binarer Schalter
durch Koordinaten-Rotation gegen die Aktivierungs-Achsen;
(3) Geometric-Continuity (arXiv 2605.04971, bekannt aus N45) —
MLP/Transformer, Kontinuum, nicht rekurrent. Die Frage
"toetet EINE rotierte Dimension das Lernen in einem TRAINIERTEN
tanh-Netz — und kollabiert ein DM-Analog parallel?" bleibt
nach unserer Recherche unbeantwortet.

## Frage (T27)
Kippt das GRADIENTEN-Training dasselbe Phaenomen wie das
eingefrorene Reservoir: P_lernt kollabiert bei a >= 1
(Konjugation Q_a W_init Q_a^T) gegenueber a=0 — und folgt
ein DM-Analog dem Kollaps?

## Design (eingefroren)
MLP (nicht rekurrent — bewusste Vereinfachung, Transfer
zuerst am billigsten Objekt): H=48, tanh, delayed recall
(t_seq=24), Adam lr=5e-3, 600 Steps, batch=32. Bias der
Input-Projektion NULL (G5-Audit-Lektion: versteckter Bias-
Lotto). Init: W = Q_a T Q_a^T, T aus dem Seed, Q_SEED=2020 —
exakt die T20/T25-Konstruktion.
Stufen a in {0, 1, 24}, je 1 frischer 20er-Block:
  a=0: 910..929 | a=1: 930..949 | a=24: 950..969
R6-Sanity: der a=0-Block ist Anker — P(0) muss im Band
[0.25, 0.64] liegen (T22 10-Block-Statistik), sonst
UNENTSCHEIDEN. R7: a=1 ist die Replikations-Stufe (Reservoir-
Niveau T23/T25), a=24 die Tiefe, a=0 die Klammer.
Kosten-Schaetzung: 60 Trainings, ~2-4 s/Run CPU = variabel
~4 min + Fix 5.0 = ~9 min (Cap 60, bewusst weit drunter —
Compute-Disziplin: Mini-Lab, keine Verschwendung).

## Masse (R2: an JEDEM Arm)
pf (Task-Performance nach Training), Initial-Loss, Final-Loss,
DM_init je Seed (T20-Definition am INITIALISIERTEN W),
DM_final je Seed (am TRAINIERTEN W), colnorm. Trennung
gelernt vs gescheitert je Kandidat je Stufe (Anschluss an
T26: falls T26 einen Intra-Block-Marker findet, wird er hier
mitgeprueft).

## Verdict-Schwellen (unantastbar)
- PASS — SCHALTER-TRAIN: P(1) <= 0.15 UND P(24) <= 0.15 UND
  P(0) - P(1) >= 3 * SE(im Lauf gemessen) UND P(0) im Band.
  Konsequenz: Bruecke Reservoir -> trainiert BESTAETIGT;
  naechster Stein: Ordnungsparameter im trainierten Netz
  (DM_final-Analog).
- KILL — TRAIN-NETZ-RESISTENT: P(1) > 0.30 ODER P(24) > 0.30
  ODER Differenz < 3 * SE. Konsequenz: Der Schalter ist eine
  Eigenschaft des Eingefroren-Reservoirs — Grenze des Befunds
  praecise gezeichnet (das ist ein WERTVOLLER KILL: er sagt,
  wo die Theorie endet).
- TEIL-VERDICT (registriert, keine Nachverhandlung): P kollabiert
  (PASS-Kriterien P), aber DM_final trennt die Stufen NICHT
  (Differenz < 3 * SE) -> SCHALTER-OHNE-ORDNUNGSPARAMETER:
  der Schalter transferiert, der Marker nicht.
- UNENTSCHEIDEN: Crash (P(0) ausserhalb Band, colnorm-
  Verletzung, Trainings-Divergenz > 10 Prozent der Seeds).
  R5: EINE Reparatur, dann Linie tot. R4: PASS/KILL/TEIL
  fuehren zu verschiedenen naechsten Steinen.

## Vorhersageband (mechanisch, R8-Haertung an duennen Quellen)
- P(0) in [0.25, 0.64] (T22 10-Block — Sanity, kein Verdikt-
  Mass)
- P(1) in [0.00, 0.30] (Reservoir: 0.025-0.05, T23/T24/T25;
  +50% Weite und Transfer-Puffer: das trainierte Netz KOENNTE
  resistenter sein — das Band erlaubt es dem Befund zu
  bestehen, ohne ihm die PASS-Schwelle zu schenken)
- P(24) in [0.00, 0.30] (Reservoir: 0.10-0.15 — R8)
- DM_init(0) in [8.80, 19.80] (T25 n=11 — R8)
- DM-Analog: SE ungetestet im Trainierten -> DOPPELKLAUSEL
  (Schwellwert VOR dem Readout fixiert + 3 * SE gemessen),
  Lehre aus T25.

## Kernel-Bilanz (Stand)
11 kernel-generierte Steine (T15..T25), 8 saubere Verdikte.
T27 ist der erste Stein AUSSERHALB des Reservoir-Regimes.
Generator-Lehre vom Tag (2 Gate-Faenge, N47) eingebracht:
py_compile-Probe auf JEDEM neu generierten Instrument vor dem
Push, Quote-Bilanz je String-Literal, Klammer-Bilanz je
Funktion — die Fehlerklasse ist bekannt, die Proben stehen
jetzt VOR dem Push, nicht im Run.
