# Tiefenanalyse XIV (2026-11-04): Der F30.4-Kill, die Gräber der Anderen, und die FTLE-Wette

## 1. Der Kill (registriert, keine Nachverhandlung)
F30.4 Transfer-Kill-Test, vorab genagelt: TRANSFER-PASS nur wenn
corr(MC, TASK_R2) >= +0.4 in BEIDEN Bedingungen.
- Bedingung A (H=256, sinus_memory): +0.363 — GEFALLEN.
- Bedingung B (H=32, NARMA-10): +0.203 — GEFALLEN.
- VERDICT: KILL. F30.3 (+0.748) war ein Ein-Fenster-Resultat.

## 2. Forensik: Was die +0.748 wirklich war
- F30.3 (H=32): Die Korrelation wurde groesstenteils vom Chaos/Tod-Split
  getragen (s=1.6-Bloecke ben>0, TASK_R2 ~0.03-0.24; s=0.0 stabil,
  TASK_R2 ~0.32-0.44, MC hoch). MC kovariierte am kleinen Gitter
  zufaellig mit der Stabilitaet mit.
- F30.4 A: Split verschaerft (s>=0.8 stark chaotisch), corr +0.363,
  UND: ben schlaegt MC als Praediktor (-0.430 vs +0.363). Innerhalb
  des einzigen stabilen Blocks (s=0.0, 16 Konfigs) ist die MC-Task-
  Relation NEGATIV (MC faellt 2.5->1.4 mit lam, TASK_R2 steigt
  0.43->0.66). Innerhalb des arbeitenden Regimes zeigt MC in die
  falsche Richtung.
- F30.4 B: TASK_R2 ~0.5 ueberall (MC 2.9 oder 0.04, egal) — die zweite
  Aufgabenfamilie sieht die Ressource ueberhaupt nicht.
- Robust in ALLEN drei Laeufen: corr(ben, TASK_R2) < 0. Stabilitaet ist
  der einzige Transfer-harte Task-Praediktor.

## 3. Konsequenzen (ehrliche Abwertungen)
- TOT: "MC/kappa-Ressource sagt Aufgabenleistung vorher" als allgemeine
  Wette-Basis.
- ABGEWERTET: E3b als "Optimierer zerstoert eine task-relevante
  Ressource". F27s Messungen (kappa_T-Erosion um 5 Groessenordnungen,
  zweiphasig, Adam am tiefsten, Loss verbessert) bleiben deskriptiv
  gueltig, aber der Task-Bezug ist nicht gedeckt — im Gegenteil:
  F30.4-A zeigt MC-negativ im stabilen Block, Adam verbessert die
  Aufgabe. F32 (Homeostat auf Tr Sigma) als naechster Schritt GEPLATZT:
  ein Regler fuer eine Ressource, die Aufgaben nicht sehen, regelt
  einen Geist.
- UEBERLEBT: F24 (+0.624 bei FESTEM ben per Bisektion, kleines
  absolutes Mass), F27-Phaenomenologie (Zoo-Dynamik, Divergenzkante,
  Zwei-Phasen), F31-P2 (DMFT-Insuffizienz bei hohem Exzess),
  Stabilitaet als Praediktor.

## 4. Die Graeber der Anderen (Recherche 2026-11-04)
Unser Fehler ist kein Erstfund. Vier beschriftete Graeber:
1. "Do Reservoir Computers Work Best at the Edge of Chaos?"
   (Carroll, arXiv:2012.01409) — Edge-of-Chaos-Optimum verschaebt sich
   beim Transfer; zentraler Fund: Instabilitaet durch positive LOKALE
   Lyapunov-Exponenten bei negativem GLOBALEN.
2. "Beyond the Edge of Chaos: Stability-Expressivity Transfer in
   Reservoir Forecasting" (arXiv:2607.17909) — Forecasting-Optimum wird
   von KEINER Lyapunov-Kante vorhergesagt; Ziel wird getragen von
   einem breiten Band von Moden, stabil im Mittel, stark fluktuierend
   in endlicher Zeit.
3. "Memory of Recurrent Networks: Do We Compute It Right?"
   (JMLR 25/23-0568) + "Memory Capacity of Nonlinear Recurrent
   Networks: Is it Informative?" (arXiv:2502.04832) — MC-Schaetzung
   methodisch angreifbar; bei nichtlinearen ESNs ist MC nicht durch
   die Architektur festgelegt. MC war von vornherein ein wackliger
   Kandidat als harte Zahl.
4. "Chaos helps until it is trained" (SSRN 7434435) — Chaos-Kanten-
   Vorteil verschwindet unter End-to-End-Training; trainierte Modelle
   operieren on-task kontrahierend. = Unser F27-Befund, erklaert, warum
   die Ressource unter Training keine Rolle spielte.
   Zusaetzlich: PMC4092026 (2014) — von fuenf Reservoir-Metriken
   ueberleben nur Lyapunov und Kernel-Qualitaet den Benchmark-Transfer.

## 5. Die Loesung (abgeleitet statt geraten)
Drei Literatur-Linien plus unsere Daten konvergieren:
- Robuster Praediktor ist Stabilitaet, aber nicht als SKALAR, sondern
  als SPEKTRUM: die Verteilung der finite-time-Lyapunov-Modi
  (Carroll lokal-vs-global; 2607.17909 Modi-Band; EPJ B s10051-026-
  01204-4 FTLE als Werkzeug).
- Unsere eigene F18c fand "Regime B: lokal expansiv, asymptotisch
  stabil" — exakt Carrolls Befund. Wir hatten das Fenster schon,
  haben es dann mit skalaren Kills/Festen ueberbaut.
- Der saubere Test ist stability-gematcht, nicht grid-konfundiert:
  F24 (Bisektion auf festes ben) ist unser einziges unter Kontrolle
  bestandenes Design.

## 6. Neue registrierte Wette (F33, Skizze)
FRAGE: Ist die FORM des FTLE-Spektrums (Verteilung der endzeitlichen
Stabilitaetsmodi, nicht nur ben_max) der Betriebszustand, der
Aufgabenleistung vorhersagt?
DESIGN (F24-Muster):
- Stability-gematchtes Gitter: lam-Bisektion auf festes ben-Ziel pro
  (s, seed) — der Konfound ist kontrolliert, nicht nur gemessen.
- Messung: FTLE-Spektrum (Modi-Verteilung: Mittel, Streuung, Anteil
  positiver lokaler Moden, obere Quantile), plus MC/ben als Kontroll-
  Skalare, kappa-Zoo als deskriptive Kovariate.
- Aufgaben: sinus_memory UND NARMA-10, H=32 UND H=256 — Transfer von
  Anfang an Teil der Registrierung, nicht ein Nachtrag.
VORAB (bei Feuung zu nageln): Primaer-Kriterium FTLE-Feature-Set
(registrierte Auswahl VOR dem Lauf, kein Post-hoc-Zoo), Kontroll-
Kriterium: FTLE-Features schlagen ben_max in mindestens drei der vier
Bedingungs-Zellen. KILL-Konsequenz steht dann fest.
Kein Sekundaer-Vorhersagen-Nachrutschen; ein Ergebnis pro Stein.

## 7. Was nicht stirbt
Methoden-Repertoire (registrierte Vorhersagen, Snapshot-Patterns,
ehrliches None, Transfer-Tests von Anfang an), CI-Infrastruktur,
F24-Instrument (Bisektion), F27-Phaenomenologie, F31-DMFT-Befund.
Und die eigentliche Lektion: Der F30.4-Kill ist der wertvollste Output
des Projekts bis heute — er hat eine Woche Fundament-Arbeit vor einem
Gebaeude bewahrt, das auf einem Fensterputzer-Effekt stand.
