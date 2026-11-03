# Tiefenanalyse XIII (2026-11-03): F30.3-PASS, F31 DMFT-Selbstschluss (Run 22/23), Literatur-Verortung

## Tageskern in einem Satz
Die Ressource ist task-validiert (F30.3: corr = +0.748 gegen vorab genagelte Schwelle +0.4), der
DMFT-Selbstschluss faellt genau dort, wo der Exzess lebt (F31 P2), und die Literatur-Recherche zeigt:
die Einzelsteine existieren, der geschlossene Kausalbogen aus Ressource, Antagonist, Regler und
Fluktuations-Evidenz gehoert niemandem sonst.

## 1. F30.3 — Ressource ist task-validiert (PASS)
- corr(MC, TASK_R2) = +0.748 (Schwelle +0.4, vorab registriert).
- Sekundaer-Vorhersagen trafen wie registriert: NLM +0.639, ben -0.707.
- Ressource ist optimizer-frei gemessen (kein Optimierer im Messpfad).
- F30-v2-Kill (-0.499) bleibt als Antagonisten-Kanal stehen: E3b als gemessenes Phaenomen
  ("der Optimierer verdunkelt die Ressource"), siehe Tiefenanalyse XII.

## 2. F31 — DMFT-Selbstschluss (Run 22 mit Bug, Run 23 mit Fix)
### Bug & Fix
Run 22: DMFT-Suche startete bei d=1 -> supra-linear -> sofort None bei 20/24 Konfigs.
Fix (Commit 637425a): bei Instabilitaet d*0.75 schrumpfen (max 40x), nach Loop
`if not converged: return None`. Run 23: 6/24 ok statt 4/24; s=0.8/lam=0.6 konvergiert neu.

### Run 23 Festresultate
- P1: d_gap mean 0.146 (s=0.0: 0.035, s=0.8: 0.373) — d_gap ist das Nichtlinearitaets-Meter.
- P2 GEFALLEN: |dmft-ben| mean 0.194 vs |open-ben| 0.095; DMFT besser nur 2/6.
  => Offenes Feld NICHT durch DMFT ersetzen: Fluktuationen tragen Information.
- P3: corr(MC_theorie, MC_gemessen) = +0.936.
- P4: Exzess E s=0.0 mean 1.17 vs s=0.8 mean 9.70; corr(log E, log g) = +0.984.
  VORBEHALT (Selbstkritik): Zoo-Rangfolge ist bei 6 ok-Konfigs nicht stabil
  (Run 22: kappa_cond +0.999; Run 23: delta_H +0.977, kappa_cond +0.841).
  Robust ist der Exzess selbst, nicht der beste Zoo-Vertreter.
- P5: g_opt monoton in beta (1.03 -> 1.56 -> 3.34 -> 4.87), Beta abhaengig, kein Landschafts-Eigentum.
- 18 NO: s=1.6 alles (ben>0, ehrliches None); s=0.8/lam>=1.0 (ben<0, aber kein stabiler
  Gauß-Fixpunkt trotz 40x Schrumpfen) — strukturelles, kein numerisches Ergebnis.

### Lesart
P2 ist kein Schaden, sondern die zweite Messung der These: ein Mittelwert-Schluss sieht den
Betriebszustand nicht. Das schärft F32: Homeostat v3 lebt auf offenem Tr(Sigma_emp), nicht auf
der DMFT-Zurueckfaltung.

## 3. Literatur-Verortung (Recherche 2026-11-03)
### Nachbarn
- DMFT des Lernens in RNNs (naechster Nachbar): "Learning-Induced Dynamical Transition in
  Recurrent Neural Networks", arXiv:2609.19288 — Nichtgleichgewichts-DMFT fuer die Lerntrajektorie
  selbst, Kollaps schneller Fluktuationen, Chaos->stabil. Behauptet quantitative DMFT-Gueltigkeit.
  UNSER GEGENPUNKT: F31-P2 zeigt, dass der Selbstschluss genau bei hohem Exzess versagt.
- "Predictable Mean-Field Chaos in Random Recurrent Neural Networks", arXiv:2606.08805 —
  Mean-Field-Fluktuationen tragen deterministische Struktur. Unterstuetzt unsere Sprache fuer P2.
- Edge-of-Chaos-Kritik: "Beyond the Edge of Chaos: Stability-Expressivity Transfer in Reservoir
  Forecasting", arXiv:2607.17909 — Forecasting-Optimum wird NICHT durch die Lyapunov-Kante
  vorhergesagt. Entlastet uns: Lyapunov allein ist nicht die Ressource.
- Metrik-Korrelation ist nicht neu: PLOS One 10.1371/journal.pone.0101792 (Lyapunov-Exponent und
  Kernel-Qualitaet korrelieren mit Leistung). Unsere +0.748 ist besser validiert
  (registriert, optimizer-frei, Sekundaer-Vorhersagen), aber die Korrelation per se ist kein Durchbruch.
- Optimierer/Chaos als Gradienten-Theorie: arXiv:2110.07238 (divergente Gradienten <-> groesster
  Lyapunov-Exponent freier Orbits), NeurIPS 2022 "Chaotic dynamics are intrinsic to NN training
  with SGD". Dort ist Chaos Symptom der Optimierungsdynamik — niemand misst Optimierer als
  Zerstoerer einer internen, aufgabenrelevanten Ressource der getriebenen Dynamik, niemand
  baut einen Regler dagegen.

### Die Luecke (unsere Alleinstellung)
Nicht die Einzelsteine, sondern der geschlossene Kausalbogen:
1. Ressource: fluktuationstragend, O(H^2) online messbar (Tr Sigma / kappa), vorab-validiert
   gegen Aufgabenleistung (F30.3 PASS).
2. Antagonist: gemessener kappa-Drift unter Adam (E3b, F30-v2-Kill, F27 zu feuern).
3. Fluktuations-Evidenz: DMFT-Insuffizienz (F31-P2-Fall) — Ressource lebt nicht in den
   Ordnungsparametern.
4. Regler: F14b pinben (Benettin-Regler schlaegt SVD-pin), F32 Homeostat v3 auf Tr(Sigma_emp).

### Strategische Konsequenz
Die 2609.19288-Fraktion ist nahe am Territorium, aber von der Theorie-Seite und ohne Antagonist/
Regler. Deswegen: F27 (Antagonisten-Messung) wasserdicht machen, F32 zuegig, und die
DMFT-Insuffizienz als direkten Gegenpunkt zur "quantitativen DMFT-Gueltigkeit" positionieren.

## 4. Naechste Schritte
1. F27 feuern (kappa-Drift SGD/Momentum/Adam, multi-seed, echte Memory-Regression).
2. Danach Feuerplan: F24 -> F28 -> F26 -> F25 (q-Sweep) -> F21 -> F22/F23 -> F11.2.
3. F32 skizzieren: Homeostat v3 auf Tr(Sigma_emp), O(H^2) online messbar, gegen den Antagonisten.
