# Papier-Skeleton (Arbeitstitel)

## "The Optimizer Eats Its Own Memory"
### Learning erodiert die dynamische Ressource, auf der Lernen und Erinnern beruhen — Messung, Mechanismus und ein Regler dagegen

*Status: Skeleton v1 (2026-10-14). Jede Behauptung ist getaggt:
[E+Fxx] = eigene Experiment-Linie (laeuft/gehaertet/pending),
[LIT] = externe Quelle, [THESE] = noch ungetestet (F-Nummer dahinter).
Ziel-Format: Workshop/Conference-Paper (8-10 Seiten) + Tech-Report
mit voller Tiefenanalyse (Runden I-VII) als Anhang/Supplement.

---

## Abstract (Entwurf, 150 Woerter)

Recurrent networks store information not only in weights but in
their driven dynamics — a resource with a two-dimensional operating
window (driven Lyapunov exponent × non-normality kappa) that lies
strictly below the edge of chaos [E:F12, F18c]. We present evidence
that this resource is consumed by the very process that should build
it: gradient-based training [E:F16]. Three literatures describe the
symptoms of this erosion in four languages — loss of plasticity
[LIT: Dohare/Sutton], catastrophic forgetting as singular-vector
rotation [LIT: SVD-studies], progressive sharpening at the optimizer's
own edge of stability [LIT: Cohen], and kappa-drift in recurrent
networks [E:F16] — none connect them, because the connecting object,
the non-normality of the update operator itself [LIT: 2605.23476],
is invisible to the symmetric (Hessian) analysis each field uses.
We measure the erosion [E:F27], test its causal direction in a
rejuvenation duel at equal perturbation norm [E:F28], and close the
loop with a homeostatic regulator acting in the resource's own
coordinates [E:F21/F22]. [THESE — Gesamtkette, haelt nur, wenn F27/F28
positive.]

---

## 1. Introduction: Drei Faehigkeiten, eine fehlt
- E1 lernen (industrialisiert), E2 in der Dynamik erinnern
  (Mamba/S4-Mode), E3 den Betriebszustand als RESSOURCE verwalten
  (fehlt als Architekturprinzip). [E: alle]
- Der Satz des Papiers: "Das Lernverfahren ist der Antagonist des
  Gedaechtnisses" — nicht rhetorisch, sondern messbar in zwei
  Experimenten [E:F27, F28].
- Beitrag 1: Das 2D-Fenster (lambda getrieben x kappa), subkritisch
  [E:F12, F18c, F24 pending; LIT: Schuecker, Ganguli].
- Beitrag 2: Optimierer-Ressource-Interferenz als Mechanismus
  [E:F27 pending; LIT: 2605.23476 als Operatoren-Grundlage].
- Beitrag 3: Homeostase in Ressourcen-Koordinaten statt Zufalls-
  Injektion [E:F21/F22/F28; LIT: Sutton als Duell-Gegner].

## 2. Background & Related Work (Kollisions-Karte)
- 2.1 Edge of Chaos / Lyapunov Learning (ICML 2025): Ziel lambda->0.
  Unsere Evidenz: subkritisch [E:F12, F20b]. 
- 2.2 Kortex-Kritikalitaet: reverberating, m in 0.94-0.998 [LIT:
  Review; E:F25 pending]. Quasi-Kritikalitaet als Linie, nicht Punkt.
- 2.3 Nicht-Normalitaet als Feature: Ganguli 2008 (Fisher-Kurve),
  Orhan/Pitkow 2019, Wainrib/Galtier/Schuecker (expandiv-stabiles
  Regime = Regime B) [E:F18c; F24 pending].
- 2.4 Plasticity/Forgetting: Dohare/Sutton (random reinit), LLM-SVD
  (Singulaervektor-Rotation), Sculpting Subspaces [alle LIT; unsere
  Bruecke: F16, F27, F28].
- 2.5 Edge of Stability: Cohen 2021, Arora 2022 — die OPTIMIERER-
  Kante; Bruecke zu nicht-normalen Update-Operatoren [LIT:
  2605.23476]. Zwei Kanten, ein Antrieb [THESE, F27].

## 3. Das Fenster: Messung der Ressource
- 3.1 Getriebener Benettin als Messgeraet (F11-Lehre: SVD misst
  Artefakte) [E:F11].
- 3.2 MC-Optimum subkritisch bei ben ~ -0.33 [E:F12].
- 3.3 Zwei-Regime-Gesetz: T0 pur bei kappa<5, +const bei kappa>=5
  [E:F18c]. Numerische Abszisse omega widerlegt [E:F18e].
- 3.4 F24: MC vs kappa bei FESTEM lambda (Bisektion) — Kollision mit
  Ganguli/Orhan/Schuecker [E:F24 pending]. Vorhersage: MC steigt mit
  kappa; sonst war F12 Kovariations-Artefakt.
- 3.5 F25: m als Regime-Koordinate (lam-Sweep gegen Beiprodukt-Falle),
  Quasi-Kritikalitaet (tauS, tauT, gamma doppelt) [E:F25 pending].

## 4. Der Antagonist: Training erodiert das Fenster
- 4.1 kappa-Drift unter SGD (GRU +0.89) [E:F16].
- 4.2 Kein Gradienten-Pull zur Kante; lambda traege, startabhaengig
  [E:F20b]. H4-Optimum-These nach 8-Seed-Haertung GEFALLEN (p=0.732)
  — ehrlich als widerlegt berichtet.
- 4.3 F27: Optimierer-Ressource-Interferenz. P1 Adam>MGD>SGD
  (Drift), P2 Drift ~ kappa(V_opt), P3 drift auch bei Rausch-Targets
  [E:F27 pending]. Mechanismus-Kandidat: transiente Verstaerkung
  des nicht-normalen Update-Operators wirkt selektiv in Schur-
  Richtungen [LIT: 2605.23476; THESE: die Kopplung ist unsere].
- 4.4 Die vier Sprachen der Erosion (Synthese-Tabelle aus Runde VII):
  Diversitaetsverlust | Singulaervektor-Rotation | Progressive
  Sharpening | kappa-Drift — ein Prozess, vier Beschreibungen.

## 5. Das Duell: Gezielte vs zufaellige Restaurierung
- 5.1 F28-Design: 10-Block-Serie, Erosion messen (Adaptionszeit),
  dann Intervention bei IDENTISCHER Frobenius-Norm [E:F28 pending].
- 5.2 P2 ist der Scharfrichter: targeted > random => die Richtung
  ist die Information; Suttons 'Zufall noetig' wird ersetzt.
  Faellt P2 => These faellt, ehrlich berichten.
- 5.3 P3: Praevention (Homeostat) schlaegt Reparatur — die Bau-
  Empfehlung fuer E3.

## 6. Der Regler: Homeostase in Ressourcen-Koordinaten
- 6.1 F21: Block-Homeostat (Benettin alle 50 Steps, Penalty auf
  (ben-target)^2), Arme A/B/C/C0 [E:F21 pending]. C0 (target 0,
  ICML-These) gegen C (target -0.3, unsere) = direkter Design-
  Wettkampf.
- 6.2 F22: kontinuierlicher Gram-Proxy-Regler (K=8, QR-frei) —
  Schnelligkeit ist Pflicht [LIT: PLOS 'homeostasis must be fast'].
- 6.3 F23: Marathon (10 Bloecke, Retentions-Kurve) — der AGI-relevante
  Showdown: Anti-Forgetting als Nebenprodukt von E3, ohne Replay.
- 6.4 Kosten: Overhead-Messung; Versagens-Regime bei starker
  rekurrenter Erregung ist HSE-erwartet, kein Bug [LIT: Turrigiano/
  HSE].

## 7. Diskussion: Reichweite und Grenzen
- Was wir NICHT behaupten: keine AGI, keine Skalierungs-Behauptung
  ueber hidden=32-64, keine Behauptung ueber Transformer (nur
  Uebertragungs-Hypothese via Singularvektor-Literatur).
- Null-Evidenz-Register: F20b-H4 gefallen (p=0.732), F18e widerlegt,
  F19 widerlegt (1.833), Kovarianz-Propagation mit +I-Injektion
  gescheitert — das Papier zeigt seine Leichen, das ist die
  Glaubwuerdigkeit.
- Kritik-Vorwegnahme: (a) MC vs kappa koennte lam-Kovariation sein
  (->F24), (b) m koennte Latent-Variablen-Beiprodukt sein (->F25
  lam-Sweep), (c) targeted rejuvenation koennte schlicht bessere
  Regularisierung sein (-> Norm-Kontrolle eingebaut).
- Zukunft: F26 strukturierte Nicht-Normalitaet (Band-Kette vs
  Rauschen im Schur-Delta — 'so baut man sie').

## 8. Anhang / Supplement
- Tiefenanalyse-Runden I-VII als hergeleitete Related-Work-Map.
- ROADMAP.md-Befunde (F11-F20b) als Evidenz-Register.
- Reset-Day-Protokoll (Reproduzierbarkeit der Laeufe).

---

## Lauf-Zuordnung (wer liefert welchen Satz)
| Satz im Abstract | Experiment | Status |
|---|---|---|
| 2D-Fenster subkritisch | F12, F18c | gehaertet |
| kappa zahlt bei festem lambda | F24 | pending (1.11.) |
| m ist Regime-Koordinate | F25 | pending |
| Optimierer frisst Ressource | F27 | pending |
| Gezielt schlaegt Zufall | F28 | pending |
| Homeostat haelt Fenster | F21, F22 | pending |
| Praevention schlaegt Reparatur | F28-D, F23 | pending |

## Kill-Kaskade (was faellt zuerst bei welchem Ausgang)
- F24 negativ => Beitrag 1 schrumpft auf F12-Linie; kappa-These
  stirbt, Fenster bleibt 1D (lambda). Papier ueberlebt.
- F27 negativ => Beitrag 2 wird zu 'Drift ohne Mechanismus';
  vierfache-Tod-Synthese nur als Analogie. Papier ueberlebt
  abgeschwaecht.
- F28-P2 negativ => Beitraege 2+3 faellen ZUSAEMMEN; Rest ist
  'interessantes Fenster + kritische Literatur'. Ehrliches
  Workshop-Paper statt Nature-Anspruch. Auch okay.

## Titel-Alternativen
1. "The Optimizer Eats Its Own Memory" (mutig)
2. "Subcritical by Design: Homeostasis of the Driven Operating
   Point in Recurrent Networks" (sicher)
3. "Four Deaths, One Erosion: Plasticity, Forgetting, Sharpness and
   Non-Normality as Resource Loss" (Synthese-fokussiert)
