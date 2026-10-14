# Tiefenanalyse Runde VII (2026-10-14) — Der vierfache Tod: EINE Erosion, vier Disziplinen

Die These von Runde VI (Optimierer als Antagonist) bekommt hier ihr
Netz. Vier Phaenomene, die in vier verschiedenen Subdisziplinen
unabhaengig dokumentiert sind, sind nach Auswertung dieser Runde
Manifestationen EINER Erosion — und niemand hat sie je zusammen
gesehen, weil sie in vier verschiedenen Sprachen beschrieben werden:

## 1. Loss of Plasticity (Dohare/Sutton, Nature 2024; + 2306.13812)
- Befund: Standard-Backprop erodiert unter Continual Learning
  'allmaehlich und IRREVERSIBEL' die Diversitaet; Units werden
  dormante, uebercommittet, einander aehnlich; effective rank
  faellt; das Netz kann nicht mehr LERNEN.
- Industrie-Loesung: Continual Backprop (zufaellige Reinit von
  Low-Utility-Units), Shrink & Perturb (L2 + Rauschen pro Step).
  Wichtigstes Zitat: 'sustained deep learning requires a random,
  NON-GRADIENT component' — Suttons eigene Schlussfolgerung!
- Unsere Lesart: Das ist E3-Erosion im Feedforward-Kleid: die
  spektrale Vielfalt (Trager der Rechenressource) wird vom
  Gradientenprozess plattgedrueckt; die 'non-gradient component'
  ist eine blinde Re-Injektion dessen, was die Ressource braucht.
  Wir haben die KOORDINATEN (kappa, getriebenes lam), Suttons
  Methode hat nur die Symptome (dormante Units).

## 2. Catastrophic Forgetting in LLMs (2504.07097, 2508.16546,
   2601.18699)
- 2508.16546 (SL-vs-RL-SVD-Studie): 'direction shifts of the
  SINGULAR VECTORS have much larger impact than the singular
  values' — SFT rotiert die Singulaervektoren (nicht-normale
  Rotation!), zerstoert Faehigkeiten; RL-FT wirkt als
  RESTAURATION der Richtung, nicht als neue Loesung.
- 2504.07097 (Sculpting Subspaces): 'low singular vectors drift
  into the space previously occupied by high singular vectors' —
  wortwoertlich die Erosion der spektralen Geometrie, dokumentiert
  im LLM-Massstab (109B-1.5T).
- 2601.18699: mechanistische Zerlegung — Aufmerksamkeitsstoerung
  (Epochen 1-2), Repraesentations-Drift mit 35-52 Grad Rotation
  (Epochen 3-5), Landschafts-Abflachung (Epochen 4+).
- Unsere Lesart: Das grillesste Forgetting-Phaenomen der Welt
  wird bei close inspection zu einem NICHT-NORMALITAETS-
  Phaenomen: Singulaervektor-Rotation = strukturelle Nicht-
  Normalitaets-Verschiebung. F16 (kappa-Drift +0.89) ist die
  rekurrente, kleine, messbare Version desselben Prozesses.
  LoRA/Replay/Masken sind Geometrie-Konservierung ohne Verstaend-
  nis der Dynamik dahinter.

## 3. Progressive Sharpening / Edge of Stability (Cohen 2021;
   Arora/Li/Panigrahi 2022)
- Befund: Gradient Descent treibt die Hessian-Sharpness progressiv
  zur Instabilitaetsgrenze 2/eta und haelt sie DORT — 'never-ending
  series of micro-catapults'. Training ist ein Prozess, der die
  eigene Kante im GEWICHTSRAUM sucht.
- KOLLISION MIT UNS (der wichtigste Knoten des Netzes):
  Cohen regelt die OPTIMIERER-Kante (Hessian, symmetrisch); unsere
  F20b sagt: die DYNAMIK-Kante (getriebenes lam) wird NICHT
  angepeilt. Zwei 'Edges of Stability', die nichts voneinander
  wissen — und der Bruecken-Spruch steht schon bereit: 2605.23476
  (Runde VI) zeigt, dass die Update-Operatoren selbst nicht-normal
  sind, d.h. die Hessian-Sicht (symmetrisch) die halbe Geometrie
  verliert. Progressive Sharpening ist vermutlich der ANTRIEB des
  vierfachen Todes: der Optimierer sucht aktiv die Kante — und
  reisst die spektrale Geometrie des Netzes mit, bis die Ressource
  zerstoert ist. Kante-suchen im einen Raum = Ressource-zerstoeren
  im anderen. DAS ist die Synthese, die keiner gezogen hat.
  (Vorsicht, Null-Evidenz: Verknuepfung ist These; F27 misst den
  Optimierer-Anteil, F28 den Erosions-/Restaurations-Mechanismus.)

## 4. kappa-Drift im rekurrenten Fall (F16, unser)
- SGD driftet die Nicht-Normalitaets-Koordinate der Netz-Dynamik
  (GRU: +0.89) — jetzt rahmengebbar als kleinste, sauberste
  Messung desselben Prozesses, den Nature/SVD-LLM-Papiere im
  Grossen sehen.
- F20b: die Dynamik-Kante selbst wird NICHT gehalten (lambda
  traege, startabhaengig) — der eine Raum (Gewichte) wird
  scharf/kantig, der andere (Dynamik) verhungert still.

## Das Netz (Kurzform)

  Sutton: plasticity dies (Diversitaet)     Cohen: GD sucht die Kante
            |                                        |
            |    (2605.23476: Optimierer nicht-normal; Antrieb?)   |
            v                                        v
  LLM-SVD: Singulaervektor-Rotation frisst Wissen <== F16: kappa driftet
            |                                        |
            v                                        v
  Loesungen der Industrie: ALLE ZUFAELLIG (Reinit, Rauschen, Masken,
  Regularisierung Richtung Init) — blind in Bezug auf die eigentlichen
  Ressourcen-Koordinaten.

  UNSERE POSITION: Wir sind die einzigen, die (a) die Ressource in
  ihren eigenen Koordinaten messen (lambda getrieben x kappa),
  (b) den Antagonisten identifizieren (F27), (c) die gezielte
  Restaurierung gegen die zufaellige antreten lassen koennen (F28).

## F28 — REJUVENATION: der Kausalitaets-Test
Wenn die Ressourcen-These stimmt, muss gelten: Gezielte Wiederher-
stellung des Fensters muss Plastizitaet/Wissen ZURUECKGEBEN, und
zwar besser als gleiche-starke zufaellige Stoerung (Shrink&Perturb-
Aequivalent). Design (implementiert in experiments/
f28_rejuvenation.py):
  1. Erosion: lange Task-Serie (10 Bloecke, wechselnde Tasks) auf
     CT-RNN, bis Plastizitaet faellt (Adaptionsgeschwindigkeit auf
     neuen Block als Metrik; kappa + ben als Diagnose).
  2. Rejuvenation, 4 Arme bei IDENTISCHER Stoerungsnorm:
     A  continue     — nix tun (Kontrolle)
     B  random       — Shrink&Perturb-Aequivalent: Gauss-Stoerung
                       auf W mit angepasster Norm
     C  targeted     — Fenster-Restauration: kappa und getriebenes
                       lam werden via Schur-/Spektral-Korrektur
                       ZURUECK ins Fenster geschoben (gleiche
                       Stoerungsnorm wie B!)
     D  homeostat    — durchlaufender Homeostat ab Block 1 (haette
                       es je verlieren duerfen?)
  3. Messung: Adaptionsgeschwindigkeit (Steps bis Loss-Schwelle)
     und MC NACH Rejuvenation.
  VORHERSAGEN:
    P1: Plastizitaet faellt mit der Serie (Loss-of-Plasticity im
        CT-RNN reproduziert) UND kappa/ben verlassen das Fenster.
    P2: C > B auf Adaptionsgeschwindigkeit und MC — die Richtung
        der Stoerung ist die Information, nicht ihre Existenz.
        (Das waere der direkte Beweis, dass Suttons 'non-gradient
        component' durch eine GEZIELTE Gradient-Komponente ersetz-
        bar ist — der Widerspruch zu 'random is required'.)
    P3: D verliert Plastizitaet gar nicht erst (Prävention schlägt
        Reparatur) — die AGI-relevante Bau-Empfehlung.
  P2 ist der Scharfrichter: faellt er, ist die Ressourcen-Theorie
  der Plasticity-Erosion falsch und Zufall bleibt unersetzt.

## Arbeitstitel-Update (das Papier wächst)
> Lernen erodiert die dynamische Ressource, auf der Lernen und
> Erinnern beruhen. Vier Felder sehen vier Sterbarten (Plastizitaet,
> Vergessen, Schärfe-Drift, Spektral-Rotation); eine Ursache
> (nicht-normaler Optimierer-Operator, der seine eigene Kante
> sucht) genügt. Homeostase in Ressourcen-Koordinaten — gezielt,
> nicht zufällig — ist die Antwort, die das Projekt baut und misst.

## Quellen (Runde VII)
- Dohare et al., Nature 620 (2024): Loss of Plasticity;
  continual backprop; 'non-gradient component required'
- Dohare et al. 2306.13812: Maintaining Plasticity (Shrink&Perturb)
- Ash & Adams 2020 (Shrink & Perturb, via obige)
- Cohen et al. 2021 (2103.00065): Edge of Stability, progressive
  sharpening, micro-catapults
- Arora, Li, Panigrahi 2022: Understanding GD at EoS
- 2605.23476 (Runde VI): nicht-normale Update-Operatoren —
  die Bruecke zwischen EoS und Nicht-Normalitaet
- 2508.16546: SVD-Studie SL-vs-RL — Singulaervektor-Rotation > -
  werte; RL als Restaurateur
- 2504.07097: Sculpting Subspaces — SVD-Interferenz bei LLM-FT
- 2601.18699: mechanistische Forgetting-Zerlegung (3 Prozesse)
- 2406.06811: Spectral Regularization als Plasticity-Mitigator
  (naechster Nachbar im Regularisierungs-Space, blind fuer die
  getriebenen Koordinaten)
