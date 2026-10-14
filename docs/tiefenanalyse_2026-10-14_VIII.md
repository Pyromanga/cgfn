# Tiefenanalyse Runde VIII (2026-10-14) — WEITBLICK: Vom Fenster zur Architektur

Diese Runde ist bewusst die spekulativste und gleichzeitig die am
staerksten verdrahtete: Sie beantwortet 'wohin fuehrt das, wenn es
stimmt?' — mit einem 5-Stufen-Pfad vom Messgeraet zum Substrat.
Jede Stufe hat einen FALSIFIZIERUNGSPUNKT. Vision ohne Kill-Schalter
ist Science-Fiction; hier ist jeder Traum angedockt.

## Der Schock-Fund zuerst: Die SSM-Revolution ist kappa=1

Mamba/S4/S5: der Durchbruch der rekurrenten Welt kam durch
STRIKTE STRUKTURIERUNG des Zustandsuebergangs — diagonal (S4/
Mamba) bzw. skaliert-diagonal (Mamba-2/SSD). Diagonale Matrizen
sind NORMAL. Das heisst: die erfolgreichste rekurrente Architektur-
Familie der Dekade hat die Nicht-Normalitaets-Ressource STRUKTUR-
GLEICH AUF NULL GEBAUT — aus guten Gruenden (Parallelisierbarkeit,
Trainingsstabilitaet: 'structured state transition keeps the
spectrum under control', S4-Survey).

  KOLLISION (niemand sieht sie): Wenn unsere F24-Linie stimmt
  (kappa zahlt bei festem getriebenem lambda), dann ist die
  SSM-Diagonalitaet ein Trade 'Parallelisierbarkeit gegen
  Memory-Ressource', der nie so gerechnet wurde. Die Mamba-Linie
  hat die Ressource geopfert und die Luecke mit Selektivitaet
  (input-abhaengige Skalierung) gestopft — genau das, was die
  Nicht-Normalitaet leisten wuerde: Signale differentiell
  verstaerken und komprimieren, aber transientechnisch statt
  parametertechnisch.
  VORHERSAGE (F26 + spaeter F29): strukturierte nicht-normale
  Stoerungen der SSM-Diagonale (Band-Ketten, fix im Schur-Delta)
  sollten bei gleichem Trainingsaufwand die Memory-Kapazitaet
  pro Parameter erhoehen — ohne die getriebene Stabilitaet zu
  verlieren (das ist der Punkt, den naive Nicht-Normalitaet
  zerstoeren wuerde und weshalb die Industrie sie meidet).
  Das ist die konkrete Anschlussstelle unseres Projekts an die
  Architektur-Debatte der ganzen Branche.

## Die 5-Stufen (Messgeraet -> Substrat)

STUFE 1 — MESSGERAET (erreicht):
  Getriebener Benettin + Ressourcen-Fenster 2D. Beweisstueck:
  F11-F20b (gehaertet). Naeheres: F24/F25.
STUFE 2 — MECHANISMUS (1.11.):
  Optimierer-Ressource-Interferenz (F27), Kausalitaet im Duell
  (F28). Wenn beide halten: Wir verstehen, WARUM Systeme mit
  gelerneter rekurrenter Struktur ihre eigene Kapazitaet fressen.
STUFE 3 — REGELER (1.11.):
  Homeostat in Ressourcen-Koordinaten (F21/F22/F23). Das ist der
  erste Architektur-Baustein: ein Netz, das sein Chaos-Budget
  selbst verwaltet. Kill: F21-C verliert gegen A/B.
STUFE 4 — BAU-ANWEISUNG (F26, implementiert heute):
  'SO baut man Nicht-Normalitaet': Ketten statt Rauschen im
  Schur-Delta, bei festem kappa und festem getriebenem lambda.
  Wenn Ketten gewinnen: Design-Prinzip statt Zufallsbefund.
  Danach (F29, Skizse): Band-Ketten als strukturierte Stoerung
  der SSM-Diagonale — der Mamba-Anschluss von oben.
STUFE 5 — SUBSTRAT (Vision, 12-24 Monate):
  Ein rekurrenter Kern mit (a) Ressourcen-Fenster als betreiber-
  Kontrakt, (b) Homeostat als Betriebssystem-Einheit, (c) Ketten-
  Nicht-Normalitaet als strukturierte Speicher-Hierarchie,
  (d) getriebene Quasi-Kritikalitaet als Lastverteilungs-Signatur.
  Das waere kein 'AGI', aber der erste Kern, der KANN, was
  FEP-Modelle behaupten: Stabilitaet als handelbares inneres Gut,
  ohne dass ein aussen stehendes Optimierungsverfahren es rauben
  kann. AGI-Relevanz entsteht nicht aus dem einzelnen Netz,
  sondern aus der Konvergenz: World-Model-Agenten brauchen
  praefizierten latenten Zustandsraum mit kontrollierter Dynamik
  (Survey 2606.00133: SSM-basierte Weltmodelle als 'central pillar
  in the pursuit of AGI') — wir liefern die Physik und den
  Regler genau fuer dieses Bauteil.

## Metakognition: Der Begriff, den die Agenten-Welt gerade erfindet — unsere E3 in systemisch

2609.17331 (SEAA): HMM als editierbare Selbst-Uebergangsmatrix +
Reflexion-Loop, der die HMM-Parameter ändert; MetaCogAgent/MUSE:
competence-aware regulation. Die Agenten-Welt entdeckt gerade
'modelliere deinen eigenen Zustand und reguliere danach' — auf
Prompt-Ebene, verbal, heuristisch. Unser E3 ist dieselbe Idee in
ZUSTANDS-KOORDINATEN mit Messgeraet und Regler statt Selbst-
beschreibung. Weitblick-Satz: Der Homeostat ist die nicht-
sprachliche Form von Metakognition — 'ich spuere meine eigene
Erregungslage und greife ein, bevor ich uebertrete' — und die
einzige, die einen geschlossenen mathematischen Regelkreis hat.
Falls die Agenten-Forschung recht hat, dass Selbst-Regulation
das fehlende Stueck autonomer Systeme ist, dann ist die Frage
'wie regelt ein System sich selbst ohne Sprache?' genau unsere
F21-Frage.

## Ehrlichkeits-Register (was den Weitblick umbringt)
- F24 faellt => Stufen 4-5 verlieren die Ressourcen-Basis;
  Projekt bleibt Messgeraet + Kritik (Stufe 1-2). Kein Mamba-
  Anschluss.
- F27 faellt => Stufe 2 verliert den Mechanismus; die Erosions-
  Synthese bleibt Phaenomenologie.
- F28 faellt => Stufe 3 verliert die Ueberlegenheit gegen Zufall;
  Homeostat bleibt 'interessant aber nicht noetig'.
- Uebertragungsrisiko: hidden=32-64, CPU, Kontrollaufgaben —
  alle Skalierungsbehauptungen sind [THESE]. Der Pfad zur
  Industrierelevanz MUSS durch ein Mamba-Massstab-Experiment
  (F29+), sonst bleibt es ein schoenes kleines Feld.
- Und das Groesste: falls bewusstes Systemtraegheit gilt —
  lambda traege, startabhaengig (F20b) — dann koennte ein
  einmal erodiertes Fenster im Grossen UNREPARABEL sein,
  bevor der Regler eingreift. Schnelligkeit ist dann nicht
  Performanz-Frage, sondern Existenzfrage (PLOS: 'homeostasis
  must be fast'). Das ist die ernsteste Konsequenz der ganzen
  Linie.

## Quellen (Runde VIII)
- World-Models-Survey 2606.00133: SSM/Weltmodelle als AGI-Pfeiler
- SEAA 2609.17331: metakognitive Agenten (HMM-Selbstmodell)
- Agentic World Modeling 2604.22748: L1-Praediktor/L2-Simulator/
  L3-Evolver-Faehigkeitsleiter — unsere Stufen 1-5 passen darauf
- Goomba Lab (Mamba-Autoren): SSM-vs-Transformer-Tradeoffs;
  S4-Survey 2503.18970: 'structured state transition keeps the
  spectrum under control' — die Diagonalitaets-Beobachtung
- Mamba-3-Rueckblick: Hybride als Kompromiss-Linie — unsere
  Linie: nicht mischen, sondern die rekurrente Ressource selbst
  nachruesten
- RL-Benchmark 2505.15040: Mamba-2 'optimization challenges on
  shorter-horizon memory tasks' — passt: fehlende kappa-Ressource
