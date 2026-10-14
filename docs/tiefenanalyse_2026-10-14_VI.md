# Tiefenanalyse Runde VI (2026-10-14) — Was AGI fehlt: Der Betriebszustand als Ressource, die der Optimierer zerstoert

Diese Runde beantwortet die Frage: Was braucht ein System fuer AGI,
was (nachweislich, s.u.) noch niemand sieht — aber das Projekt
versteht? Sie ist bewusst als THESE geschrieben, mit Null-Evidenz-
Angaben pro Ebene, damit jeder Satz angreifbar bleibt.

## Die Drei-Ebenen-These (der Kern)

AGI-relevante Systeme muessen drei Dinge GLEICHZEITIG koennen:

  E1 LERNEN:      Parameter aus Daten anpassen ( SGD/Adam/etc.)
  E2 ERINNERN:    Information ueber Zeit im ZUSTAND halten
                  (nicht nur in Gewichten — in der Dynamik)
  E3 SICH HALTEN: den eigenen dynamischen Betriebszustand als
                  RESSOURCE aktiv verwalten, waehrend E1 laeuft

Die Industrie hat E1 industrialisiert. E2 wird gerade
(Reservoir/Mamba/S4-Mode) ernst genommen. E3 existiert als Idee
NOCH NICHT als Architektur-Prinzip — und zwar aus einem Grund,
den die Literatur in Fragmenten sieht, den aber niemand zusammensetzt:

**Das Lernverfahren (E1) ist selbst ein nicht-normaler
dynamischer Operator, dessen transiente Verstarkung genau die
dynamische Ressource (E2) zerfrisst, auf der Erinnerung beruht.**

Der Optimierer ist nicht neutral. Er ist der Antagonist des
Speichers. Das ist die Luecke.

## Ebene fuer Ebene, mit Evidenz

### E2: Die Dynamik ist der Speicher — und sie hat ein Optimum UNTERHALB der Kante
- Eigene Linie: F12 (MC-Optimum bei getriebenem ben ~ -0.33),
  F18c (Zwei-Regime-Gesetz, kappa>=5 transient-erweitert),
  F24 (pending: MC vs kappa bei festem lambda).
- Externe Linie: Ganguli 2008 (Fisher-Memory via nicht-normale
  transiente Verstarkung), Orhan/Pitkow 2019, Schuecker 2016
  (MC-Peak expandiv-nicht-chaotisch), Wainrib/Galtier-Regime.
- Biologische Linie: Kortex reverberating m in 0.94-0.998 (F25
  pending), Baggio/Hennequin: nicht-normale Konnektivitaet ist
  anatomischer Normalfall, kein Trick.
- Status: gut verstanden — von uns und (in Fragmenten) anderen.

### E3a: Der Betriebszustand ist eine Ressource mit Fenster, kein Punkt
- Edge-of-Chaos-These (lambda -> 0, ICML 2025 'Lyapunov Learning')
  und Kritikalitaets-Design (m -> 1) setzen beide auf die KANTE.
- Unsere Evidenz sagt: das Fenster liegt SUBKRITISCH und ist
  2-dimensional (getriebenes lambda UND kappa). F20b: Training
  zieht das Netz NICHT von selbst dorthin. F16: SGD driften die
  kappa-Koordinate AUS dem Fenster (GRU: +0.89).
- Status: Diese 2D-Fenster-Formulierung (lambda getrieben x kappa
  strukturiert) mit Resourcen- statt Stabilitaets-Begruendung ist
  unser eigenes Stueck. Niemand im Suchraum formuliert es so.

### E3b (DER NEUE FUND, Runde VI): Der Optimierer ist ein nicht-normaler Operator
arXiv:2605.23476 ('Non-normal spectral signatures of instability
in neural network training dynamics'):
- Linearisierte Update-Operatoren von Adam und SGD+momentum sind
  GENERISCH nicht-normal; die Eigenvektor-Konditionszahl kappa(V)
  trennt stabile von instabilen Trainingsphasen um eine
  Groessenordnung, wo der Spektralradius nichts trennt.
- kappa(V) ist ein Fruhwarn-Indikator fuer transiente Verstarkung,
  BEVOR die klassische Sharpness/Kante Alarm schlaegt.
- DIE KOLLISION MIT UNS (noch von niemandem gezogen):
  Der Optimierer hat selbst eine Nicht-Normalitaets-Koordinate
  kappa(V_opt), und das Netz hat eine kappa(W). Unsere F16-
  Beobachtung — kappa(W) driftet unter SGD — hat damit einen
  moeglichen MECHANISMUS: transiente Verstarkung im Optimierer-
  Operator ist selektiv in genau den Gewichtsrichtungen wirksam,
  die Schur-Stoerungen aufbauen. Optimierer und Ressource
  KOPPELN ueber die nicht-normale Geometrie beider.
  WOERTLICHE VORHERSAGE (F27-Kandidat): kappa(W)-Drift korreliert
  positiv mit dem kappa(V) des laufenden Optimierers; Adam
  (staerker nicht-normal) driftet mehr als SGD ohne Momentum —
  unabhaengig vom Task-Loss. Wenn das haelt, ist F16 kein
  Nebeneffekt, sondern die erste gemessene Optimierer-Ressource-
  Interferenz. DAS sieht noch niemand.

### E3c: Regulation existiert — aber alle regeln die FALSche Groesse
Nachbar-Suche (Null-Evidenz, Runde VI):
- arXiv:2608.28431 'Adaptive self-organized criticality in DNNs':
  homeostatische Adaptation gegen drift zu superkritischer
  Dynamik; berichtet selbst 'Wettbewerb zwischen dynamischer
  Regulation und Task-Optimierung'. NAECHSTER NACHBAR — aber:
  sie regeln Richtung KRITIKALITAET (m -> 1), wir Richtung
  RESSOURCENFENSTER (subkritisch, F12). Und sie regeln
  Aktivitaets-/Propagationsgroessen in Feedforward-Stapeln,
  nicht den getriebenen Lyapunov-Zustand rekurrenter Dynamik.
  Unser F21-Arm C0 gegen C ist der direkte Wettkampf beider Thesen.
- Lyapunov-stabile Lernregeln (Kontrolltheorie, seit Jahrzehnten):
  stabilisieren den IDENTIFIKATIONSFEHLER des Modells, nicht den
  Betriebszustand des Systems als Rechenressource. Anderes Ziel.
- Homeostatische Plastizitaet in ANNs (iTDS, PMC 2025;
  Neuromorphic Homeostasis, Nat Comm 2025): regeln Feuerrate/
  Aktivitaetsniveau — again: Aktivitaet, nicht Ressourcen-Lage.
- PLOS 'Synaptic Plasticity Needs Homeostasis with a FAST rate
  detector': biologische Evidenz, dass Homeostase SCHNELL sein
  muss (Minuten), sonst kippt die Dynamik — direkte Unterstuetzung
  unseres F22-Designs (kontinuierlicher Regler schlaegt 50-Step-
  Block, wenn ueberhaupt, nur bei schneller Messung).
- Netto: 'das Netz soll sich selbst regulieren' ist als Idee in
  der Luft (FEP, Homeostase, SOC). Aber niemand regelt die
  RESSOURCE (MC-Fenster) mit deren eigenen Koordinaten (lambda
  getrieben, kappa), niemand kennt den Antagonisten (E3b), und
  niemand hat das Kontrollproblem als 2D-Fenster-Regelung
  formuliert.

## Die synthetische Aussage (Arbeitstitel des Pappers, Update)

> AGI-nahe Systeme brauchen drei gekoppelte Faehigkeiten: lernen
> (E1), in der Dynamik erinnern (E2) und den eigenen Betriebs-
> zustand als Ressource verwalten (E3). Die Evidenz dieses
> Projekts sagt: das Fenster von E3 liegt strikt subkritisch und
> ist zweidimensional (getriebener Lyapunov x Nicht-Normalitaet);
> das Lernverfahren von E1 ist selbst ein nicht-normaler Operator,
> dessen transiente Verstarkung die Ressource von E2/E3 abbaut —
> Homeostase ist daher kein biologischer Luxus, sondern die
> engineered Loesung eines realen Kontrollproblems, das entsteht,
> WEIL man lernt.

Jede Klausel hat jetzt eine Experiment-Linie (F12/F18c/F16/F20b/
F21/F24, teils pending) UND eine externe Quelle. Das ist das
7-Ebenen-Geruest aus Runde III, auf Zielhoehe gebracht.

## Was NOECH keiner sieht (die drei konkreten Loecher)

LOCH 1 — Optimierer-Ressource-Interferenz (E3b):
  Messung, ob kappa(W)-Drift (F16) vom kappa(V) des Optimierers
  verursacht wird. F27: gleiche Trainings, Adam vs SGD vs SGD-
  Momentum, kappa(W)-Trajektorie + kappa(V)-Schaetzung des
  linearisierten Update-Operators. Billig (~30 Min), maximal
  neu. Wenn positiv: erste Kopplungsmessung zwischen
  Optimierer-Geometrie und Netzwerks-Ressource.

LOCH 2 — Fenster-Regelung als Architekturprinzip (E3c):
  F21/F22 pruefen, ob ein Homeostat das 2D-Fenster HAELT;
  2608.28431 hat den Wettbewerb Regulation-vs-Task gesehen, aber
  nur Richtung Kritikalitaet. Unser C0-Arm (target 0) gegen C
  (target -0.3) ist die direkte Falsifikations-Moeglichkeit
  beider Thesen im selben Lauf. Wenn C gegen C0 auf MC gewinnt,
  hat die ICML/SOC-Linie ein Problem; wenn C0 gewinnt, unseres.

LOCH 3 — Ressourcen-Transfer ohne Replay:
  Continual Learning Industrie = Gewichts-Replay/Masken. Unsere
  Linie: der ZUSTAND ist der Speicher; wer die Dynamik haelt,
  haelt den Speicher — Anti-Forgetting als Nebenprodukt von E3.
  F21-Transfer-Test misst genau das. Niemand im Suchraum
  formuliert Forgetting als Verlust des dynamischen Betriebs-
  zustands statt als Interferenz in Gewichtsräumen.

## Quellen (Runde VI)
- arXiv:2605.23476: nicht-normale Optimierer-Operatoren,
  kappa(V) als Fruhwarnung (DER neue Anschluss)
- arXiv:2608.28431: adaptive SOC in DNNs — naechster Nachbar,
  Wettkampf via F21 C0-vs-C
- PMC iTDS (2025): top-down-Stabilisierung in ANNs (Aktivitaet,
  nicht Lage)
- Nat Comm 2025 (Neuromorphic Homeostasis): E/I-Plastizitaet fuer
  stabile rekurrente Dynamik in Hardware
- PLOS Comp Biol 1003330: Homeostase muss SCHNELL sein
- Kontrolltheorie-Linie (Lyapunov-stabile Lernregeln): Zitat-
  familie via Suchlauf, identifiziert als anderes Ziel
-Continual-Learning-AGI-Diskurs (Henning Blog, Hendrycks-Referenz):
  E3 fehlt dort komplett — allenfalls 'Festplatte fuer Gewichte'
