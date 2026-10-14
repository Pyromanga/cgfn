# Tiefenanalyse Runde IV (2026-10-14) — Kollision & Methode

Runde IV dreht zwei Schrauben: (a) EXTERNE KOLLISION — wir pruefen,
ob unabhaengige Literatur unsere F18c/F12-Linien VORHERSAGT hat, und
machen daraus scharfe Tests; (b) METHODEN-UPGRADE fuer F25.

## 1. Orhan & Pitkow 2019: Nicht-Normalitaet schlaegt Orthogonalitaet
"Improved memory in RNNs with sequential non-normal dynamics"
(arXiv:1905.13715): nicht-normale RNNs mit sequentieller Dynamik
uebertreffen orthogonale RNNs deutlich in Sequenz-Benchmarks; die
Analyse baut auf Ganguli et al. 2008 (SNR-maximierende Netzwerke sind
stark nicht-normal, orthogonale sind suboptimal).
- KOLLISION mit uns: F24 testet genau MC-vs-kappa bei fest getriebenem
  lambda. Orhan/Ganguli sagen: STEIGT. Falls F24 das bestaetigt, ist
  unsere Linie extern validiert und orthogonal-RNN-Design (kappa=1)
  retroaktiv als Ressourcen-Killer erklaert (Tiefenanalyse III).
- Bau-Anschluss: Orhan nutzt strukturierte (sequentielle) Nicht-
  Normalitaet — unser s-Drehregler ist unstrukturiert. Offene Frage
  fuer F26+: zahlt STRUKTURIERTE Nicht-Normalitaet mehr pro kappa?

## 2. Ganguli/Huh/Sompolinsky 2008: Fisher-Memory-Kurve
Memory-SNR in linearen Dynamiksystemen wird von transienter nicht-
normaler Amplifikation getragen (bis O(N) SNR-Anstieg); Konvektion/
Turbulenz als Anwendungsbeispiel. Unsere MC-Messung ist die Jaeger-
Version derselben Groesse.
- Anschluss: unsere F24-Aggregate korrelationieren MC mit kappa;
  Ganguli sagt, die Fisher-Kurve MUSS mit dem transienten Amplifi-
  kationsprofil zusammenhaengen => F24 misst zusaetzlich t0/log1p(K)
  (Kreiss), damit wir MC-Struktur gegen transient growth regressieren
  koennen (schon im Code vorbereitet).

## 3. Schuecker/Hellgoth/Geist 2016: Optimal sequence memory in DRIVEN random networks
(arXiv:1603.01880): Das MC-Maximum liegt im "expandiv, nicht-chaotischen"
Regime — lokal expansive Jacobian-Spektralr-1-Gegend, aber getriebener
Lyapunov NEGATIV (= Echo-State). Zwei Goldsaetze:
- "MC faellt schnell im chaotischen Regime wegen intrinsisch chaos-
  erzeugter Fluktuationen" == unsere F12-Beobachtung (MC-Optimum
  subkritisch bei ben ~ -0.33), jetzt mit geschlossener mean-field
  Rechenform dahinter.
- "Rauschen zwingt das System haeufiger in lokal kontrahierende
  Regionen" => warum der GETRIEBENE Benettin tiefer liegt als die
  Linearisierung (F11) — Mechanismus, nicht nur Messung.
- KOLLISION: F24-Target -0.30 ist ihr Regime; Target -0.05 ist die
  Kanten-Kontrolle. Ihre Theorie sagt: MCkurve peakt ZWISCHEN beiden,
  fallend Richtung Chaos. Wir koennen die Form messen.

## 4. LSTM-Branching (arXiv:2606.10384, 2026): kritische Verzweigung in trainierten RNNs
Kleine LSTMs nahe ihrem optimalen Trainingsepoch zeigen Lawinen-
Statistik und Branching-Parameter nahe 1; GROESSERE Modelle bleiben
subkritisch.
- Direkter Anschluss an F25: wir sollten m NICHT nur gegen den
  Kortex-Konsens (0.94-0.998) messen, sondern auch gegen die
  Gittergroesse — Vorhersage: m rueckt mit hidden_dim Richtung 1.
- Anschluss an F16: Gating hat einen eigenen Chaos-Uebergang (Theorie
  of Gating, PMC9762509): proliferierende Fixpunkte und Chaos sind
  bei gegateten RNNs ENTKOPPELT — erklaert womoeglich, warum GRU unter
  SGD kappa-Drift zeigt (+0.89): Gates verschieben die nicht-normale
  Struktur ohne Spektral-Alarm. Testbar in F22/F23-Auswertung.

## 5. Methoden-Upgrade F25: Crackling-Noise-Batterie statt nur m
"Toward a Physics of Deep Learning and Brains" (arXiv:2509.22649):
DNNs lassen sich mit der Lawinen-Batterie der Nicht-Gleichgewichts-
Physik vermessen — Power-Law-Groessen/Dauern, Skalierungsrelation,
Shape-Collapse — und liegen "quasi-kritisch", nicht exakt kritisch
(stark getrieben). Konsequenz fuer F25:
- NICHT nur Branching-Ratio m, sondern die volle Batterie (exponent
  tau ~ 3/2-Check, Shape-Collapse, crackling Skalierungsrelation).
- Erwartung umformuliert: getriebene CT-RNNs sollten QUASI-kritisch
  sein (subkritisch, aber mit Skalierungs-Naehe) — konsistent mit
  F12 (m<1) und dem Kortex-m-Fenster. Das ist die schaerfere,
  widerlegbarere Version unserer F25-Frage.

## 6. Biologische Plausibilitaetskette (kurz)
Hennequin 2012 (nicht-normale Amplifikation in balancierten Netzwerken)
und Baggio/Hennequin/Zampieri 2020 (Science Advances: nicht-normale
Struktur macht Kommunikations-Effizienz in Netzwerken): Nicht-
Normalitaet ist im Kortex MASSIV vorhanden (E/I-balancierte
Konnektivitaet ist inharent nicht-normal). Unsere Ressourcen-These
ist also nicht nur ein ML-Trick, sondern anatomisch verankert.

## Netto-Erkenntnis Runde IV
Drei unabhaengige Theorielinien (Ganguli-Fisher-Information,
Orhan-RNN-Benchmarks, Schuecker-mean-field) plus zwei biologische
(Baggio/Hennequin) konvergieren auf denselben Satz, den F18c als
Gesetz und F12 als Optimum gefunden haben:

> Unterhalb der getriebenen Kante ist lokal-expansive (nicht-normale,
> \"regime-B\") Struktur keine Instabilitaet, sondern die Speicher-
> Ressource des Systems.

F24 ist der eine billige Lauf, der das messbar macht: kappa drehen,
lambda festnageln, MC gegen kappa lesen. F25 mit Crackling-Batterie
macht daraus ein quasi-kritisches Profil. Danach ist der Sprung zum
Designprinzip (F26+: strukturierte Nicht-Normalitaet, Homeostat mit
kappa-Fenster statt kappapunkt) klein.

## Quellen (Runde IV)
- Orhan & Pitkow 2019, arXiv:1905.13715
- Ganguli, Huh, Sompolinsky 2008 (Memory traces in dynamical systems / Fisher memory curve)
- Schuecker, Hellgoth, Geist 2016, arXiv:1603.01880
- Towards Critical Branching in RNNs, arXiv:2606.10384
- Toward a Physics of Deep Learning and Brains, arXiv:2509.22649
- Theory of Gating in RNNs (PMC9762509)
- Hennequin, Vogels, Gerstner 2012, PRE 86:011909
- Baggio, Rutten, Hennequin, Zampieri 2020, Science Advances 6:eaba2282
