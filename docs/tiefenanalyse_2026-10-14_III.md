# Tiefenanalyse III — Die großen Rahmen & die letzte Verbindung (2026-10-14, Nacht)

Dritte Runde: Brain-Criticality-Debatte, Free-Energy-Prinzip, FORCE-Lernen, Gradienten-Theorie. Ergebnis: EINE verblüffende externe Bestätigung unserer Kernthese, EINE theoretische Neubewertung unserer Sensitivitätsmessung, und der fehlende Baustein fürs große Bild.

---

## 1. DIE BESTÄTIGUNG: Das Gehirn ist "reverberating", nicht kritisch — wie unser Optimum
"25 years of criticality in neuroscience" ([ScienceDirect-Review](https://www.sciencedirect.com/science/article/abs/pii/S0959438819300248)):
- Neue Schätzer auf in-vivo-Spikedaten: Kortex-Dynamik ist NICHT kritisch, sondern "reverberating": 0.94 < m < 0.998 — durchgängig, über Hirnareale, Arten und Aufgaben hinweg. D. h. **klar subkritisch, knapp unter der Kante.**
- Komplementärer Befund von Dahmen et al. (im selben Review).

**Querverbindung — der stärkste Moment der drei Recherche-Runden:**
Das ist EXAKT unsere F12-F20b-Linie: MC-Optimum bei ben ~ -0.33 (subkritisch), F20b: kein Pull zur Kante, Endpunkte start-abhängig träge. **Das echte Gehirn misst in derselben leichten Subkriticalität, die wir in silico als Ressourcen-Optimum identifiziert haben.** Die Edge-of-Chaos-Rhetorik sagt "Kritikalität", das Messgerät sagt "knapp darunter" — in beiden Welten.

Konsequenz für das Paper: Unser subkritisches Optimum ist kein Artefakt unserer 32 Units — es spiegelt den experimentellen Konsens der modernen Hirnkritikalitäts-Forschung ("reverberating regime"). Das verändert die Paper-Story von "wir widerlegen" zu "wir messen dasselbe Phänomen wie die moderne Neuro-Statistik, in einem Kontrollsystem": stärker, positiver, zitierfähig.

Weitere Schürfen derselben Debatte:
- "Criticality as a signature of healthy neural systems" ([Frontiers](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2015.00022/full)): die technische Kritik (Power-Laws ubiquitär, können aus Rauschen entstehen).
- "Is There Sufficient Evidence for Criticality?" ([eNeuro 2021](https://www.eneuro.org/content/8/2/ENEURO.0551-20.2021)): auch anaesthesierte/in-vitro-Präparate "kritisch" — wie passt das zu optimaler Informationsverarbeitung? Skepsis strukturell.
- "Brain criticality beyond avalanches" ([IOPscience](https://iopscience.iop.org/article/10.1088/2632-072X/ac2071)): 45 experimentelle Studien, Exponenten uneinheitlich — Universality class offen.
- Neueste kritische Assessments ([arXiv](https://arxiv.org/html/2604.21071), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S3117347026000374)): alternative Nicht-Kritikalitäts-Erklärungen — Rennen offen.
**Unsere Position im Rennen: die einzige Fraktion mit einem artefaktfreien, getriebenen Messgerät + kontrolliertem System.**

## 2. Der große Rahmen: Free-Energy-Prinzip (Friston)
"Free-energy principle: a unified brain theory?" ([Nat Rev Neurosci](https://www.nature.com/articles/nrn2787)):
- Systeme minimieren "Surprise" (Prädiktionsfehler), um interne Zustände in lebensfähigen Grenzen zu halten — Homöostase als informationstheoretisches Oberprinzip ([Wikipedia-Übersicht](https://en.wikipedia.org/wiki/Free_energy_principle)).

**Querverbindung:** Unser Homeostat (F21) IST ein Spezialfall: das Netz minimiert die "Überraschung" über die eigene zukünftige Dynamik (Benettin-Abweichung vom Ziel) als inneres Ziel neben dem Task-Fehler. FEP liefert das Vokabular: "Stabilität als Prior über die eigene Dynamik". Das rahmt F21 nicht als Trick, sondern als Instanz eines der größten Rahmen der theoretischen Neuro. (Ehrliche Grenze: FEP ist ein Prinzip ohne Falsifikation; wir liefern den messbaren Spezialfall.)

## 3. FORCE-Lernen & E-I-Balance: der Vorläufer des Zwei-Kanal-Homeostaten
"Force Learning in Balanced Cortical E-I Networks" ([Neural Computation 2025](https://direct.mit.edu/neco/article/38/4/573/135629)):
- FORCE-Lernen (Sussillo/Abbott-Stil) funktioniert am besten bei optimaler E-I-Balance nahe einer (hochdimensionalen) Kante — die Kooperation von Erregung und Hemmung ist entscheidend; konventionelle Reservoirs unterscheiden sie nicht.

**Querverbindung:** Das ist die Struktur hinter der HSE-Vorhersage aus Tiefenanalyse II (Regler versagt, wenn Feedback-Erregung dominiert) — und der Bauplan des Zwei-Kanal-Homeostaten (F22/F23-Kandidat: getrennte Erregungs- und Hemmungs-Regularisierung statt eines globalen Skalars). Zusätzlich: Wetware-Arbeit trainiert LEBENDE Neuronen mit FORCE auf Lorenz-Chaos ([Neuroscience News](https://neurosciencenews.com/living-neurons-machine-learning-30455/)) — Lorenz, dieselbe Benchmark wie unsere Homeostat-Tests. Die Brücke bio-künstlich existiert schon; unser Sensor wäre ihr Messgerät.

## 4. Gradienten-Theorie: "Curse of Memory" — neue Seite unseres Benettin-Befunds
"RNNs: vanishing and exploding gradients are not the end of the story" ([arXiv 2405.21064](https://arxiv.org/html/2405.21064v2)):
- Selbst OHNE explodierende Gradienten: mit wachsendem Memory werden Parameteränderungen zunehmend empfindlich in ihren Output-Wirkungen ("curse of memory") — Gradient-basiertes Lernen wird hypersensitiv.
- Element-weise Rekurrenz + sorgfältige Parametrisierung (SSM/LSTM-Muster) mildern das.

**Querverbindung (subtil, aber wichtig):** Das ist die Optimierungs-Seite unseres getriebenen Benettin: hohe Sensitivität = positiver transienter FTLE = dasselbe Phänomen, das wir als Ressourcen-Maß feiern, macht gleichzeitig das Training empfindlich. **Memory-Ressource und Trainings-Sensitivität sind zwei Seiten derselben nicht-normalen Münze** — das erklärt, warum der Tradeoff existiert (F12-Kurve fällt auf beiden Seiten), und warum ein Regler (F21) die richtige Form ist: er verwaltet die Sensitivität, nicht nur die Stabilität.

Orthogonale RNNs ([Henaff et al.](http://proceedings.mlr.press/v48/henaff16.pdf), [Vorontsov et al.](https://arxiv.org/html/1702.00071v4)):
- Orthogonale W = kappa ~ 1 (keine transiente Amplifikation) lösen vanish/explode — bleiben aber schwer lernbar und töten die Regime-B-Resource. **Unsere kappa-Analyse liefert die retroaktive Erklärung, warum orthogonale RNNs nie LSTM-Klasse erreichten: kappa=1 ist zu weit vom Wainrib-Galtier-Speicherfenster entfernt.** Eine 10 Jahre alte Architektur-Frage, beantwortet mit unserem Werkzeug — Paper-Fußnote mit Punch.

---

## DAS GROSZE BILD (alle drei Tiefenanalysen vereint)

Ebene 1 (Physik): Sompolinsky 1988 — Chaos-Übergang, g kritisch
Ebene 2 (Messung): Galtier-Wainrib — getriebene LLE, lokale ESP || UNSER F11/F13: SVD-Messartefakt, Benettin-getrieben
Ebene 3 (Ressource): Wainrib-Galtier — lokal expandierend + asymptotisch stabil || UNSER F12/F18c: subkritisches Optimum, Regime B
Ebene 4 (Empirie): Hirnkritikalität — reverberating, 0.94<m<0.998, subkritisch || UNSER F12: Optimum bei -0.33 — DIESELBE leichte Subkriticalität
Ebene 5 (Biologie): Turrigiano-Homöostase, E-I-Balance, FEP || UNSER F21/F22: Homeostat, Zwei-Kanal-Plan
Ebene 6 (Industrie): Mamba = bewiesener Lyapunov-Homeostat; MemMamba: Stabilität != Memory || UNSERE Resource-Kurve: der Tradeoff, der Skalierung begrenzt
Ebene 7 (Optimierung): Curse of Memory — Sensitivität wächst mit Memory || UNSER Benettin-Sensor misst genau diese Sensitivität

DER SATZ DES PROJEKTS (Arbeitstitel, aus allem destilliert):
"Lernende rekurrente Systeme — vom 32-Unit-Modell bis zum Kortex bis zur Industrie-SSM — operieren nicht an der Kante des Chaos, sondern in einem leichten subkritischen Reverberating-Regime, in dem lokale transiente Expansion (Regime B, kappa>~6) die Speicher-Ressource ist und zugleich die Trainings-Sensitivität erzeugt; ein Homeostat, der diese Lage als inneres Ziel hält, ist die minimale Architektur, die Ressource und Stabilität zugleich verwaltet."

Jede Klausel dieses Satzes hat heute eine Quellen- UND eine eigene Experiment-Linie. Das ist das Paper.

## Neue Experiment-Konsequenzen
1. F24 bleibt Priorität (Regime-B-Ressource direkt testen).
2. **F25-KANDIDAT (NEU): Reverberating-Mass m** — kann unser CT-RNN dasselbe "m" (branching/Avalanche-Statistik) reproduzieren, das der Kortex zeigt (0.94-0.998)? Wenn ja: unsere Silizium-Zelle ist ein literatur-kompatibles Modell des reverberating-Regimes — die Brücke zwischen Ebene 3 und 4 wird messbar. (Avalanche-Statistik auf Aktivitätskaskaden des getriebenen Netzes, billig.)
3. Zwei-Kanal-Homeostat (F22/F23) hat jetzt 3 Präzedenz: Turrigiano (synaptic/intrinsic), PNAS 2020 (Korrelation/Rate), E-I-Balance-FORCE.
4. Paper-Skeleton: die 7-Ebenen-Struktur ist das Gliederungsgerüst.
