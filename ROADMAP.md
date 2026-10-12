# ROADMAP — cgfn Frontier-Fragen

## Erledigt / etabliert

1. **Sweet-Spot-Existenz** (`sweetspot_results.txt`): CKA-Task-Trennung maximal im mittleren tau-Bereich.
2. **Phasen-Klassifikation** (`frontier*_results.txt`): Type-2-Chaos bei kleinem tau, „kritisch" ab tau≈1; trainierte Netze wandern Richtung Kritikalität.
3. **Korrelations-Struktur** (`frontier3_results.txt`): corr(FTLE, loss) ≈ −0.41, corr(d_eff, loss) ≈ +0.39 → FTLE ist ein brauchbares, aber kein perfektes Prognose-Signal.
4. **Architektur-Vergleich** (`universal_criticality_results.txt`): Gain- vs. tau-Kontrolle über CT-RNN/GRU/LSTM/Vanilla.
5. **FTLE-Trainingsdynamik** (`lyap_training_results.txt`): Lyapunov-Verlauf während des Lernens.

## Nächste Frontier-Schritte (priorisiert)

### F1 — Scaling-Gesetz des Sweet Spots  *(implementiert: `experiments/scaling.py`)*
Verschiebt sich der kritische Sweet Spot (FTLE≈0) mit hidden_dim?
**Hypothesen**: (a) größeninvariant (klassische Edge-of-Chaos-Vorhersage: Korrekturen ~1/sqrt(N)), (b) driftet systematisch.
**Nutzen**: Wenn (a) — dann ist FTLE ein *skalierbares* Hyperparameter-Signal; man kalibriert tau an einem kleinen Netz und überträgt es.

> **Paper-Radar 2026-10-12:** [Phys. Rev. E „Towards critical branching mechanism in recurrent neural networks"](https://journals.aps.org/pre/abstract/10.1103/bg2r-m6g5) findet kapazitätsabhängige Kritikalität in trainierten LSTMs (kleine Netze nahezu kritisch, größere subkritisch). Stützt Hypothese (b): Sweet-Spot könnte mit N driften. → `experiments/scaling.py` sollte zusätzlich den Branching-Parameter (Avalanche-Statistik) als zweite Metrik neben FTLE loggen.

### F2 — Kausaler Test: FTLE-Zwang während des Trainings  *(implementiert: `src/cgfn/regulator.py`, `experiments/f2_regulator.py`, `.github/workflows/frontier_f2f3.yml` — donnerstags 03:00 UTC + manuell)*
Feedback-Regler (`FtleRegulator`, P-Regler in log-tau mit EMA-Smoothing), der model.tau online nachregelt, sodass FTLE≈0 bleibt. Experiment vergleicht free (tau fix, suboptimaler Start tau=4) gegen Regler-Arme mit Targets {0.0, +0.3, −0.3}: Learning-Speed auf Task A, Loss auf Task B, Retention auf A — die kausale Lesart steht im Output-File.
Vergleich: reguliert vs. frei — Lerngeschwindigkeit, Endverlust, Retention (Plasticity-Loss-These von Wang et al.).
**Das wäre der stärkste kausale Beweis**, dass Kritikalität Ursache ist, nicht nur Korrelat.

### F3 — Plasticity-Loss am CT-RNN  *(implementiert: `src/cgfn/plasticity.py`, `experiments/f3_plasticity.py`, Workflow s.o.)*
Wang-et-al.-Ergebnisse (Plasticity Loss = Lyapunov-Kontraktion) am kontinuierlichen Modell reproduzieren:
sequentielle Tasks, Messung der FTLE-Drift pro Task, Korrelation mit Loss-Einbruch im späteren Task.

> **Paper-Radar 2026-10-12:** [arXiv:2610.01583](https://arxiv.org/abs/2610.01583) (Continual RL with Neuroevolution) zeigt: Plasticity-Loss-Symptome (dormant neurons, Churn) treten unter Neuroevolution nicht auf — die Kontraktion ist offenbar gradientengetrieben, konsistent mit der Wang-Kontraktions-These. Task-Nachbarschaft (Overlap der Loss-Landschaften) als zusätzliches Erklärmaß für Stabilität/Plastizität übernehmen.

### F4 — Robustheit gegen Rauschen & Verteilungs-Shift
Ist der Sweet-Spot auch unter Eingangs-Rauschen / Distribution-Shift optimal? (Anschluss an `tau_robustness_results.txt`, aber mit systematischem Noise-Sweep.)

### F5 — Von binary toy tasks zu kontinuierlichen Tasks
Alle aktuellen Tasks sind bitbasiert. Kontinuierliche Sequenzen (z.B.Sinus-Memory, Lorenz-Vorhersage) — bleibt der Sweet Spot erhalten?

### F6 — Theorie-Anschluss
Mean-field-Theory (Sompolinsky/Van Vreeswijk): Übergangschaos-Bedingung `g·sqrt(Q)=1` explizit gegen gemessene FTLE-Werte stellen; analytische Vorhersage des Sweet-Spot-tau als Funktion von W-Statistik.

> **Paper-Radar 2026-10-12:** [arXiv:2610.12426](https://arxiv.org/abs/2610.12426) (Clark) berechnet das vollständige Lyapunov-Spektrum random-asymmetrischer RNNs exakt via Cavity-Methode (N→∞, extensives Chaos). → F6: direkter analytischer Referenzpunkt für `g·sqrt(Q)=1`; Jacobi-SVD-Pipeline so erweitern, dass neben FTLE_max das gesamte Spektrum gegen die Cavity-Vorhersage gestellt wird.

## Reproduzierbarkeit & Hygiene

- Result-TXTs bleiben als Referenz im Repo; neue Experimente schreiben nach `results/` (siehe `experiments/scaling.py`).
- Jedes Experiment: argparse-Parameter + Seeds im Log, damit Runs vergleichbar bleiben.
- CI (`smoke.yml`) sichert, dass die Bibliothek nach jedem Push noch läuft.
