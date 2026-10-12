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

### F4 — Robustheit gegen Rauschen & Verteilungs-Shift  *(implementiert: `experiments/f4_noise.py`, `.github/workflows/frontier_f4f6.yml` — freitags 03:00 UTC + manuell)*
Ist der Sweet-Spot auch unter Eingangs-Rauschen / Distribution-Shift optimal? (Anschluss an `tau_robustness_results.txt`, aber mit systematischem Noise-Sweep.)
Design: Training noisy (additives Gauß auf Input, sigma-Sweep), Evaluation clean — misst, ob tau*(min clean loss) mit sigma driftet (FTLE-Kalibrierung überträgt sich oder nicht).

### F5 — Von binary toy tasks zu kontinuierlichen Tasks  *(implementiert: `src/cgfn/continuous.py` (sinus_memory/lorenz/ou_noise), `experiments/f5_continuous.py`, Workflow s.o.)*
Alle aktuellen Tasks sind bitbasiert. Kontinuierliche Sequenzen (Sinus-Memory, Lorenz-Vorhersage, Ornstein-Uhlenbeck — MSE statt BCE) — bleibt der Sweet Spot erhalten?
Auswertung pro Task: tau*(min loss) vs. tau*(min |FTLE|), referenziert gegen den binary-Befund (Sweet-Spot mittleres tau, kritisch ab tau≈1).

### F6 — Theorie-Anschluss  *(implementiert: `src/cgfn/cavity.py`, `experiments/f6_spectrum.py`, Workflow s.o.)*
Mean-field-Theory (Sompolinsky/Van Vreeswijk): Übergangschaos-Bedingung `g·sqrt(Q)=1` explizit gegen gemessene FTLE-Werte stellen; analytische Vorhersage des Sweet-Spot-tau als Funktion von W-Statistik.
Implementierung: Chaos-Kriterium `tau·g·sqrt(Q) > 1` (nach tau-Zeitreskalierung) gegen Vorzeichen des gemessenen FTLE_max — Trefferquote über g-Sweep × tau-Sweep, untrainiert vs. kurz trainiert. Plus: volles linearisiertes Jacobi-Spektrum `J0 = -I/tau + W` gegen Trajektorien-FTLE-Spektrum (sortierte Spektrum-Korrelation).

> **Paper-Radar 2026-10-12:** [arXiv:2610.12426](https://arxiv.org/abs/2610.12426) (Clark) berechnet das vollständige Lyapunov-Spektrum random-asymmetrischer RNNs exakt via Cavity-Methode (N→∞, extensives Chaos). → F6: direkter analytischer Referenzpunkt für `g·sqrt(Q)=1`; Jacobi-SVD-Pipeline so erweitern, dass neben FTLE_max das gesamte Spektrum gegen die Cavity-Vorhersage gestellt wird. ✔ umgesetzt in `compare_spectrum()`.

## Reproduzierbarkeit & Hygiene

- Result-TXTs bleiben als Referenz im Repo; neue Experimente schreiben nach `results/` (siehe `experiments/scaling.py`).
- Jedes Experiment: argparse-Parameter + Seeds im Log, damit Runs vergleichbar bleiben.
- CI (`smoke.yml`) sichert, dass die Bibliothek nach jedem Push noch läuft; `tests/test_f4f6.py` deckt continuous/cavity ab.
