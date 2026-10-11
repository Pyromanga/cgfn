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

### F2 — Kausaler Test: FTLE-Zwang während des Trainings
Feedback-Regler, der tau (oder W-Skalierung) online nachregelt, sodass FTLE≈0 bleibt.
Vergleich: reguliert vs. frei — Lerngeschwindigkeit, Endverlust, Retention (Plasticity-Loss-These von Wang et al.).
**Das wäre der stärkste kausale Beweis**, dass Kritikalität Ursache ist, nicht nur Korrelat.

### F3 — Plasticity-Loss am CT-RNN
Wang-et-al.-Ergebnisse (Plasticity Loss = Lyapunov-Kontraktion) am kontinuierlichen Modell reproduzieren:
sequentielle Tasks, Messung der FTLE-Drift pro Task, Korrelation mit Loss-Einbruch im späteren Task.

### F4 — Robustheit gegen Rauschen & Verteilungs-Shift
Ist der Sweet-Spot auch unter Eingangs-Rauschen / Distribution-Shift optimal? (Anschluss an `tau_robustness_results.txt`, aber mit systematischem Noise-Sweep.)

### F5 — Von binary toy tasks zu kontinuierlichen Tasks
Alle aktuellen Tasks sind bitbasiert. Kontinuierliche Sequenzen (z.B.Sinus-Memory, Lorenz-Vorhersage) — bleibt der Sweet Spot erhalten?

### F6 — Theorie-Anschluss
Mean-field-Theory (Sompolinsky/Van Vreeswijk): Übergangschaos-Bedingung `g·sqrt(Q)=1` explizit gegen gemessene FTLE-Werte stellen; analytische Vorhersage des Sweet-Spot-tau als Funktion von W-Statistik.

## Reproduzierbarkeit & Hygiene

- Result-TXTs bleiben als Referenz im Repo; neue Experimente schreiben nach `results/` (siehe `experiments/scaling.py`).
- Jedes Experiment: argparse-Parameter + Seeds im Log, damit Runs vergleichbar bleiben.
- CI (`smoke.yml`) sichert, dass die Bibliothek nach jedem Push noch läuft.
