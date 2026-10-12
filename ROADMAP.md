# ROADMAP — cgfn Frontier-Fragen

## Erledigt / etabliert

1. **Sweet-Spot-Existenz** (`sweetspot_results.txt`): CKA-Task-Trennung maximal im mittleren tau-Bereich.
2. **Phasen-Klassifikation** (`frontier*_results.txt`): Type-2-Chaos bei kleinem tau, „kritisch" ab tau≈1; trainierte Netze wandern Richtung Kritikalität.
3. **Korrelations-Struktur** (`frontier3_results.txt`): corr(FTLE, loss) ≈ −0.41, corr(d_eff, loss) ≈ +0.39 → FTLE ist ein brauchbares, aber kein perfektes Prognose-Signal.
4. **Architektur-Vergleich** (`universal_criticality_results.txt`): Gain- vs. tau-Kontrolle über CT-RNN/GRU/LSTM/Vanilla.
5. **FTLE-Trainingsdynamik** (`lyap_training_results.txt`): Lyapunov-Verlauf während des Lernens.


## Erste Befunde aus den Erstrunden-Laewenfen (2026-10-12, results/*.txt)

- **F4 (Noise):** tau* ist vollstaendig sigma-invariant (copy: tau*=0.5, parity: tau*=1.0 ueber sigma in {0, 0.1, 0.3}); FTLE-Werte kaum veraendert => FTLE-Kalibrierung uebertraegt sich auf noisy Deployment. Bonbon: copy bei sigma=0.1 besser als sigma=0 (Noise als Regularisierer). OFFEN: parity bei 1000 Steps noch auf Zufallsniveau (loss~0.68~ln2) — F4 mit --steps>=3000 wiederholen, sonst taugt die parity-Spalte nicht als Beweis.
- **F5 (Kontinuierlich):** sinus_memory (Memory-Task): tau*(loss)=1.0 = tau*(|FTLE|)=1.0 — EXAKTER Treffer auf dem Kritikalitaetspunkt, scharfes Minimum (0.0146 vs 0.146 bei tau=2). lorenz/ou_noise (reactive Tasks): flache Minima mit leichter Praeferenz tau=0.5. Muster: Memory-Tasks sitzen auf FTLE~0, reactive Tasks brauchen schnelle Units.
- **F6 (Theorie):** Spektrum-Korrelation 0.88–1.00 ueberall — die Trajektorien-Lyapunov-Struktur ist fast vollstaendig durch die STATISCHE Jacobi J0=-I/tau+W determiniert. Sompolinsky-Trefferquote 76.7%, ABER einseitig exakt: kein einziger Fall "Chaos vorhergesagt, keins gemessen"; alle 21 Misses konservativ bei kleinem Q. => Verdacht: Bulk-Mittel (g*sqrt Q) unterschaetzt Chaos nahe dem Ursprung, wo der spektrale Rand von W (Nicht-Normalitaet) dominiert. Genau das testet F7.

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


### F7 — Spectral Ruler: Kritikalitaet aus der Gewichtsmatrix  *(implementiert: `src/cgfn/cavity.py` erweitert, `experiments/f7_spectral.py`, `.github/workflows/frontier_f7.yml` — samstags 03:00 UTC + Marker-Commit `.trigger/f7`)*
Motiviert durch den F6-Befund (Spektrum-Korrelation 0.88–1.00; Kriterium einseitig exakt, alle 21 Misses konservativ bei kleinem Q): DREI konkurrierende Chaos-Regeln aus W allein —
(a) Mean-Field `tau*g*sqrt(Q)>1`, (b) Eigenwert `tau*Re lambda_max(W)>1` (exakte lineare Stabilitaet), (c) Singulärwert `tau*(log sv_max + log sqrt(Q))>1` (nicht-normale Kante) — gegen gemessenes SVD-FTLE, untrainiert vs. trainiert, plus Nicht-Normalitaet kappa=sigma_max/|lam_max| als Miss-Erklaerung.
**Ziel-Claim ("Durchbruch"):** Wenn (b) gewinnt, ist der Sweet-Spot geschlossen berechenbar — `tau* ~ 1/Re lambda_max(W)` — Hyperparameter-Kalibrierung ohne Rollout. Die F6-Luecke waere dann kein Widerspruch zur Mean-Field-Theorie, sondern deren praegnantere spektrale Verfeinerung (Bulk vs. spektraler Rand nicht-normaler Matrizen).

## Reproduzierbarkeit & Hygiene

- Result-TXTs bleiben als Referenz im Repo; neue Experimente schreiben nach `results/` (siehe `experiments/scaling.py`).
- Jedes Experiment: argparse-Parameter + Seeds im Log, damit Runs vergleichbar bleiben.
- CI (`smoke.yml`) sichert, dass die Bibliothek nach jedem Push noch läuft; `tests/test_f4f6.py` deckt continuous/cavity ab.
