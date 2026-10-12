# ROADMAP — cgfn Frontier-Fragen

## Erledigt / etabliert

1. **Sweet-Spot-Existenz** (`sweetspot_results.txt`): CKA-Task-Trennung maximal im mittleren tau-Bereich.
2. **Phasen-Klassifikation** (`frontier*_results.txt`): Type-2-Chaos bei kleinem tau, „kritisch" ab tau≈1; trainierte Netze wandern Richtung Kritikalität.
3. **Korrelations-Struktur** (`frontier3_results.txt`): corr(FTLE, loss) ≈ −0.41, corr(d_eff, loss) ≈ +0.39 → FTLE ist ein brauchbares, aber kein perfektes Prognose-Signal.
4. **Architektur-Vergleich** (`universal_criticality_results.txt`): Gain- vs. tau-Kontrolle über CT-RNN/GRU/LSTM/Vanilla.
5. **FTLE-Trainingsdynamik** (`lyap_training_results.txt`): Lyapunov-Verlauf während des Lernens.


## Erste Befunde aus den Erstrunden-Läufen (2026-10-12, results/*.txt)

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
(a) Mean-Field `tau*g*sqrt(Q)>1`, (b) Eigenwert `tau*Re lambda_max(W)>1` (exakte lineare Stabilitaet), (c) Singulärwert `tau*sv_max(W)*sqrt(Q)>1` (nicht-normale spektrale Kante) — gegen gemessenes SVD-FTLE, untrainiert vs. trainiert, plus Nicht-Normalitaet kappa=sv_max/|lam_max| als Miss-Erklaerung.
**Ziel-Claim ("Durchbruch"):** Wenn (b) gewinnt, ist der Sweet-Spot geschlossen berechenbar — `tau* ~ 1/Re lambda_max(W)` — Hyperparameter-Kalibrierung ohne Rollout. Die F6-Luecke waere dann kein Widerspruch zur Mean-Field-Theorie, sondern deren praegnantere spektrale Verfeinerung (Bulk vs. spektraler Rand nicht-normaler Matrizen).
*Hygiene-Notiz:* Run 1 von `frontier_f7` war rot, weil `tests/test_f4f6.py` die Rückgabe von `chaos_criterion` auf exakte Key-Gleichheit prüfte und F7 die Keys erweitert hat — Test auf Subset-Prüfung umgestellt (gleicher Commit wie F8/F9).

### F8 — Ruler-Anwendung: Wo funktioniert der Spectral Ruler wirklich?  *(implementiert: `src/cgfn/ruler.py`, `experiments/f8_ruler.py`, `.github/workflows/frontier_f8f9.yml` — sonntags 03:00 UTC + Marker-Commit `.trigger/f8f9`)*
Der Spectral Ruler wird vom Diagnose-Tool zum getesteten Instrument: Checkpoint-Sweep über das Training — pro Checkpoint `spectral_report()` + gemessenes FTLE-Vorzeichen.
Zwei Fragen: (1) **Regel-Güte**: Welche Chaos-Regel (MF/Eig/SV) trifft das FTLE-Vorzeichen zu jedem Trainingszeitpunkt, und wo genau versagt Mean-Field (Kartierung der F6-Lücke)? (2) **Kritikalitäts-Drift**: Wandert Re lambda_max(W) im Training Richtung 1/tau („Training findet Kritikalität von selbst")? — das wäre die stärkste Form des Durchbruch-Claims: die Lernregel selbst ist ein Kritikalitäts-Regler.
Diagnose-Output des Tools pro Netz: Phase (chaotisch/transient/subkritisch), Sweet-Spot-Vorhersage ohne Rollout, Stabilitätsrand, kappa, Q, d_eff-Ratio, dormant-Anteil (Plasticity-Frühwarnung).

### F9 — Generatives Pendant: Kritikalitaet by construction  *(implementiert: `src/cgfn/design.py`, `experiments/f9_design.py`, Workflow s.o.)*
Umkehrung der Diagnose: Gewichte per Schur-Bau mit vorgeschriebenem Re lambda_max(W) und einstellbarer Nicht-Normalitaet s konstruieren; Reservoir-Setting (freeze_w, nur U/Readout lernen), damit die Vorhersage über das Training gültig bleibt.
Prüfung: Sagt der Ruler das Verhalten VORHER — liegt das empirische FTLE-Crossing bei tau_pred_eig=1/lam_max (bei s=0 zwingend, Konsistenz-Check) oder wandert es mit wachsendem kappa zur SVD-Kante tau_pred_sv? Und sitzt tau*(loss) auf dem Crossing?
**Mögliche Kernentdeckung:** Wenn kappa der Schalter zwischen Eigenwert- und SVD-Regel ist, haben wir keinen Widerspruch, sondern ein *zwei-Regime-Gesetz* — Ruler = f(kappa) — und Mean-Fields 76.7% wären der Grenzfall normaler Matrizen.

### F10 — Sättigungs-korrigierter Ruler: das eine Gesetz?  *(implementiert: `src/cgfn/cavity.py`+`src/cgfn/ruler.py` erweitert, `experiments/f10_qbar.py`, `.github/workflows/frontier_f10.yml` — sonntags 03:30 UTC + Marker `.trigger/f10`)*
F9 zeigte bei kappa=1: das FTLE-Crossing liegt NICHT bei 1/lam_max, sondern deutlich darüber (1.25→1.84, 0.87→2.54). These: Der Ruler am Ursprung (d=sech²(0)=1) überschätzt Instabilität bei input-getriebenen Netzen; entlang der Trajektorie gilt J(z) = -I/tau + W·diag(sech²(z)), also Chaos <=> tau·d·Re lambda_max(W) > 1 mit d = <sech²(z)> = 1-Q. Messbar aus EINEM Rollout, self-consistent iterierbar: tau* = 1/(lam_max·d(tau*)).
**Prüfung:** (A) Designed Networks (kappa=1): Crossing vs. 1/lam vs. self-consistent 1/(lam·d). (B) F8-Replikat: Regel-Güte MF/Eig/EigQbar/SV — schließt die Korrektur die 5 hit_eig-Misses, ersetzt sie die SVD-Regel? Wenn ja: EIN Gesetz statt dreier Regeln — der Durchbruch-Claim.

**F10-Befund (2026-10-12, results/f10_qbar_results.txt):** Die d-Korrektur verbessert die Designed-Network-Vorhersage (mittlerer Fehler 1.30 -> 0.91), schliesst die Luecke aber NICHT (Crossing 2.01 vs. tSC 2.56 bei lam=0.8; Richtung inkonsistent zwischen lam-Werten) — das skalare Ansatz-Modell J ~ -I/tau + d*W ist zu grob (die echte Jacobi ist pro-Unit diag(sech^2)). Teil B: eigqbar faellt auf 47.5% (vs. eig 87.5%), SVD bleibt 100%. Fazit: KEIN Ersatz — die SVD-Regel bleibt Champion, aber sie teilt die mathematische Struktur mit unserem SVD-basierten FTLE-Messgeraet (Zirkularitaets-Verdacht, s. F11).

### F11 — Messplatz-Validierung: SVD-FTLE vs. Benettin-QR  *(implementiert: `src/cgfn/benettin.py`, `experiments/f11_benettin.py`, `.github/workflows/frontier_f11.yml` — dienstags 03:00 UTC + Marker `.trigger/f11`)*
Der Zirkularitaets-Verdacht aus F10: Die SVD-Regel (100% in F8) und unser FTLE-Messgeraet sind beide SVD-basiert — die Regel koennte das Messgeraet vorhersagen statt das System. F11 fuehrt das Standard-Messgeraet (Benettin-QR: Tangenten-Dynamik -dz/tau + W diag(sech^2(z)) dz, simultan mit RK2/substeps integriert, QR-Reorthonormalisierung pro Substep) als unabhaengige zweite Instanz ein und stellt beide Regeln gegen BEIDE Messgeraete. Plus Stabilitaets-Audit sv_max(J0)*dt < 2 (RK2-Stabilitaetsregion) pro Konfiguration — die s=3-Daten aus F9 (kappa~14.6) liegen vermutlich ausserhalb. Entscheidende Frage: Wenn eig unter Benettin aufholt, war die SVD-Dominanz von F7/F8 ein Messartefakt; wenn sv auch unter Benettin gewinnt, ist das Zwei-Regime-Gesetz Ruler=f(kappa) echt.

## Literatur-Positionierung (Recherche 2026-10-12)
- **Sättigungs-korrigierte Chaos-Regel (F10):** In dieser Form NEU — als empirisch motivierte Verfeinerung des SCS/Sompolinsky gJ=1-Kriteriums für getriebene, gesättigte Netze positionieren; SCS/Schuecker-Linie als Kontext zitieren.
- **Nicht-Normalität:** Phänomen bekannt (Trefethen/Embree Pseudospektren, "non-normal route to chaos", Non-Normality-Index K), aber keine explizite Formel Kanten-Abstand = f(kappa) — unsere F7/F9-Charakterisierung ist ein Neuheitsbeitrag. Trefethen als theoretische Grundlage zitieren.
- **Self-consistency-Iteration:** Standard in der Mean-Field-Literatur ist gedämpfte Fixpunkt-Iteration — unseren Ansatz (Formel, Dämpfungsfaktor, Konvergenzkriterium) explizit dokumentieren; Reviewer erwartet Konvergenzanalyse.
- **SGD-Drift-Claim (F8: 7/8 Richtung Regelrand):** WICHTIG — Literatur findet den GEGENTEILIGEN Effekt (SGD treibt autonome Netze in den superkritischen Bereich; nur homeostatische Regeln stabilisieren Kritikalität). Unser Befund ist entweder falsch gemessen, oder der entscheidende Unterschied ist: (a) unsere Netze sind GETRIEBEN, (b) wir messen den Rand über tau*lam_max statt lambda_0. Muss vor Paper-Abgabe mit einem Kontroll-Experiment adressiert werden (F8.2: laengeres Training, mehr Seeds, Drift uber lam_max ODER FTLE getrennt ausweisen).
- **Kritikalitaet als Hyperparameter-Signal:** Mean-field-basierte Ansätze existieren (analytische kritische Initialisierung, auto-critical reservoirs) — unseren Ruler als SPEKTRALEN Ersatz positionieren: keine Aktivierungs-/Verteilungsannahmen, direkte Ablesung aus W.
- **Methodik:** Benettin-QR ist der Standard fuer Lyapunov-Spektren; SVD-FTLEs ueber kurze Rollouts sind bei kappa>>1 transient-verzerrt (=> F11). RK2-Stabilitaet: |lambda|*dt < ~2 — F9 s=3-Konfigurationen verletzen das vermutlich (=> Stabilitaets-Audit in jedem Experiment, verdächtige Zeilen markieren).

## Das Validierungs-Dreieck (Messung / Theorie / Konstruktion)

Warum drei „Teilnehmer" statt zwei: Eine Messung, die nur gegen eine Theorie geprüft wird (Dyade), ist zirkulär — beide können denselben systematischen Fehler teilen (hier: Mean-Field-Annahme g*sqrt(Q) + SVD-FTLE-Messung, die beide die nicht-normale Kante verwischen). Erst ein drittes, unabhängiges Bein bricht den Zirkel:
1. **Messung** (F1–F5): FTLE-Spektrum, tau*(loss) — empirisch, aber messwerkzeug-abhängig.
2. **Theorie** (F6/F7): Cavity-/Sompolinsky-Kriterium, Eigenwert-/SVD-Regeln — geschlossen, aber näherungsbehaftet.
3. **Konstruktion** (F9): Designed W, bei dem die Vorhersage *per Bau* bekannt ist — externe Referenz, die weder das Messwerkzeug noch die Theorie-Annahmen teilt.
F9 ist damit der „externe Anker": Wenn Ruler-Vorhersage und gemessenes Verhalten auch bei konstruierten Netzen übereinstimmen, ist die Übereinstimmung nicht zirkulär. Ein Durchbruch-Claim braucht alle drei Beine — F8/F9 schließen das Dreieck.

## Reproduzierbarkeit & Hygiene

- Result-TXTs bleiben als Referenz im Repo; neue Experimente schreiben nach `results/` (siehe `experiments/scaling.py`).
- Jedes Experiment: argparse-Parameter + Seeds im Log, damit Runs vergleichbar bleiben.
- CI (`smoke.yml`) sichert, dass die Bibliothek nach jedem Push noch läuft; `tests/test_f4f6.py` deckt continuous/cavity ab, `tests/test_f8f9.py` design/ruler.
