# cgfn — Criticality & Geometry of Recurrent Networks

Forschungsrepo zur **Edge-of-Chaos / Criticality-Hypothese in rekurrenten Netzen**:
Gibt es einen universellen Sweet Spot zwischen ordered und chaotischer Dynamik,
in dem rekurrente Netze am besten lernen — und lässt er sich mechanistisch
vorhersagen (FTLE) statt per Trial-and-Error (Hyperparameter-Sweep)?

## Kernidee

- **Modell**: Continuous-Time RNN (CT-RNN): `dz/dt = -z/tau + W·tanh(z) + U·[x; task_vec]`, RK2-integriert.
- **Leitsignal**: Finite-Time Lyapunov Exponents (FTLE) über die Jacobi-Matrix der Dynamik — inspiriert von *„The Dual Nature of Plasticity Loss in Deep Continual Learning"* (Wang et al., NeurIPS 2025).
- **Maße**: FTLE-Spektrum, effektive Dimension (d_eff), CKA (Repräsentations-Trennung), Jacobian Spectral Radius, Branching Ratio.
- **Aufgaben**: copy, reverse, parity, mod3, delayed_xor (Multi-Task über Task-One-Hot).

## Stand der Forschung (Befunde aus den Result-Dateien)

| Ergebnis-Datei | Frage | Kern-Befund |
|---|---|---|
| `sweetspot_results.txt` | Gibt es ein tau-Sweet-Spot für Task-Trennung (CKA)? | Ja — CKA-Trennung hat ein Maximum im mittleren tau-Bereich. |
| `universal_criticality_results.txt` | Ist der Sweet Spot architektur-universell (CT-RNN/GRU/LSTM/Vanilla)? | Siehe Datei — gain vs. tau als Kontrolle. |
| `frontier_results*.txt`, `frontier2_results.txt` | Phasen-Klassifikation der Repräsentationen | Kollaps-Typen: Type-2 (Chaos) bei kleinem tau, kritisch ab tau≈1. |
| `frontier3_results.txt` | Phasenübergang + Korrelationen | **corr(FTLE, loss) = −0.41, corr(d_eff, loss) = +0.39, corr(FTLE, d_eff) = −0.13.** Phase „kritisch" dominiert bei trainierten Netzen; Type-2-Chaos nur bei tau ≤ 0.5. |
| `tau_mechanics_results.txt`, `tau_robustness_results.txt` | Mechanik & Robustheit der tau-Abhängigkeit | siehe Dateien. |
| `lyap_training_results.txt` | FTLE-Verlauf *während* des Trainings | siehe Datei. |

## Struktur

```
src/cgfn/          Kern-Bibliothek
  model.py           CT-RNN (RK2)
  ftle.py            FTLE-Spektrum via Jacobi-SVD
  criticality.py     Spectral Radius, Branching Ratio
  effdim.py          effektive Dimension
  cka.py, dsa.py     Repräsentations-Ähnlichkeit (DSA)
  sweetspot.py       Sweet-Spot-Finder (tau-Grid, CKA-geleitet)
  unified.py         Architektur-Vergleich (CT-RNN/GRU/LSTM/Vanilla)
  mechanics.py, phase.py, geometry.py, ...
experiments/       Laufbare Experimente (jeweils argparse-Skripte)
tests/             Unit-Tests (pytest)
frontier*.txt etc.  Rohe Ergebnis-Logs (Referenz in README-Tabelle)
.github/workflows  CI: smoke.yml (Tests + Smoke-Experiment), scaling.yml (Scaling-Experiment)
```

## Nutzung

```bash
pip install -e ".[dev]"
python -m pytest -q                                  # Unit-Tests
python experiments/run_smoke.py --seeds 0,1,2        # schnelles Smoke-Experiment
python experiments/scaling.py --dims 16,32,64,128    # Scaling-Frontier (neu)
```

CI (`smoke.yml`) läuft bei jedem Push: Unit-Tests + Smoke-Experiment als Artifact.
`scaling.yml` ist manuell/scheduled — siehe Workflow-Datei.

## Siehe auch

- `ROADMAP.md` — was als nächstes ansteht.
