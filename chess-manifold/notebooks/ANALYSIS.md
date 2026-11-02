# Analyse-Notizen

## Was in `results/report.json` zu finden ist

- `persistent_homology.<becken>.betti_curve` — Betti-0/1 ueber Epsilon-Grid + H1-Gesamtpersistenz
- `geodesic_start_to_basins` — diskrete Geodaetik Start -> Becken (Mittel ueber Beckenmitglieder)
- `geodesic_early_mean_to_basins` — dasselbe gemittelt fuer Ply 8-20
- `bootstrap_geodesic_test.draw_minus_{white,black}` — Delta, 95%-CI, einseitiger p-Wert
- `coherence_knn_accuracy` — Kohaerenzmetrik (k-NN), Vergleichswert Literatur: 0.917
- `contradiction_examples` — FEN + wp + topologische Vorhersage (Gegenbeispielsuche)
- `baseline_depth1_accuracy_all` — Stockfish depth=1 als "ohne Suche"-Baseline

## Wie man ein eigenes Notebook anknuepft

`LT`python
import pandas as pd, json
emb = pd.read_csv("data/embeddings.csv")
rep = json.load(open("results/report.json"))
`LT`

## Interpretationshinweise

- Breite Bootstrap-CIs bei kleinem Remis-Becken sind zu erwarten (Random-Selfplay
  erzeugt wenige ausgeglichene Stellungen) -> Run mit Profil full fuer belastbare Zahlen.
- Die topologische Vorhersage fuer die Startstellung ist nur dann konsistent,
  wenn draw_minus_white UND draw_minus_black negative CIs haben.
