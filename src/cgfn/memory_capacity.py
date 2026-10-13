"""
F12 — Memory Capacity (Jaeger 2001) als Ressourcen-Mass:
Nicht-Normalitaet als Speicher-Hebel.

Klassisches Reservoir-Mass: Wie viel Information ueber die PASTEINGABE
steckt noch im Zustand? Pro Verzoegerung lag wird ein linearer Readout
(Ridge-Regression, geschlossene Loesung — kein SGD, kein Zufall) aus
dem Zustand z_t auf das vergangene Input-Symbol x_{t-lag} gefittet und
die quadrierte Korrelation auf einem DISJUNKTEN Testset gemessen:

    MC = sum_lag  corr^2( W_readout(z_t), x_{t-lag} )

Eigenschaften (warum dieses Mass fuer F12):
  - Deterministisch: Ridge hat eine geschlossene Loesung, kein Training.
  - Kein getrickster Readout: linear ist die untere Schranke der
    tatsaechlich dekodierbaren Information (nichts wird versteckt).
  - Standard in der Reservoir-Literatur (Jaeger, Echo-State-Referenz) —
    direkt vergleichbar.
  - MC <= H (Zustandsdimension) — die Kapazitaetsgrenze.

Der Antrieb ist WEISSES RAUSCHEN (Standard fuer MC) — dieselben Input-
Statistiken wie beim Benettin-Messgeraet im F12-Experiment, damit die
"getriebene Kante" (F11-Befund) und die Speicherressource auf
DENSELBEN Trajektorien-Statistiken gemessen werden.

Bekannte Kleinigkeit: Lag-Fenster kreuzt Batch-Grenzen (konkatenierte
Trajektorien) — bei weissem Rauschen ist die Kontamination vernachlaessigbar
(zufaellige Phasen), das disjunkte Testset bleibt unberuehrt.
"""
import torch


def _ridge_r2(Z_tr, y_tr, Z_te, y_te, ridge=1e-3):
    """R^2 (= corr^2 zwischen Vorhersage und Ziel) via Ridge-Regression.

    Geschlossene Loesung: w = (Z^T Z + r I)^-1 Z^T y — deterministisch.
    R^2 via quadrierter Pearson-Korrelation (das MC-Mass, nicht MSE-R^2):
    robust gegen Skalierung des Readouts.
    """
    H = Z_tr.shape[1]
    A = Z_tr.T @ Z_tr + ridge * torch.eye(H)
    w = torch.linalg.solve(A, Z_tr.T @ y_tr)
    pred = Z_te @ w
    vz = pred - pred.mean()
    vy = y_te - y_te.mean()
    denom = (vz.norm() * vy.norm()).clamp_min(1e-12)
    return ((vz @ vy) / denom).item() ** 2


@torch.no_grad()
def memory_capacity(model, max_lag=20, train_batches=8, test_batches=8,
                    seq_len=64, input_scale=0.5, ridge=1e-3):
    """MC = sum_lag corr^2(z_t -> x_{t-lag}) mit disjunktem Testset.

    Rueckgabe:
      mc        — float, Summe ueber lags 1..max_lag (lag 0 bewusst NICHT
                  gezaehlt: reine Input-Injektion, keine Speicherleistung)
      r2_by_lag — list[float], corr^2 pro lag 0..max_lag (Diagnose)
    """
    # MC-eigener deterministischer Generator — unabhaengig vom Aufrufer-
    # Seed (der die NETZE determiniert): zwei Laeufen desselben Netzes
    # liefern identisches MC (siehe test_mc_deterministic_for_same_seed).
    gen = torch.Generator().manual_seed(20261013)

    def collect(n_batches):
        states_all, inputs_all = [], []
        for _ in range(n_batches):
            x = torch.randn(1, seq_len, 1, generator=gen) * input_scale
            _, states = model(x, None)
            states_all.append(states[0])
            inputs_all.append(x[0, :, 0])
        return torch.cat(states_all, dim=0), torch.cat(inputs_all, dim=0)

    Z_tr, x_tr = collect(train_batches)
    Z_te, x_te = collect(test_batches)

    r2_by_lag = []
    for lag in range(max_lag + 1):
        f_tr, t_tr = (Z_tr[lag:], x_tr[:-lag]) if lag else (Z_tr, x_tr)
        f_te, t_te = (Z_te[lag:], x_te[:-lag]) if lag else (Z_te, x_te)
        r2_by_lag.append(max(0.0, _ridge_r2(f_tr, t_tr, f_te, t_te, ridge)))

    mc = sum(r2_by_lag[1:])  # lag 0 zaehlt nicht (s.o.)
    return mc, r2_by_lag
