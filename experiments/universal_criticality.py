import torch
from cgfn.unified import (
    ARCHS, find_sweet_spot_arch
)

device = "cpu"
TASKS_SUBSET = ["copy", "parity", "mod3"]
SEEDS = [0, 1]

print("=" * 70)
print("UNIVERSELLE KRITIKALITAET: Sweet Spot ueber Architekturen")
print(f"tasks={TASKS_SUBSET}  seeds={SEEDS}")
print("=" * 70)

# CT-RNN: tau-Gitter
print("\n>>> CT-RNN: tau-Gitter")
tau_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
ct_results = find_sweet_spot_arch(
    ARCHS["CT-RNN"], "CT-RNN", tau_grid,
    TASKS_SUBSET, SEEDS, hidden_dim=32, device=device)

# GRU, LSTM, Vanilla: gain-Gitter (Gewichts-Init-Skalierung)
gain_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
for name in ["GRU", "LSTM", "Vanilla"]:
    print(f"\n>>> {name}: gain-Gitter")
    results = find_sweet_spot_arch(
        ARCHS[name], name, gain_grid,
        TASKS_SUBSET, SEEDS, hidden_dim=32, device=device)

# Zusammenfassung
print("\n" + "=" * 70)
print("ZUSAMMENFASSUNG: Sweet Spot pro Architektur")
print("=" * 70)
print(f"{'Architektur':>12s} {'Sweet Spot':>12s} {'CKA_min':>10s} "
      f"{'ftle@spot':>12s} {'copy_loss':>10s}")

# CT-RNN
valid = [r for r in ct_results if r["copy_loss"] < 0.05]
if valid:
    best = min(valid, key=lambda r: r["cka_mean"])
    print(f"{'CT-RNN':>12s} {best['param']:12.2f} {best['cka_mean']:10.4f} "
          f"{best['ftle_mean']:+12.4f} {best['copy_loss']:10.4f}")

# Andere Architekturen
for name in ["GRU", "LSTM", "Vanilla"]:
    print(f"\n--- {name} ---")
    print("  (Ergebnisse oben im Detail)")

# Kritischer Punkt pro Architektur
print("\n" + "=" * 70)
print("KRITISCHER PUNKT: Wo ist ftle ~ 0?")
print("=" * 70)
print("CT-RNN:")
for r in ct_results:
    if abs(r["ftle_mean"]) < 0.1:
        print(f"  tau={r['param']:.2f}  ftle={r['ftle_mean']:+.4f}  "
              f"CKA={r['cka_mean']:.4f}")

print("\n=== FERTIG ===")
print("Interpretation:")
print("  - Wenn GRU/LSTM/Vanilla einen Sweet Spot im gain-Gitter haben,")
print("    ist das Prinzip architektur-unabhaengig.")
print("  - Wenn nicht, ist es CT-RNN-spezifisch.")
print("  - Der Vergleich der ftle@spot-Werte zeigt, ob alle Architekturen")
print("    am kritischen Punkt optimale Trennung erreichen.")
