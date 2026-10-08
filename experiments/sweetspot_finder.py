import torch
from cgfn.sweetspot import find_sweet_spot

# Feines Gitter um den vermuteten Sweet Spot herum
tau_grid = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
tasks = ["copy", "parity", "mod3"]
seeds = [0, 1]

print("=" * 74)
print("SWEET-SPOT-FINDER: Feines tau-Gitter um den kritischen Punkt")
print(f"taus={tau_grid}")
print(f"tasks={tasks}  seeds={seeds}")
print("=" * 74)

results = find_sweet_spot(tau_grid, tasks, seeds, steps=1500, hidden_dim=32)

print("\n" + "=" * 74)
print("ZUSAMMENFASSUNG")
print("=" * 74)
print(f"{'tau':>6s} {'CKA_mean':>10s} {'ftle_mean':>10s} {'copy_loss':>10s} "
      f"{'parity_loss':>12s} {'mod3_loss':>10s}")
for r in results:
    print(f"{r['tau']:6.2f} {r['cka_mean']:10.4f} {r['ftle_mean']:+10.4f} "
          f"{r['losses']['copy']:10.4f} {r['losses']['parity']:12.4f} "
          f"{r['losses']['mod3']:10.4f}")

# Finde Sweet Spot: minimale CKA_mean, solange copy_loss < 0.05
valid = [r for r in results if r["losses"]["copy"] < 0.05]
if valid:
    best = min(valid, key=lambda r: r["cka_mean"])
    print(f"\nSweet Spot (min CKA_mean, copy_loss < 0.05):")
    print(f"  tau       = {best['tau']:.2f}")
    print(f"  CKA_mean  = {best['cka_mean']:.4f}")
    print(f"  ftle_mean = {best['ftle_mean']:+.4f}")
else:
    print("\nKein tau erfuellt copy_loss < 0.05")

# Bestimme, wo ftle = 0 liegt (kritischer Punkt)
print("\nKritischer Punkt (ftle ~ 0):")
sorted_r = sorted(results, key=lambda r: abs(r["ftle_mean"]))
closest = sorted_r[0]
print(f"  tau       = {closest['tau']:.2f}")
print(f"  ftle_mean = {closest['ftle_mean']:+.4f}")

print("\n=== FERTIG ===")
