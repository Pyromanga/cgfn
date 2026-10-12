"""
F3 — Plasticity-Loss-Metriken am CT-RNN.

These (Wang et al., NeurIPS 2025, 'The Dual Nature of Plasticity Loss'):
Plasticity-Verlust ist Lyapunov-KONTRAKTION — das Netz wird traeg, weil die
Dynamik zu stabil wird. Metriken hier, um das am kontinuierlichen Modell
sequentiell zu messen:

  - dormant_fraction : Anteil Units mit (fast) konstanter Aktivierung ueber Zeit
  - weight_churn    : relative L2-Aenderung der W-Matrix waehrend einer Phase
  - plasticity_snapshot : FTLE + Dormancy + W-Norm als ein Dict

Gegenthese-Check (Paper-Radar 2610.01583): Churn/Dormancy sollten unter
gradientengetriebenem Training stark, unter eingefrorenen W (hier: lr=0) null
sein — der Test `test_weight_churn_zero_for_frozen` sichert die Baseline.
"""
import torch

from .ftle import ftle_summary


@torch.no_grad()
def dormant_fraction(states, threshold=0.05):
    """Anteil der Units mit (fast) konstanter Aktivierung. states: (B, T, H)."""
    if states.dim() == 4:
        states = states.squeeze(0)
    var = states.var(dim=1)          # (B, H)
    frac = (var.mean(dim=0) < threshold).float().mean()
    return float(frac.item())


@torch.no_grad()
def weight_churn(model_before, model_after):
    """Relative L2-Aenderung der rekurrenten W-Matrix (Gradient-Churn)."""
    d = model_after.W - model_before.W
    denom = model_before.W.norm().clamp_min(1e-12)
    return float((d.norm() / denom).item())


@torch.no_grad()
def plasticity_snapshot(model, states):
    """FTLE + Dormancy + W-Norm — ein Schnappschuss pro Trainingsphase."""
    f = ftle_summary(model, states)
    return {
        "ftle_mean": f["ftle_mean"],
        "ftle_max": f["ftle_max"],
        "dormant_fraction": dormant_fraction(states),
        "w_norm": float(model.W.norm().item()),
    }


def correlate(xs, ys):
    """Pearson-Korrelation ohne numpy (dependencies minimal halten)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5
