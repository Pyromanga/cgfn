"""
Phasenübergangs-Analyse für Repräsentationskollaps.
Verbindet FTLE, effektive Dimension und Kritikalität.
"""
import torch


@torch.no_grad()
def collapse_phase(ftle_summary, d_eff_ratio):
    """
    Klassifiziert den Kollaps-Typ basierend auf FTLE und d_eff.
    Type-1: stark negative FTLE, niedrige d_eff (Überkontraktion).
    Type-2: stark positive FTLE, hohe d_eff (Chaos).
    Critical: FTLE ≈ 0, hohe d_eff (Rand des Chaos).
    """
    ftle_mean = ftle_summary["ftle_mean"]
    if ftle_mean < -0.5:
        return "Type-1 (Überkontraktion)"
    elif ftle_mean > 0.5:
        return "Type-2 (Chaos)"
    else:
        return "Kritisch"


@torch.no_grad()
def criticality_distance(ftle_summary):
    """Abstand vom kritischen Punkt (|ftle_mean|)."""
    return abs(ftle_summary["ftle_mean"])
