"""
Effective Dimension als geometrisches Generalisierungsmaß.
Basierend auf 'On the Relationship Between Representation Geometry
and Generalization in Deep Neural Networks' (2026).
Misst die intrinsische Dimension der Repräsentationsmannigfaltigkeit.
"""
import torch


@torch.no_grad()
def effective_dimension(states, threshold=0.99):
    """
    states: (n, H) Repräsentationen.
    Schätzt die effektive Dimension über die Singulärwerte der
    zentrierten Datenmatrix. threshold: kumulativer Varianzanteil.
    """
    X = states - states.mean(dim=0, keepdim=True)
    # SVD
    try:
        S = torch.linalg.svdvals(X)
    except Exception:
        return float('nan')
    S2 = S ** 2
    total = S2.sum()
    if total <= 0:
        return 0.0
    cum = torch.cumsum(S2, dim=0) / total
    d_eff = (cum < threshold).sum().item() + 1
    return float(d_eff)


@torch.no_grad()
def effective_dimension_ratio(states):
    """Verhältnis effektive Dimension / nominale Dimension."""
    d_eff = effective_dimension(states)
    H = states.shape[-1]
    return d_eff / H
