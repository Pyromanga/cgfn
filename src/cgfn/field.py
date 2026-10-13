"""
Feld-Gesetz (F17/F18): selbstkonsistente Pro-Unit-sech^2-Iteration.

Bibliotheksmodule (aus experiments/f17_battery.py gehoben — F18 crashte
am Script-Modus-Import; Messgeraete gehoeren in die Bibliothek).
"""
import torch

DRIVE_GEN_SEED = 20261013


def drive_input(batch, T, scale):
    gen = torch.Generator().manual_seed(DRIVE_GEN_SEED)
    return torch.randn(batch, T, 1, generator=gen) * scale


@torch.no_grad()
def sech2_field_iteration(model, x_probe, n_iter=40, damp=0.7):
    """Pro-Unit-Gain d_i = <sech^2(z_i)> aus der getriebenen Trajektorie
    (gedaempft gemittelt); d ist die selbstkonsistente Loesung der
    Zuordnung Trajektorie <-> linearisierte Analyse."""
    m = model
    H = m.hidden_dim
    d = torch.ones(H)
    _, states = m(x_probe, None)
    for _ in range(n_iter):
        d_new = (1.0 - torch.tanh(states[0]) ** 2).mean(dim=0)
        d = damp * d + (1 - damp) * d_new
    return d


@torch.no_grad()
def predict_driven_ftle_field(model, x_probe):
    """Feld-Gesetz: groesster LLE von J = -I/tau + W diag(d(traj))."""
    d = sech2_field_iteration(model, x_probe)
    H = model.hidden_dim
    J = -torch.eye(H) / model.tau + model.W * d.unsqueeze(0)
    return torch.linalg.eigvals(J).real.max().item()


def kreiss_constant(J):
    """K(J) = sv_max/sv_min - 1 (Transienten-Scharnier)."""
    sv = torch.linalg.svdvals(J)
    sv_min = sv[-1].item() if sv[-1].item() > 1e-12 else 1e-12
    return (sv[0].item() / sv_min) - 1.0


def transient_bound_pseudo(J, n_grid=16):
    """Trefethen-Unterschranke: max_s sv((J-sI)^{-1})^{-1}... hier:
    max_s 1/sv_min(J - sI) ueber ein 1D-Gitter im Spektralbereich."""
    H = J.shape[0]
    lams = torch.linalg.eigvals(J).real
    grid = torch.linspace(lams.min().item(), lams.max().item(), n_grid)
    I = torch.eye(H)
    best = 0.0
    for s in grid:
        sv = torch.linalg.svdvals(J - s * I)
        if sv[-1].item() < 1e-12:
            return float("inf")
        best = max(best, 1.0 / sv[-1].item())
    return best
