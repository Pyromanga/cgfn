"""
Finite-Time Lyapunov Exponents (FTLE) für rekurrente Netze.
Basierend auf 'The Dual Nature of Plasticity Loss in Deep Continual Learning'
(Wang et al., NeurIPS 2025).
Misst die lokale Expansions-/Kontraktionsrate entlang einer Trajektorie.
"""
import torch


@torch.no_grad()
def ftle_spectrum(model, states, dt=1.0):
    """
    states: (B, T, H) Trajektorie.
    Schätzt FTLE pro Zeitschritt über die Jacobi-Matrix der Dynamik.
    """
    H = states.shape[-1]
    B, T, _ = states.shape
    ftles = []
    for t in range(T - 1):
        z = states[:, t]  # (B, H)
        sech2 = 1.0 - torch.tanh(z) ** 2  # (B, H)
        J = -torch.eye(H, device=z.device) / model.tau
        J = J + model.W.unsqueeze(0) * sech2.unsqueeze(1)
        # Singulärwerte der Jacobi-Matrix
        sv = torch.linalg.svdvals(J)  # (B, H)
        log_sv = torch.log(sv + 1e-12)
        ftles.append(log_sv.mean(dim=0))  # (H,)
    ftle_mean = torch.stack(ftles, dim=0).mean(dim=0)  # (H,)
    return ftle_mean  # (H,) FTLE-Spektrum


@torch.no_grad()
def ftle_summary(model, states):
    """Zusammenfassung: max, min, mean des FTLE-Spektrums."""
    ftle = ftle_spectrum(model, states)
    return {
        "ftle_max": ftle.max().item(),
        "ftle_min": ftle.min().item(),
        "ftle_mean": ftle.mean().item(),
        "ftle_std": ftle.std().item(),
    }
