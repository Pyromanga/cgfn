"""
Mechanistische Analyse des tau-Effekts.
Drei Perspektiven:
1. Untrained Dynamics: Was macht tau mit zufaelligen Gewichten?
2. Critical tau: Wo ist die Jacobi-Matrix am Rand der Stabilitaet?
3. Trained sweet spot: Wo trennt das Training Aufgaben am besten?
"""
import torch


@torch.no_grad()
def jacobian_spectral_radius_at_fixed_point(W, tau, z=None):
    """
    Spektralradius der Jacobi-Matrix J = -I/tau + W * diag(1 - tanh^2 z).
    Ohne z: am Ursprung (z=0), dann diag=1.
    """
    H = W.shape[0]
    if z is None:
        J = -torch.eye(H) / tau + W
    else:
        sech2 = 1.0 - torch.tanh(z) ** 2
        J = -torch.eye(H) / tau + W * sech2.unsqueeze(0)
    sv = torch.linalg.svdvals(J)
    return sv.max().item()


@torch.no_grad()
def untrained_state_stats(model, input_scale=1.0, seq_len=8, batch=32, device="cpu"):
    """
    Laesst ein untrained Modell mit zufaelligem Input laufen.
    Misst: state norm, state variance, FTLE (mittlere log-Singulaerwerte).
    """
    H = model.hidden_dim
    z = torch.zeros(batch, H, device=device)
    states = []
    x_seq = torch.randn(batch, seq_len, 1, device=device) * input_scale
    with torch.no_grad():
        for t in range(seq_len):
            x = x_seq[:, t]
            for _ in range(model.substeps):
                k1 = model.dynamics(z, x, torch.zeros(batch, 0, device=device))
                k2 = model.dynamics(z + 0.5 * model.dt * k1, x,
                                    torch.zeros(batch, 0, device=device))
                z = z + model.dt * k2
            states.append(z)
    states = torch.stack(states, dim=1)
    # FTLE ueber Jacobi am letzten Zustand
    z_last = states[:, -1]
    sech2 = 1.0 - torch.tanh(z_last) ** 2
    J = -torch.eye(H, device=device) / model.tau
    J = J.unsqueeze(0) + model.W.unsqueeze(0) * sech2.unsqueeze(1)
    sv = torch.linalg.svdvals(J)
    log_sv = torch.log(sv + 1e-12)
    return {
        "state_norm": states[:, -1].norm(dim=-1).mean().item(),
        "state_std": states[:, -1].std(dim=-1).mean().item(),
        "ftle_max": log_sv[:, 0].mean().item(),
        "ftle_min": log_sv[:, -1].mean().item(),
    }
