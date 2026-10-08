"""
InputDSA: Trennt intrinsische Dynamik von input-getriebener Dynamik.
Basierend auf 'InputDSA: Demixing then comparing recurrent and externally
driven dynamics' (ICLR 2026).
Vereinfachte Implementierung: Lineare DMDc-Schätzung.
"""
import torch


@torch.no_grad()
def estimate_dynamics(model, task, seq_len=8, batch=32, device="cpu"):
    """Schätzt intrinsische (A) und input-getriebene (B) Dynamik.
    Modell: dz/dt = f(z, x). Lineare Näherung: z_next ≈ A z + B x."""
    from cgfn.tasks import make_batch, TASKS
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    if hasattr(model, 'task_dim') and model.task_dim > 0:
        idx = TASKS.index(task)
        tv = torch.zeros(batch, model.task_dim, device=device)
        tv[:, idx] = 1.0
        _, states = model(x, tv)
    else:
        _, states = model(x)
    if states.dim() == 4:
        states = states.squeeze(0)
    H = states.shape[-1]
    z = states[:, :-1, :].reshape(-1, H)
    z_next = states[:, 1:, :].reshape(-1, H)
    x_flat = x[:, :-1, :].reshape(-1, x.shape[-1])
    # Design-Matrix [z, x]
    D = torch.cat([z, x_flat], dim=-1)
    # Ridge-Regression: [A B] = (D^T D + λI)^-1 D^T z_next
    lam = 1e-3
    DtD = D.T @ D + lam * torch.eye(D.shape[1], device=device)
    Dtz = D.T @ z_next
    W = torch.linalg.solve(DtD, Dtz)
    A = W[:H, :].T
    B = W[H:, :].T
    return A, B


@torch.no_grad()
def inputdsa(A1, B1, A2, B2):
    """Vergleicht intrinsische und input-getriebene Dynamik.
    Rückgabe: (dsa_intrinsic, dsa_input)."""
    def norm_diff(X, Y):
        num = ((X - Y) ** 2).mean().sqrt()
        den = (X ** 2).mean().sqrt() + (Y ** 2).mean().sqrt() + 1e-12
        return (num / den).item()
    return norm_diff(A1, A2), norm_diff(B1, B2)
