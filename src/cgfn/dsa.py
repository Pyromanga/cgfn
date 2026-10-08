import torch


@torch.no_grad()
def collect_states(model, task, seq_len=8, batch=32, device="cpu"):
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
    return states


def _procrustes_Q(X, Y):
    """Findet Q mit X @ Q ~ Y. X, Y: (n, H). Rueckgabe: Q (H, H), orthogonal."""
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    U, _, Vh = torch.linalg.svd(Xc.T @ Yc)
    return U @ Vh


def _apply_Q(X, Q, Y_ref):
    Xc = X - X.mean(dim=0, keepdim=True)
    return Xc @ Q + Y_ref.mean(dim=0, keepdim=True)


@torch.no_grad()
def dsa(model_a, model_b, task, seq_len=8, device="cpu"):
    """DSA mit EINER Procrustes-Rotation fuer z und z_next."""
    A = collect_states(model_a, task, seq_len, device=device)
    B = collect_states(model_b, task, seq_len, device=device)
    H = A.shape[-1]

    za = A[:, :-1, :].reshape(-1, H)
    za_next = A[:, 1:, :].reshape(-1, H)
    zb = B[:, :-1, :].reshape(-1, H)
    zb_next = B[:, 1:, :].reshape(-1, H)

    # EINE Rotation, angewendet auf beide Zeitschritte
    Q = _procrustes_Q(zb, za)
    zb_aligned = _apply_Q(zb, Q, za)
    zb_next_aligned = _apply_Q(zb_next, Q, za_next)

    lambda_reg = 1e-3
    M_a = za.T @ za + lambda_reg * torch.eye(H, device=device)
    W_a = torch.linalg.solve(M_a, za.T @ za_next)

    err_self = ((za @ W_a - za_next) ** 2).mean().item()
    err_cross = ((zb_aligned @ W_a - zb_next_aligned) ** 2).mean().item()
    raw = 1.0 - (err_cross / (err_self + 1e-12))
    return {
        "err_self": err_self,
        "err_cross": err_cross,
        "raw": raw,
        "score": max(0.0, min(1.0, raw)),
    }
