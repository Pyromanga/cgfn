"""
Dynamical Similarity Analysis (DSA) nach Ostrow et al. (2024).
Misst, ob zwei RNNs dieselbe *Dynamik* nutzen, nicht nur dieselbe
Repraesentationsgeometrie. DSA ist robuster gegen Rauschen und
identifiziert behaviorale Konvergenz zuverlaessiger als CKA.
"""
import torch


@torch.no_grad()
def collect_dynamics(model, task, seq_len=8, batch=32, device="cpu"):
    """Sammelt (state, next_state) Paare ueber eine Trajektorie."""
    from cgfn.tasks import make_batch
    from cgfn.model import ContinuousRNN
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    if hasattr(model, 'task_dim') and model.task_dim > 0:
        from cgfn.tasks import TASKS
        idx = TASKS.index(task)
        tv = torch.zeros(batch, model.task_dim, device=device)
        tv[:, idx] = 1.0
        _, states = model(x, tv)
    else:
        _, states = model(x)
    z = states[:, :-1, :].reshape(-1, states.shape[-1])
    z_next = states[:, 1:, :].reshape(-1, states.shape[-1])
    return z, z_next


def dsa(model_a, model_b, task, seq_len=8, device="cpu"):
    """
    Vergleicht zwei Modelle ueber die lineare Vorhersagbarkeit
    der Dynamik. Aendert sich die Dynamik, sinkt DSA.
    """
    za, za_next = collect_dynamics(model_a, task, seq_len, device=device)
    zb, zb_next = collect_dynamics(model_b, task, seq_len, device=device)

    # Ridge-Regression: z_next ~ z
    lambda_reg = 1e-3
    Ha = za.shape[1]

    # Modell A: Vorhersage von za_next aus za
    A_a = (za.T @ za + lambda_reg * torch.eye(Ha, device=device))
    W_a = torch.linalg.solve(A_a, za.T @ za_next)
    pred_a = za @ W_a

    # Cross-Prediction: sagt W_a die Dynamik von B voraus?
    pred_b_from_a = zb @ W_a
    err_cross = ((pred_b_from_a - zb_next) ** 2).mean()

    # Baseline: sagt W_b die eigene Dynamik voraus?
    A_b = (zb.T @ zb + lambda_reg * torch.eye(Ha, device=device))
    W_b = torch.linalg.solve(A_b, zb.T @ zb_next)
    pred_b = zb @ W_b
    err_self = ((pred_b - zb_next) ** 2).mean()

    # DSA: 1 - (cross_error / self_error), geclippt auf [0, 1]
    score = 1.0 - (err_cross / (err_self + 1e-12)).item()
    return max(0.0, min(1.0, score))
