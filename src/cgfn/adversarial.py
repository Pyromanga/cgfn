"""
Adversariale Robustheit als Funktion von tau.
Misst, wie stark kleine Input-Perturbationen den Output verändern.
"""
import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


@torch.no_grad()
def adversarial_sensitivity(model, task, task_dim, eps=0.1, device="cpu"):
    """
    Misst die maximale Output-Änderung bei Input-Perturbation der Größe eps.
    Verwendet FGSM-ähnliche Perturbation (einfacher Gradientenschritt).
    """
    idx = TASKS.index(task)
    x, y = make_batch(task, batch_size=32, seq_len=8, device=device)
    tv = torch.zeros(x.shape[0], task_dim, device=device)
    tv[:, idx] = 1.0

    # Vorwärts mit Original
    out_orig, _ = model(x, tv)

    # FGSM: Gradient der Loss nach Input
    x_adv = x.clone().detach().requires_grad_(True)
    out, _ = model(x_adv, tv)
    loss = torch.nn.BCEWithLogitsLoss()(out, y)
    loss.backward()
    grad = x_adv.grad

    # Perturbation
    x_pert = x + eps * grad.sign()
    out_pert, _ = model(x_pert, tv)

    # Maximale absolute Änderung
    diff = (out_pert - out_orig).abs().max().item()
    return diff


@torch.no_grad()
def sensitivity_curve(model, task, task_dim, eps_values, device="cpu"):
    """Sensitivität für verschiedene eps-Werte."""
    results = []
    for eps in eps_values:
        s = adversarial_sensitivity(model, task, task_dim, eps, device)
        results.append((eps, s))
    return results
