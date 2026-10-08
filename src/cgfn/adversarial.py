"""
Adversariale Robustheit als Funktion von tau.

WICHTIG: @torch.no_grad() nur um Inferenz, NICHT um backward()!
"""
import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def adversarial_sensitivity(model, task, task_dim, eps=0.1, device="cpu"):
    """
    FGSM-Sensitivitaet: maximale Output-Aenderung bei eps-Perturbation.
    """
    idx = TASKS.index(task)
    x, y = make_batch(task, batch_size=32, seq_len=8, device=device)
    tv = torch.zeros(x.shape[0], task_dim, device=device)
    tv[:, idx] = 1.0

    # Original-Output (kein Gradient noetig)
    with torch.no_grad():
        out_orig, _ = model(x, tv)

    # Gradienten-Pfad: braucht Autograd
    x_adv = x.clone().detach().requires_grad_(True)
    out, _ = model(x_adv, tv)
    loss = torch.nn.BCEWithLogitsLoss()(out, y)
    loss.backward()
    grad = x_adv.grad

    if grad is None:
        return float("nan")

    # Perturbierter Output (wieder no_grad)
    with torch.no_grad():
        x_pert = x + eps * grad.sign()
        out_pert, _ = model(x_pert, tv)
        diff = (out_pert - out_orig).abs().max().item()
    return diff


def sensitivity_curve(model, task, task_dim, eps_values, device="cpu"):
    results = []
    for eps in eps_values:
        s = adversarial_sensitivity(model, task, task_dim, eps, device)
        results.append((eps, s))
    return results
