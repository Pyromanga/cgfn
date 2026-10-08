"""
Training mit FTLE-Regularisierung.
Statt tau manuell zu tunen, wird das Modell gezwungen,
am kritischen Punkt zu bleiben (ftle_mean ~ 0).
"""
import torch
import torch.nn as nn
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def ftle_penalty(model, z, target=0.0):
    """
    Penalty: (ftle_mean - target)^2.
    z: (B, H) letzter Zustand.
    """
    H = model.hidden_dim
    sech2 = 1.0 - torch.tanh(z) ** 2
    J = -torch.eye(H, device=z.device) / model.tau
    J = J.unsqueeze(0) + model.W.unsqueeze(0) * sech2.unsqueeze(1)
    sv = torch.linalg.svdvals(J)
    log_sv = torch.log(sv + 1e-12)
    ftle_mean = log_sv.mean()
    return (ftle_mean - target) ** 2


def train_lyap(seed, task, lambda_ftle=0.1, target_ftle=0.0,
               steps=1500, lr=1e-3, hidden_dim=32, tau=1.0,
               task_dim=None, seq_len=8, device="cpu"):
    """
    Trainiert mit FTLE-Regularisierung.
    lambda_ftle=0: normales Training.
    lambda_ftle>0: zieht ftle_mean gegen target_ftle.
    """
    if task_dim is None:
        task_dim = len(TASKS)
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim, tau=tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)

    history = []
    for step in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = torch.zeros(x.shape[0], task_dim, device=device)
        tv[:, idx] = 1.0
        out, states = model(x, tv)
        task_loss = lossfn(out, y)
        if lambda_ftle > 0:
            z_last = states[:, -1]
            ftle_loss = ftle_penalty(model, z_last, target_ftle)
            loss = task_loss + lambda_ftle * ftle_loss
        else:
            loss = task_loss
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 250 == 0:
            with torch.no_grad():
                z_last = states[:, -1]
                H = model.hidden_dim
                sech2 = 1.0 - torch.tanh(z_last) ** 2
                J = -torch.eye(H, device=device) / model.tau
                J = J.unsqueeze(0) + model.W.unsqueeze(0) * sech2.unsqueeze(1)
                sv = torch.linalg.svdvals(J)
                ftle_now = torch.log(sv + 1e-12).mean().item()
            history.append((step, task_loss.item(), ftle_now))

    return model, task_loss.item(), history
