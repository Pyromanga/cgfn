"""
Unified Criticality Analysis: Testet, ob der Sweet Spot
ein universelles Prinzip rekurrenter Architekturen ist.
Architekturen: CT-RNN, GRU, LSTM, VanillaRNN.
Hyperparameter: Gewichts-Init-Skalierung (gain) fuer alle,
                tau nur fuer CT-RNN.
Mass: FTLE (via Jacobi-SVD), CKA (Repraesentationstrennung).
"""
import torch
import torch.nn as nn
from cgfn.model import ContinuousRNN
from cgfn.baselines import GRUBaseline, LSTMBaseline, VanillaRNNBaseline
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.ftle import ftle_summary


ARCHS = {
    "CT-RNN": ContinuousRNN,
    "GRU": GRUBaseline,
    "LSTM": LSTMBaseline,
    "Vanilla": VanillaRNNBaseline,
}


def scale_weights(model, gain):
    """Skaliert die rekurrenten Gewichte um Faktor gain."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "W" in name or "weight_ih" in name or "weight_hh" in name:
                if p.dim() >= 2:
                    p.mul_(gain)


def train(seed, task, ModelClass, hidden_dim=32, steps=1500, lr=1e-3,
          tau=1.5, gain=1.0, task_dim=None, seq_len=8, device="cpu"):
    """Trainiert ein Modell mit gegebener Init-Skalierung."""
    if task_dim is None:
        task_dim = len(TASKS)
    torch.manual_seed(seed)
    if ModelClass is ContinuousRNN:
        model = ModelClass(task_dim=task_dim, hidden_dim=hidden_dim, tau=tau)
    else:
        model = ModelClass(task_dim=task_dim, hidden_dim=hidden_dim)
    if gain != 1.0:
        scale_weights(model, gain)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    for _ in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = torch.zeros(x.shape[0], task_dim, device=device)
        tv[:, idx] = 1.0
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()


@torch.no_grad()
def compute_ftle(model, task, task_dim, seq_len=8, device="cpu"):
    """FTLE ueber Jacobi der rekurrenten Gewichte."""
    H = model.hidden_dim
    idx = TASKS.index(task)
    x, _ = make_batch(task, batch_size=32, seq_len=seq_len, device=device)
    tv = torch.zeros(x.shape[0], task_dim, device=device)
    tv[:, idx] = 1.0
    _, states = model(x, tv)
    if states.dim() == 4:
        states = states.squeeze(0)
    z = states[:, -1]
    sech2 = 1.0 - torch.tanh(z) ** 2
    if hasattr(model, 'W'):
        J = -torch.eye(H, device=device) / model.tau
        J = J + model.W * sech2.unsqueeze(1)
    elif hasattr(model, 'gru'):
        J = torch.eye(H, device=device)
    else:
        J = torch.eye(H, device=device)
    sv = torch.linalg.svdvals(J)
    log_sv = torch.log(sv + 1e-12)
    return log_sv.mean().item()


def find_sweet_spot_arch(ModelClass, arch_name, hyper_grid, tasks,
                         seeds, hidden_dim=32, **kwargs):
    """Findet den Sweet Spot fuer eine Architektur ueber ein Gitter."""
    print(f"\n{'=' * 70}")
    print(f"ARCHITEKTUR: {arch_name}  (hidden_dim={hidden_dim})")
    print(f"{'=' * 70}")
    print(f"{'gain/tau':>10s} {'CKA_mean':>10s} {'ftle_mean':>10s} "
          f"{'copy_loss':>10s} {'worst_loss':>10s}")

    results = []
    for param in hyper_grid:
        if arch_name == "CT-RNN":
            train_kwargs = {"tau": param}
        else:
            train_kwargs = {"gain": param}

        all_reps = {}
        all_losses = {}
        ftle_vals = []

        for seed in seeds:
            for task in tasks:
                model, loss = train(seed, task, ModelClass,
                                    hidden_dim=hidden_dim, **train_kwargs,
                                    **kwargs)
                all_losses[(seed, task)] = loss
                ftle_vals.append(compute_ftle(model, task, len(TASKS)))

                # Reps
                idx = TASKS.index(task)
                x, _ = make_batch(task, batch_size=32, seq_len=8,
                                  device=kwargs.get("device", "cpu"))
                tv = torch.zeros(x.shape[0], len(TASKS),
                                 device=kwargs.get("device", "cpu"))
                tv[:, idx] = 1.0
                with torch.no_grad():
                    _, s = model(x, tv)
                if s.dim() == 4:
                    s = s.squeeze(0)
                B, T, H = s.shape
                all_reps[(seed, task)] = s.reshape(B * T, H)

        # CKA-Matrix
        cka_vals = []
        for i, ta in enumerate(tasks):
            for j, tb in enumerate(tasks):
                if j <= i:
                    continue
                for seed in seeds:
                    ra = all_reps[(seed, ta)]
                    rb = all_reps[(seed, tb)]
                    cka_vals.append(cka(ra, rb, kernel="linear"))
        cka_mean = sum(cka_vals) / len(cka_vals) if cka_vals else float("nan")

        mean_losses = {}
        for task in tasks:
            vals = [all_losses[(s, task)] for s in seeds]
            mean_losses[task] = sum(vals) / len(vals)

        ftle_mean = sum(ftle_vals) / len(ftle_vals)
        worst_loss = max(mean_losses.values())
        copy_loss = mean_losses.get("copy", float("nan"))

        results.append({
            "param": param,
            "cka_mean": cka_mean,
            "ftle_mean": ftle_mean,
            "copy_loss": copy_loss,
            "worst_loss": worst_loss,
            "losses": mean_losses,
        })

        print(f"{param:10.3f} {cka_mean:10.4f} {ftle_mean:+10.4f} "
              f"{copy_loss:10.4f} {worst_loss:10.4f}")

    return results
