"""
Sweet-Spot-Finder: Findet tau, bei dem die Aufgaben-Trennung
maximal ist, mit FTLE als mechanistischem Leitsignal.
"""
import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.ftle import ftle_summary


@torch.no_grad()
def measure_at_tau(tau, tasks, seeds, steps=1500, lr=1e-3,
                   hidden_dim=32, seq_len=8, device="cpu",
                   task_dim=None):
    """
    Trainiert Modelle für gegebene Aufgaben bei gegebenem tau,
    liefert CKA-Matrix und FTLE-Werte.
    """
    if task_dim is None:
        task_dim = len(TASKS)

    def one_hot(idx, bs):
        v = torch.zeros(bs, task_dim, device=device)
        v[:, idx] = 1.0
        return v

    def train(seed, task):
        torch.manual_seed(seed)
        m = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim, tau=tau)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        lossfn = torch.nn.BCEWithLogitsLoss()
        idx = TASKS.index(task)
        for _ in range(steps):
            x, y = make_batch(task, seq_len=seq_len, device=device)
            tv = one_hot(idx, x.shape[0])
            out, _ = m(x, tv)
            loss = lossfn(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        return m, loss.item()

    torch.manual_seed(999)
    x_fixed = torch.randint(0, 2, (32, seq_len, 1), device=device).float()

    def reps(m, task):
        idx = TASKS.index(task)
        tv = one_hot(idx, x_fixed.shape[0])
        with torch.no_grad():
            _, s = m(x_fixed, tv)
        if s.dim() == 4:
            s = s.squeeze(0)
        B, T, H = s.shape
        return s.reshape(B * T, H)

    models = {}
    losses = {}
    for seed in seeds:
        for task in tasks:
            m, l = train(seed, task)
            models[(seed, task)] = m
            losses[(seed, task)] = l

    # CKA-Matrix (gemittelt über Seeds)
    cka_matrix = {}
    for i, ta in enumerate(tasks):
        for j, tb in enumerate(tasks):
            if j <= i:
                continue
            vals = []
            for seed in seeds:
                ra = reps(models[(seed, ta)], ta)
                rb = reps(models[(seed, tb)], tb)
                vals.append(cka(ra, rb, kernel="linear"))
            cka_matrix[(ta, tb)] = sum(vals) / len(vals)

    # FTLE (gemittelt über Seeds und Aufgaben)
    ftle_vals = []
    for seed in seeds:
        for task in tasks:
            m = models[(seed, task)]
            s = reps(m, task).unsqueeze(0)  # (1, n, H)
            ftle = ftle_summary(m, s)
            ftle_vals.append(ftle["ftle_mean"])

    # Mittlerer Loss pro Aufgabe
    mean_losses = {}
    for task in tasks:
        vals = [losses[(s, task)] for s in seeds]
        mean_losses[task] = sum(vals) / len(vals)

    return {
        "tau": tau,
        "cka_matrix": cka_matrix,
        "ftle_mean": sum(ftle_vals) / len(ftle_vals),
        "losses": mean_losses,
    }


def find_sweet_spot(tau_grid, tasks, seeds, **kwargs):
    """
    Iteriert über tau_grid, misst CKA und FTLE, findet das Optimum.
    Kriterium: minimale mittlere CKA zwischen Aufgaben, solange
    die Losses der lernbaren Aufgaben unter Schwelle bleiben.
    """
    results = []
    for tau in tau_grid:
        print(f"\n>>> tau = {tau}")
        r = measure_at_tau(tau, tasks, seeds, **kwargs)
        # Mittlere Cross-Task CKA
        vals = list(r["cka_matrix"].values())
        r["cka_mean"] = sum(vals) / len(vals)
        # Schlechtester Loss (sollte klein bleiben)
        r["worst_loss"] = max(r["losses"].values())
        results.append(r)
        print(f"    CKA_mean={r['cka_mean']:.4f}  "
              f"ftle={r['ftle_mean']:+.4f}  "
              f"worst_loss={r['worst_loss']:.4f}")
    return results
