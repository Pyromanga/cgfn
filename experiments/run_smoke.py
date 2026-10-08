import torch
from collections import defaultdict
from itertools import combinations
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka

device = "cpu"
task_index = {t: i for i, t in enumerate(TASKS)}
task_dim = len(TASKS)


def one_hot(task, batch_size):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_index[task]] = 1.0
    return v


def train_multitask(seed, steps=4000, lr=1e-3, seq_len=8, hidden_dim=64):
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for step in range(steps):
        task = TASKS[step % len(TASKS)]
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = one_hot(task, x.shape[0])
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def collect_reps(model, task, seq_len=8, batch=32):
    """Gibt eine (batch*T, hidden) Matrix zurueck."""
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    tv = one_hot(task, batch)
    with torch.no_grad():
        _, states = model(x, tv)          # (B, T, H)
    B, T, H = states.shape
    return states.reshape(B * T, H)


def main():
    seeds = [0, 1, 2]
    print("=== Trainiere Multi-Task-Modelle ===")
    models = {}
    for s in seeds:
        print(f"  seed {s}...")
        models[s] = train_multitask(s)

    print("\n=== Sammle Repraesentationen ===")
    reps = {}
    for s in seeds:
        for task in TASKS:
            reps[(s, task)] = collect_reps(models[s], task)

    print("\n=== CKA: Seed-Stabilitaet pro Aufgabe (hoeher = stabiler) ===")
    for task in TASKS:
        vals = []
        for a, b in combinations(seeds, 2):
            vals.append(cka(reps[(a, task)], reps[(b, task)], kernel="linear"))
        mean = sum(vals) / len(vals)
        print(f"  {task:8s}: mean CKA={mean:.4f}  values={[f'{v:.3f}' for v in vals]}")

    print("\n=== CKA: Aufgaben-Trennung pro Seed (niedriger = besser getrennt) ===")
    for s in seeds:
        vals = []
        for a, b in combinations(TASKS, 2):
            vals.append(cka(reps[(s, a)], reps[(s, b)], kernel="linear"))
        mean = sum(vals) / len(vals)
        print(f"  seed {s}: mean CKA={mean:.4f}  values={[f'{v:.3f}' for v in vals]}")

    print("\n=== Die entscheidende Frage ===")
    print("Vergleiche: cross-seed-same-task CKA vs cross-task-same-seed CKA")
    print("Wenn ersteres >> letzteres: Repraesentationen sind aufgabenstabil.")
    print("Wenn vergleichbar: Aufgabe strukturiert den Zustandsraum nicht.")

    print("\n=== Zusatz: RBF-Kernel (nichtlinearer Vergleich) ===")
    for task in TASKS:
        vals = []
        for a, b in combinations(seeds, 2):
            vals.append(cka(reps[(a, task)], reps[(b, task)], kernel="rbf"))
        mean = sum(vals) / len(vals)
        print(f"  cross-seed {task:8s}: mean CKA={mean:.4f}")


if __name__ == "__main__":
    main()
