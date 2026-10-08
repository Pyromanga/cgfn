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


def train_multitask(seed, steps=4000, lr=1e-3, seq_len=8, hidden_dim=64,
                    task_scale=1.0):
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim,
                          task_scale=task_scale)
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


def reps_for(model, task, seq_len=8, batch=32, fixed_x=None):
    if fixed_x is None:
        x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    else:
        x = fixed_x
    tv = one_hot(task, x.shape[0])
    with torch.no_grad():
        _, states = model(x, tv)
    B, T, H = states.shape
    return states.reshape(B * T, H)


def losses_for(model, seq_len=8):
    lossfn = torch.nn.BCEWithLogitsLoss()
    out = {}
    for task in TASKS:
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = one_hot(task, x.shape[0])
        with torch.no_grad():
            pred, _ = model(x, tv)
            out[task] = lossfn(pred, y).item()
    return out


def main():
    seeds = [0, 1, 2]
    seq_len = 8

    # Fester Input für Task-Sensitivitäts-Test
    torch.manual_seed(999)
    x_fixed = torch.randint(0, 2, (32, seq_len, 1), device=device).float()

    print("=== A) Baseline: task_scale=1.0 ===")
    models_a = [train_multitask(s, task_scale=1.0) for s in seeds]
    for s, m in zip(seeds, models_a):
        losses = losses_for(m)
        print(f"  seed {s}: " +
              "  ".join(f"{t}={losses[t]:.3f}" for t in TASKS))

    print("\n=== B) Task-Sensitivitaet (fester Input, variiere Task-Vektor) ===")
    print("   CKA nahe 1.0 = Modell ignoriert Task-Vektor")
    print("   CKA deutlich < 1.0 = Modell reagiert auf Task")
    for s, m in zip(seeds, models_a):
        reps = {t: reps_for(m, t, fixed_x=x_fixed) for t in TASKS}
        print(f"  seed {s}:")
        for a, b in combinations(TASKS, 2):
            c = cka(reps[a], reps[b], kernel="linear")
            print(f"    {a:8s} vs {b:8s}: CKA={c:.4f}")

    print("\n=== C) Verstaerkter Task-Vektor: task_scale=5.0 ===")
    models_c = [train_multitask(s, task_scale=5.0) for s in seeds]
    for s, m in zip(seeds, models_c):
        losses = losses_for(m)
        print(f"  seed {s}: " +
              "  ".join(f"{t}={losses[t]:.3f}" for t in TASKS))

    print("\n=== D) Task-Sensitivitaet mit task_scale=5.0 ===")
    for s, m in zip(seeds, models_c):
        reps = {t: reps_for(m, t, fixed_x=x_fixed) for t in TASKS}
        print(f"  seed {s}:")
        for a, b in combinations(TASKS, 2):
            c = cka(reps[a], reps[b], kernel="linear")
            print(f"    {a:8s} vs {b:8s}: CKA={c:.4f}")

    print("\n=== E) Zusammenfassung ===")
    print("Wenn Baseline: copy<->parity ~0.98 und Task-Sensitivitaet auch ~0.98")
    print("-> Mode Collapse bestaetigt. Modell ignoriert Task-Vektor.")
    print("Wenn task_scale=5.0 die Losses verbessert und Task-Sensitivitaet senkt")
    print("-> War es ein Skalierungsproblem, kein Kapazitaetsproblem.")


if __name__ == "__main__":
    main()
