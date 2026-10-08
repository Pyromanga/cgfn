#!/usr/bin/env bash
set -e

echo "=== Neue Datei: src/cgfn/model.py (mit skalierbarem Task-Vektor) ==="
cat > src/cgfn/model.py <<'EOF'
import torch
import torch.nn as nn


class ContinuousRNN(nn.Module):
    """dz/dt = -z/tau + W tanh(z) + U [x; task_vec], RK2-integriert."""

    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32,
                 tau=1.0, dt=0.1, substeps=5, task_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.task_scale = task_scale
        self.tau, self.dt, self.substeps = tau, dt, substeps
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.3)
        self.U = nn.Linear(input_dim + task_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def dynamics(self, z, x, task_vec):
        if self.task_dim > 0:
            u = torch.cat([x, task_vec * self.task_scale], dim=-1)
        else:
            u = x
        return -z / self.tau + torch.tanh(z) @ self.W.T + self.U(u)

    def forward(self, x_seq, task_vec=None):
        B, T, _ = x_seq.shape
        if task_vec is None:
            task_vec = torch.zeros(B, 0, device=x_seq.device)
        z = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        states = []
        for t in range(T):
            x = x_seq[:, t]
            for _ in range(self.substeps):
                k1 = self.dynamics(z, x, task_vec)
                k2 = self.dynamics(z + 0.5 * self.dt * k1, x, task_vec)
                z = z + self.dt * k2
            states.append(z)
        states = torch.stack(states, dim=1)
        return self.readout(states), states
EOF

echo
echo "=== Neues Experiment: Task-Sensitivitaet + Mode-Collapse-Check ==="
cat > experiments/run_smoke.py <<'EOF'
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
EOF

echo
echo "=== Syntax-Check ==="
python -c "import ast; ast.parse(open('src/cgfn/model.py').read()); print('model.py OK')"
python -c "import ast; ast.parse(open('experiments/run_smoke.py').read()); print('run_smoke.py OK')"

echo
echo "=== Lauf (dauert laenger: 6 Trainingslaeufe) ==="
python experiments/run_smoke.py 2>&1 | tee smoke6.txt

echo
echo "=== Fertig. Ausgabe: smoke6.txt ==="
