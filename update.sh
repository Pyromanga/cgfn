#!/usr/bin/env bash
set -e

echo "=== Erweitere model.py um Aufgaben-Konditionierung ==="
cat > src/cgfn/model.py <<'EOF'
import torch
import torch.nn as nn


class ContinuousRNN(nn.Module):
    """dz/dt = -z/tau + W tanh(z) + U [x; task_onehot], RK2-integriert."""

    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32,
                 tau=1.0, dt=0.1, substeps=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.tau, self.dt, self.substeps = tau, dt, substeps
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.3)
        self.U = nn.Linear(input_dim + task_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def dynamics(self, z, x, task_vec):
        u = torch.cat([x, task_vec], dim=-1) if self.task_dim > 0 else x
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
echo "=== Neue Multi-Task-Version des Experiments ==="
cat > experiments/run_smoke.py <<'EOF'
import torch
from collections import defaultdict
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.criticality import branching_ratio
from cgfn.geometry import mean_state_signature, signature_distance

device = "cpu"
task_index = {t: i for i, t in enumerate(TASKS)}
task_dim = len(TASKS)


def one_hot(task, batch_size, device):
    idx = task_index[task]
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, idx] = 1.0
    return v


def train_multitask(seed, steps=4000, lr=1e-3, seq_len=8, hidden_dim=64):
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for step in range(steps):
        task = TASKS[step % len(TASKS)]
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = one_hot(task, x.shape[0], device)
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    # finale losses
    losses = {}
    for task in TASKS:
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = one_hot(task, x.shape[0], device)
        with torch.no_grad():
            out, _ = model(x, tv)
            losses[task] = lossfn(out, y).item()
    return model, losses


def collect_signature(model, task, seq_len=8, batch=64):
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    tv = one_hot(task, batch, device)
    with torch.no_grad():
        _, states = model(x, tv)
    return mean_state_signature(states), branching_ratio(states).mean().item()


def main():
    seeds = [0, 1, 2]
    print("=== Multi-Task-Modell, gleiche Gewichte fuer alle Aufgaben ===\n")
    all_sigs = defaultdict(list)
    all_branches = defaultdict(list)
    for seed in seeds:
        model, losses = train_multitask(seed)
        print(f"Seed {seed} losses: " +
              "  ".join(f"{t}={losses[t]:.4f}" for t in TASKS))
        for task in TASKS:
            sig, br = collect_signature(model, task)
            all_sigs[task].append(sig)
            all_branches[task].append(br)
        print()

    print("=== Signaturen pro Aufgabe (Mittel ueber Seeds) ===")
    means = {}
    for task in TASKS:
        S = torch.stack(all_sigs[task])
        means[task] = S.mean(dim=0)
        intra = ((S - means[task]) ** 2).sum(dim=1).mean().item()
        print(f"  {task:8s}: norm={means[task].norm():.4f}  intra_var={intra:.4f}")

    print("\n=== Inter vs Intra (jetzt mit geteilter Basis) ===")
    for i, a in enumerate(TASKS):
        for b in TASKS[i+1:]:
            inter = signature_distance(means[a], means[b])
            intra = (((torch.stack(all_sigs[a]) - means[a])**2).sum(dim=1).mean().item()
                     + ((torch.stack(all_sigs[b]) - means[b])**2).sum(dim=1).mean().item()) / 2
            ratio = inter / (intra + 1e-8)
            print(f"  {a:8s} vs {b:8s}: inter={inter:.4f} intra={intra:.4f} ratio={ratio:.4f}")

    print("\n=== Branching Ratio pro Aufgabe ===")
    for task in TASKS:
        b = torch.tensor(all_branches[task])
        print(f"  {task:8s}: mean={b.mean():.4f}  std={b.std():.4f}")


if __name__ == "__main__":
    main()
EOF

echo
echo "=== Syntax-Check ==="
python -c "import ast; ast.parse(open('src/cgfn/model.py').read()); print('model.py OK')"
python -c "import ast; ast.parse(open('experiments/run_smoke.py').read()); print('run_smoke.py OK')"

echo
echo "=== Lauf (dauert ein paar Minuten) ==="
python experiments/run_smoke.py 2>&1 | tee smoke4.txt

echo
echo "=== Fertig. Ausgabe: smoke4.txt ==="
