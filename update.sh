#!/usr/bin/env bash
set -e

cat > src/cgfn/tasks.py <<'EOF'
import torch

TASKS = ["copy", "reverse", "parity", "mod3", "delayed_xor"]


def make_batch(task, batch_size=32, seq_len=16, device="cpu"):
    x = torch.randint(0, 2, (batch_size, seq_len), device=device).float()
    if task == "copy":
        y = x
    elif task == "reverse":
        y = x.flip(1)
    elif task == "parity":
        y = torch.cumsum(x, dim=1) % 2
    elif task == "mod3":
        z = torch.randint(0, 3, (batch_size, seq_len), device=device).float()
        y = torch.cumsum(z, dim=1) % 3 / 2.0
        return z.unsqueeze(-1) / 2.0, y.unsqueeze(-1)
    elif task == "delayed_xor":
        B, T = x.shape
        y = torch.zeros_like(x)
        y[:, 2:] = (x[:, 2:] + x[:, :-2]) % 2
        y[:, :2] = x[:, :2]
    else:
        raise ValueError(task)
    return x.unsqueeze(-1), y.unsqueeze(-1)
EOF

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


def train_multitask(seed, steps=8000, lr=1e-3, seq_len=8, hidden_dim=64):
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
    torch.manual_seed(999)
    x_fixed = torch.randint(0, 2, (32, seq_len, 1), device=device).float()

    print("=== Losses (Zufall: copy=0.69, parity=0.69, mod3=0.63, dxor=0.69) ===")
    models = [train_multitask(s) for s in seeds]
    for s, m in zip(seeds, models):
        losses = losses_for(m)
        print(f"  seed {s}: " + "  ".join(f"{t}={losses[t]:.3f}" for t in TASKS))

    print("\n=== Task-Sensitivitaet, fester Input, CKA-Matrix ===")
    print("   (niedrig = getrennte Repraesentation)")
    for s, m in zip(seeds, models):
        reps = {t: reps_for(m, t, fixed_x=x_fixed) for t in TASKS}
        print(f"  seed {s}:")
        for a, b in combinations(TASKS, 2):
            c = cka(reps[a], reps[b], kernel="linear")
            mark = "  <-- verdaechtig" if c > 0.9 else ""
            print(f"    {a:12s} vs {b:12s}: CKA={c:.4f}{mark}")


if __name__ == "__main__":
    main()
EOF

python -c "import ast; ast.parse(open('src/cgfn/tasks.py').read()); print('tasks OK')"
python -c "import ast; ast.parse(open('experiments/run_smoke.py').read()); print('exp OK')"
python experiments/run_smoke.py 2>&1 | tee smoke7.txt
echo "=== smoke7.txt ==="
