#!/usr/bin/env bash
set -e

echo "=== run_smoke.py akzeptiert CLI-Parameter ==="
cat > experiments/run_smoke.py <<'EOF'
import argparse
import torch
from collections import defaultdict
from itertools import combinations
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka

device = "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2",
                   help="Komma-getrennte Seeds, z.B. 0,1,2,3,4")
    p.add_argument("--tasks", type=str, default=",".join(TASKS),
                   help="Komma-getrennte Aufgaben")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def one_hot(task_idx, batch_size, task_dim):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train(seed, task, task_list, steps, lr, seq_len, hidden_dim):
    torch.manual_seed(seed)
    task_dim = len(task_list)
    task_idx = task_list.index(task)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = one_hot(task_idx, x.shape[0], task_dim)
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()


def reps(model, task, task_list, fixed_x):
    task_idx = task_list.index(task)
    tv = one_hot(task_idx, fixed_x.shape[0], len(task_list))
    with torch.no_grad():
        _, s = model(fixed_x, tv)
    B, T, H = s.shape
    return s.reshape(B * T, H)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",")

    print(f"seeds={seeds}")
    print(f"tasks={tasks}")
    print(f"steps={args.steps}  hidden_dim={args.hidden_dim}  "
          f"seq_len={args.seq_len}  lr={args.lr}")
    print()

    torch.manual_seed(999)
    x_fixed = torch.randint(0, 2, (32, args.seq_len, 1), device=device).float()

    models = {}
    losses = {}
    print("=== Losses ===")
    for seed in seeds:
        for task in tasks:
            m, l = train(seed, task, tasks, args.steps, args.lr,
                         args.seq_len, args.hidden_dim)
            models[(seed, task)] = m
            losses[(seed, task)] = l
            print(f"  seed {seed} {task:14s} loss={l:.4f}")

    print("\n=== CKA-Matrix pro Seed ===")
    for seed in seeds:
        print(f"  seed {seed}:")
        for a, b in combinations(tasks, 2):
            ra = reps(models[(seed, a)], a, tasks, x_fixed)
            rb = reps(models[(seed, b)], b, tasks, x_fixed)
            c = cka(ra, rb, kernel="linear")
            mark = "  <-- verdaechtig" if c > 0.9 else ""
            print(f"    {a:14s} vs {b:14s}: CKA={c:.4f}{mark}")

    print("\n=== Cross-Seed Stabilität pro Aufgabe ===")
    if len(seeds) >= 2:
        for task in tasks:
            vals = []
            for a, b in combinations(seeds, 2):
                ra = reps(models[(a, task)], task, tasks, x_fixed)
                rb = reps(models[(b, task)], task, tasks, x_fixed)
                vals.append(cka(ra, rb, kernel="linear"))
            mean = sum(vals) / len(vals)
            print(f"  {task:14s}: mean CKA={mean:.4f}")


if __name__ == "__main__":
    main()
EOF

echo
echo "=== smoke.yml mit workflow_dispatch Inputs ==="
cat > .github/workflows/smoke.yml <<'EOF'
name: cgfn-smoke

on:
  push:
  workflow_dispatch:
    inputs:
      seeds:
        description: "Seeds, z.B. 0,1,2,3,4"
        default: "0,1,2"
      tasks:
        description: "Aufgaben, z.B. copy,parity,mod3"
        default: "copy,reverse,parity,mod3,delayed_xor"
      steps:
        description: "Trainingsschritte"
        default: "4000"
      hidden_dim:
        description: "hidden_dim"
        default: "64"
      seq_len:
        description: "seq_len"
        default: "8"

jobs:
  smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install torch (CPU)
        run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Install package with dev extras
        run: pip install -e ".[dev]"
      - name: Unit tests
        run: python -m pytest -q
      - name: Smoke experiment
        run: |
          python experiments/run_smoke.py \
            --seeds "${{ inputs.seeds || '0,1,2' }}" \
            --tasks "${{ inputs.tasks || 'copy,reverse,parity,mod3,delayed_xor' }}" \
            --steps "${{ inputs.steps || '4000' }}" \
            --hidden_dim "${{ inputs.hidden_dim || '64' }}" \
            --seq_len "${{ inputs.seq_len || '8' }}" \
            2>&1 | tee smoke_results.txt
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: smoke-results
          path: smoke_results.txt
          retention-days: 14
EOF

echo
echo "=== Syntax-Check ==="
python -c "import ast; ast.parse(open('experiments/run_smoke.py').read()); print('run_smoke.py OK')"
python -c "import yaml; yaml.safe_load(open('.github/workflows/smoke.yml')); print('smoke.yml OK')"

echo
echo "=== Lokal testen (schnell) ==="
python experiments/run_smoke.py --seeds 0 --tasks copy,parity --steps 200 2>&1 | tail -10

echo
echo "=== FERTIG ==="
echo "Push:  git add . && git commit -m 'parameterize smoke test' && git push"
echo "Dann auf GitHub: Actions -> cgfn-smoke -> Run workflow"
