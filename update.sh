#!/usr/bin/env bash
set -e

echo "=== Schreibe experiments/run_smoke.py neu ==="
cat > experiments/run_smoke.py <<'EOF'
import torch
from collections import defaultdict
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.criticality import branching_ratio
from cgfn.geometry import mean_state_signature, signature_distance

device = "cpu"


def train_single(task, seed, steps=2000, lr=1e-3, seq_len=8, hidden_dim=64):
    torch.manual_seed(seed)
    model = ContinuousRNN(hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        out, _ = model(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()


def collect_signature(model, seq_len=8):
    x, _ = make_batch("copy", batch_size=64, seq_len=seq_len, device=device)
    with torch.no_grad():
        _, states = model(x)
    return mean_state_signature(states), branching_ratio(states).mean().item()


def main():
    seeds = [0, 1, 2, 3, 4]
    print("=== Teil 1: Pro Aufgabe, 5 Seeds ===\n")
    print(f"{'task':10s} {'seed':>4s} {'loss':>8s} {'branch':>8s}")
    sigs = defaultdict(list)
    branches = defaultdict(list)
    for task in TASKS:
        for seed in seeds:
            model, loss = train_single(task, seed)
            sig, br = collect_signature(model)
            sigs[task].append(sig)
            branches[task].append(br)
            print(f"{task:10s} {seed:4d} {loss:8.4f} {br:8.4f}")

    print("\n=== Intra-Aufgaben-Varianz (Mittel ueber Seeds) ===")
    means = {}
    for task in TASKS:
        S = torch.stack(sigs[task])
        mu = S.mean(dim=0)
        means[task] = mu
        var = ((S - mu) ** 2).sum(dim=1).mean().item()
        print(f"  {task:8s}: mean_norm={mu.norm():.4f}  intra_var={var:.4f}")

    print("\n=== Inter-Aufgaben-Distanz (zwischen Mittelwerten) ===")
    for i, a in enumerate(TASKS):
        for b in TASKS[i+1:]:
            d = signature_distance(means[a], means[b])
            print(f"  {a:8s} vs {b:8s}: {d:.4f}")

    print("\n=== Verhaeltnis Inter/Intra (groesser = besser trennbar) ===")
    for i, a in enumerate(TASKS):
        for b in TASKS[i+1:]:
            inter = signature_distance(means[a], means[b])
            intra = (((torch.stack(sigs[a]) - means[a])**2).sum(dim=1).mean().item()
                     + ((torch.stack(sigs[b]) - means[b])**2).sum(dim=1).mean().item()) / 2
            ratio = inter / (intra + 1e-8)
            print(f"  {a:8s} vs {b:8s}: ratio={ratio:.4f}")

    print("\n=== Branching Ratio pro Aufgabe ===")
    for task in TASKS:
        b = torch.tensor(branches[task])
        print(f"  {task:8s}: mean={b.mean():.4f}  std={b.std():.4f}")


if __name__ == "__main__":
    main()
EOF

echo
echo "=== Syntax-Check ==="
python -c "import ast; ast.parse(open('experiments/run_smoke.py').read()); print('OK')"

echo
echo "=== Fuehre Experiment aus (dauert ein paar Minuten) ==="
python experiments/run_smoke.py 2>&1 | tee smoke3.txt

echo
echo "=== Fertig ==="
echo "Ausgabe: smoke3.txt"
