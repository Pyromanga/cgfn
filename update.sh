#!/usr/bin/env bash
set -e

echo "=== DSA-Bug fixen: eine Rotation fuer beide Zeitschritte ==="
cat > src/cgfn/dsa.py <<'EOF'
import torch


@torch.no_grad()
def collect_states(model, task, seq_len=8, batch=32, device="cpu"):
    from cgfn.tasks import make_batch, TASKS
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    if hasattr(model, 'task_dim') and model.task_dim > 0:
        idx = TASKS.index(task)
        tv = torch.zeros(batch, model.task_dim, device=device)
        tv[:, idx] = 1.0
        _, states = model(x, tv)
    else:
        _, states = model(x)
    if states.dim() == 4:
        states = states.squeeze(0)
    return states


def _procrustes_Q(X, Y):
    """Findet Q mit X @ Q ~ Y. X, Y: (n, H). Rueckgabe: Q (H, H), orthogonal."""
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    U, _, Vh = torch.linalg.svd(Xc.T @ Yc)
    return U @ Vh


def _apply_Q(X, Q, Y_ref):
    Xc = X - X.mean(dim=0, keepdim=True)
    return Xc @ Q + Y_ref.mean(dim=0, keepdim=True)


@torch.no_grad()
def dsa(model_a, model_b, task, seq_len=8, device="cpu"):
    """DSA mit EINER Procrustes-Rotation fuer z und z_next."""
    A = collect_states(model_a, task, seq_len, device=device)
    B = collect_states(model_b, task, seq_len, device=device)
    H = A.shape[-1]

    za = A[:, :-1, :].reshape(-1, H)
    za_next = A[:, 1:, :].reshape(-1, H)
    zb = B[:, :-1, :].reshape(-1, H)
    zb_next = B[:, 1:, :].reshape(-1, H)

    # EINE Rotation, angewendet auf beide Zeitschritte
    Q = _procrustes_Q(zb, za)
    zb_aligned = _apply_Q(zb, Q, za)
    zb_next_aligned = _apply_Q(zb_next, Q, za_next)

    lambda_reg = 1e-3
    M_a = za.T @ za + lambda_reg * torch.eye(H, device=device)
    W_a = torch.linalg.solve(M_a, za.T @ za_next)

    err_self = ((za @ W_a - za_next) ** 2).mean().item()
    err_cross = ((zb_aligned @ W_a - zb_next_aligned) ** 2).mean().item()
    raw = 1.0 - (err_cross / (err_self + 1e-12))
    return {
        "err_self": err_self,
        "err_cross": err_cross,
        "raw": raw,
        "score": max(0.0, min(1.0, raw)),
    }
EOF

echo
echo "=== Hauptexperiment: CT-RNN-Hypothese haerten ==="
cat > experiments/frontier.py <<'EOF'
import torch
from itertools import combinations
from cgfn.model import ContinuousRNN
from cgfn.baselines import GRUBaseline, LSTMBaseline, VanillaRNNBaseline
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.dsa import dsa

device = "cpu"
task_dim = len(TASKS)
SEQ_LEN = 8
HIDDEN = 32
STEPS = 2000
SEEDS = [0, 1, 2, 3, 4]
TASK_SUBSET = ["copy", "reverse", "parity", "mod3"]


def one_hot(task_idx, batch_size):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train(ModelClass, seed, task, steps=STEPS, lr=1e-3):
    torch.manual_seed(seed)
    model = ModelClass(task_dim=task_dim, hidden_dim=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    for _ in range(steps):
        x, y = make_batch(task, seq_len=SEQ_LEN, device=device)
        tv = one_hot(idx, x.shape[0])
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def reps(model, task, fixed_x):
    idx = TASKS.index(task)
    tv = one_hot(idx, fixed_x.shape[0])
    with torch.no_grad():
        _, s = model(fixed_x, tv)
    if s.dim() == 4:
        s = s.squeeze(0)
    B, T, H = s.shape
    return s.reshape(B * T, H)


def main():
    torch.manual_seed(999)
    x_fixed = torch.randint(0, 2, (32, SEQ_LEN, 1), device=device).float()

    archs = [
        ("CT-RNN", ContinuousRNN),
        ("GRU", GRUBaseline),
        ("LSTM", LSTMBaseline),
        ("Vanilla", VanillaRNNBaseline),
    ]

    print("=" * 70)
    print(f"CT-RNN-Hypothese: 5 Seeds x 4 Aufgaben x 4 Architekturen")
    print(f"hidden_dim={HIDDEN}  steps={STEPS}")
    print("=" * 70)

    for name, MC in archs:
        print(f"\n=== {name} ===")
        models = {}
        for seed in SEEDS:
            for task in TASK_SUBSET:
                models[(seed, task)] = train(MC, seed, task)

        # Cross-seed same-task CKA und DSA
        print("  Cross-seed same-task (Mittel ueber 10 Paare, 4 Aufgaben):")
        for task in TASK_SUBSET:
            cka_vals, dsa_vals = [], []
            for i, a in enumerate(SEEDS):
                for b in SEEDS[i+1:]:
                    ra = reps(models[(a, task)], task, x_fixed)
                    rb = reps(models[(b, task)], task, x_fixed)
                    cka_vals.append(cka(ra, rb, kernel="linear"))
                    d = dsa(models[(a, task)], models[(b, task)], task)
                    dsa_vals.append(d["raw"])
            cm = sum(cka_vals) / len(cka_vals)
            dm = sum(dsa_vals) / len(dsa_vals)
            print(f"    {task:10s}: CKA={cm:.4f}  DSA_raw={dm:+.4f}")

        # Same-seed cross-task CKA und DSA
        print("  Same-seed cross-task (Mittel ueber 3 Seeds, alle Paare):")
        for i, ta in enumerate(TASK_SUBSET):
            for j, tb in enumerate(TASK_SUBSET):
                if j <= i:
                    continue
                cka_vals, dsa_vals = [], []
                for seed in SEEDS:
                    ra = reps(models[(seed, ta)], ta, x_fixed)
                    rb = reps(models[(seed, tb)], tb, x_fixed)
                    cka_vals.append(cka(ra, rb, kernel="linear"))
                    d = dsa(models[(seed, ta)], models[(seed, tb)], ta)
                    dsa_vals.append(d["raw"])
                cm = sum(cka_vals) / len(cka_vals)
                dm = sum(dsa_vals) / len(dsa_vals)
                print(f"    {ta:10s} vs {tb:10s}: CKA={cm:.4f}  DSA_raw={dm:+.4f}")


if __name__ == "__main__":
    main()
EOF

python -c "import ast; ast.parse(open('src/cgfn/dsa.py').read()); print('dsa OK')"
python -c "import ast; ast.parse(open('experiments/frontier.py').read()); print('frontier OK')"

echo
echo "=== Lauf (20-40 min) ==="
python experiments/frontier.py 2>&1 | tee frontier_results3.txt
echo
echo "=== Fertig: frontier_results3.txt ==="
