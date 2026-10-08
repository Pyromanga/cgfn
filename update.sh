#!/usr/bin/env bash
set -e

echo "=== DSA mit Procrustes-Alignment ==="
cat > src/cgfn/dsa.py <<'EOF'
"""
Dynamical Similarity Analysis (DSA) mit Procrustes-Alignment.
Ohne Alignment ist DSA bedeutungslos: verschiedene Modelle leben in
verschiedenen Koordinatensystemen. Procrustes findet die orthogonale
Transformation, die die Zustaende aligniert.
"""
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


def procrustes_align(X, Y):
    """
    Findet orthogonale Matrix Q, sodass X @ Q ~ Y.
    X: (n, H), Y: (n, H). Rueckgabe: Y_aligned.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    U, _, Vh = torch.linalg.svd(Xc.T @ Yc)
    Q = U @ Vh
    return Xc @ Q + Y.mean(dim=0, keepdim=True)


@torch.no_grad()
def dsa(model_a, model_b, task, seq_len=8, device="cpu", align=True):
    """DSA mit optionalem Procrustes-Alignment."""
    A = collect_states(model_a, task, seq_len, device=device)
    B = collect_states(model_b, task, seq_len, device=device)
    za = A[:, :-1, :].reshape(-1, A.shape[-1])
    za_next = A[:, 1:, :].reshape(-1, A.shape[-1])
    zb = B[:, :-1, :].reshape(-1, B.shape[-1])
    zb_next = B[:, 1:, :].reshape(-1, B.shape[-1])

    if align:
        # Aligniere B auf A
        zb_aligned = procrustes_align(zb, za)
        zb_next_aligned = procrustes_align(zb_next, za_next)
    else:
        zb_aligned = zb
        zb_next_aligned = zb_next

    H = za.shape[1]
    lambda_reg = 1e-3

    # W_a aus A
    M_a = za.T @ za + lambda_reg * torch.eye(H, device=device)
    W_a = torch.linalg.solve(M_a, za.T @ za_next)

    # Self-Error auf A
    err_self = ((za @ W_a - za_next) ** 2).mean()

    # Cross-Error: W_a auf aligned B
    err_cross = ((zb_aligned @ W_a - zb_next_aligned) ** 2).mean()

    score = 1.0 - (err_cross / (err_self + 1e-12)).item()
    return max(0.0, min(1.0, score))
EOF

echo
echo "=== INSD: hidden_dim konsistent ==="
cat > src/cgfn/insd.py <<'EOF'
import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch

device = "cpu"


def one_hot(task_idx, batch_size, task_dim):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train_with_insd(seed, task, reference_states=None, steps=4000,
                    lr=1e-3, seq_len=8, hidden_dim=32,
                    task_dim=len(TASKS), lambda_insd=0.5):
    torch.manual_seed(seed)
    task_idx = TASKS.index(task)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()

    if reference_states is None:
        reference_states = []

    if len(reference_states) > 0:
        R = torch.cat(reference_states, dim=0)
        # Orthonormale Basis der Referenzrichtungen, H x k
        Q, _ = torch.linalg.qr(R.T)
    else:
        Q = None

    for step in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = one_hot(task_idx, x.shape[0], task_dim)
        out, states = model(x, tv)
        task_loss = lossfn(out, y)

        if Q is not None:
            B, T, H = states.shape
            Z = states.reshape(B * T, H)
            # Q ist (H, k), Projektion Z @ Q ist (B*T, k)
            proj = Z @ Q
            insd_penalty = (proj ** 2).mean()
            loss = task_loss + lambda_insd * insd_penalty
        else:
            loss = task_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, task_loss.item()


@torch.no_grad()
def collect_states_for_ref(model, task, seq_len=8, batch=32):
    task_idx = TASKS.index(task)
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    tv = one_hot(task_idx, batch, len(TASKS))
    _, states = model(x, tv)
    B, T_len, H = states.shape
    return states.reshape(B * T_len, H)
EOF

echo
echo "=== Frontier-Experiment: nur die kaputten Teile ersetzen, hidden_dim=32 ==="
cat > experiments/frontier.py <<'EOF'
import torch
from itertools import combinations
from cgfn.model import ContinuousRNN
from cgfn.baselines import GRUBaseline, LSTMBaseline, VanillaRNNBaseline
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.dsa import dsa
from cgfn.insd import train_with_insd, collect_states_for_ref

device = "cpu"
task_dim = len(TASKS)
SEQ_LEN = 8
HIDDEN = 32


def one_hot(task_idx, batch_size):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train_model(ModelClass, seed, task, steps=4000, lr=1e-3):
    torch.manual_seed(seed)
    model = ModelClass(task_dim=task_dim, hidden_dim=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    task_idx = TASKS.index(task)
    for _ in range(steps):
        x, y = make_batch(task, seq_len=SEQ_LEN, device=device)
        tv = one_hot(task_idx, x.shape[0])
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()


def collect_reps(model, task, fixed_x):
    task_idx = TASKS.index(task)
    tv = one_hot(task_idx, fixed_x.shape[0])
    with torch.no_grad():
        _, states = model(fixed_x, tv)
    if states.dim() == 4:
        states = states.squeeze(0)
    B, T, H = states.shape
    return states.reshape(B * T, H)


def main():
    seeds = [0, 1]
    torch.manual_seed(999)
    x_fixed = torch.randint(0, 2, (32, SEQ_LEN, 1), device=device).float()

    print("=" * 60)
    print("A) Architektur-Vergleich (hidden_dim=%d)" % HIDDEN)
    print("=" * 60)

    for name, ModelClass in [
        ("CT-RNN", ContinuousRNN),
        ("GRU", GRUBaseline),
        ("LSTM", LSTMBaseline),
        ("VanillaRNN", VanillaRNNBaseline),
    ]:
        print(f"\n--- {name} ---")
        models = []
        for s in seeds:
            m, loss = train_model(ModelClass, s, "copy")
            models.append(m)
            print(f"  seed {s} copy loss: {loss:.4f}")

        reps = [collect_reps(m, "copy", x_fixed) for m in models]
        if len(reps) >= 2:
            c = cka(reps[0], reps[1], kernel="linear")
            print(f"  CKA cross-seed copy: {c:.4f}")

        if len(models) >= 2:
            d = dsa(models[0], models[1], "copy", align=True)
            print(f"  DSA cross-seed copy (aligned): {d:.4f}")

        m = models[0]
        reps_c = collect_reps(m, "copy", x_fixed)
        reps_p = collect_reps(m, "parity", x_fixed)
        c = cka(reps_c, reps_p, kernel="linear")
        print(f"  CKA copy vs parity (same seed): {c:.4f}")

    print("\n" + "=" * 60)
    print("B) INSD: Simplicity Bias brechen")
    print("=" * 60)

    print("\n--- Baseline (kein INSD) ---")
    for task in ["parity", "mod3"]:
        m, loss = train_with_insd(0, task, reference_states=None, hidden_dim=HIDDEN)
        print(f"  {task}: loss={loss:.4f}")

    print("\n--- INSD mit copy als Referenz ---")
    copy_model, _ = train_model(ContinuousRNN, 0, "copy")
    ref_states = [collect_states_for_ref(copy_model, "copy")]

    for task in ["parity", "mod3"]:
        m, loss = train_with_insd(0, task, reference_states=ref_states,
                                  lambda_insd=0.5, hidden_dim=HIDDEN)
        print(f"  {task}: loss={loss:.4f} (INSD)")

    print("\n--- INSD mit copy+reverse als Referenz ---")
    rev_model, _ = train_model(ContinuousRNN, 0, "reverse")
    ref_states = [
        collect_states_for_ref(copy_model, "copy"),
        collect_states_for_ref(rev_model, "reverse"),
    ]
    for task in ["parity", "mod3"]:
        m, loss = train_with_insd(0, task, reference_states=ref_states,
                                  lambda_insd=0.5, hidden_dim=HIDDEN)
        print(f"  {task}: loss={loss:.4f} (INSD 2 refs)")

    print("\n=== FERTIG ===")


if __name__ == "__main__":
    main()
EOF

echo
echo "=== Syntax-Check ==="
for f in src/cgfn/dsa.py src/cgfn/insd.py experiments/frontier.py; do
    python -c "import ast; ast.parse(open('$f').read())" 2>&1 \
      && echo "OK: $f" \
      || echo "FAIL: $f"
done

echo
echo "=== Lauf (dauert 20-40 min) ==="
python experiments/frontier.py 2>&1 | tee frontier_results2.txt
echo
echo "=== Fertig: frontier_results2.txt ==="
