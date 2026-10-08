#!/usr/bin/env bash
set -e

echo "=== DSA-Rohwerte debuggen ==="
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

def procrustes_align(X, Y):
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    U, _, Vh = torch.linalg.svd(Xc.T @ Yc)
    Q = U @ Vh
    return Xc @ Q + Y.mean(dim=0, keepdim=True)

@torch.no_grad()
def dsa_debug(model_a, model_b, task, seq_len=8, device="cpu"):
    A = collect_states(model_a, task, seq_len, device=device)
    B = collect_states(model_b, task, seq_len, device=device)
    za = A[:, :-1, :].reshape(-1, A.shape[-1])
    za_next = A[:, 1:, :].reshape(-1, A.shape[-1])
    zb = B[:, :-1, :].reshape(-1, B.shape[-1])
    zb_next = B[:, 1:, :].reshape(-1, B.shape[-1])
    zb_aligned = procrustes_align(zb, za)
    zb_next_aligned = procrustes_align(zb_next, za_next)
    H = za.shape[1]
    lambda_reg = 1e-3
    M_a = za.T @ za + lambda_reg * torch.eye(H, device=device)
    W_a = torch.linalg.solve(M_a, za.T @ za_next)
    err_self = ((za @ W_a - za_next) ** 2).mean().item()
    err_cross = ((zb_aligned @ W_a - zb_next_aligned) ** 2).mean().item()
    raw_score = 1.0 - (err_cross / (err_self + 1e-12))
    return {"err_self": err_self, "err_cross": err_cross, "raw_score": raw_score}
EOF

echo
echo "=== Test: DSA-Rohwerte ==="
cat > experiments/dsa_debug.py <<'EOF'
import torch
from cgfn.model import ContinuousRNN
from cgfn.baselines import GRUBaseline, LSTMBaseline, VanillaRNNBaseline
from cgfn.tasks import TASKS, make_batch
from cgfn.dsa import dsa_debug

device = "cpu"
task_dim = len(TASKS)

def train(ModelClass, seed, task, steps=2000, lr=1e-3, hidden_dim=32):
    torch.manual_seed(seed)
    m = ModelClass(task_dim=task_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    for _ in range(steps):
        x, y = make_batch(task, seq_len=8, device=device)
        tv = torch.zeros(x.shape[0], task_dim, device=device); tv[:, idx] = 1.0
        out, _ = m(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

for name, MC in [("CT-RNN", ContinuousRNN), ("GRU", GRUBaseline), ("LSTM", LSTMBaseline), ("Vanilla", VanillaRNNBaseline)]:
    print(f"\n--- {name} ---")
    m1 = train(MC, 0, "copy")
    m2 = train(MC, 1, "copy")
    d = dsa_debug(m1, m2, "copy")
    print(f"  cross-seed copy: err_self={d['err_self']:.4f} err_cross={d['err_cross']:.4f} raw={d['raw_score']:.4f}")
    m3 = train(MC, 0, "parity")
    d = dsa_debug(m1, m3, "copy")
    print(f"  copy vs parity: err_self={d['err_self']:.4f} err_cross={d['err_cross']:.4f} raw={d['raw_score']:.4f}")
EOF

python experiments/dsa_debug.py 2>&1 | tee dsa_debug.txt
echo
echo "=== Fertig: dsa_debug.txt ==="
