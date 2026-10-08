#!/usr/bin/env bash
set -e

echo "================================================================"
echo "  FRONTIER EXPERIMENTS: CKA, DSA, INSD, Architektur-Vergleich"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────
# 1. DSA als Alternative zu CKA
# ─────────────────────────────────────────────────────────────────
echo
echo "[1/4] Schreibe src/cgfn/dsa.py (Dynamical Similarity Analysis)"
cat > src/cgfn/dsa.py <<'EOF'
"""
Dynamical Similarity Analysis (DSA) nach Ostrow et al. (2024).
Misst, ob zwei RNNs dieselbe *Dynamik* nutzen, nicht nur dieselbe
Repraesentationsgeometrie. DSA ist robuster gegen Rauschen und
identifiziert behaviorale Konvergenz zuverlaessiger als CKA.
"""
import torch


@torch.no_grad()
def collect_dynamics(model, task, seq_len=8, batch=32, device="cpu"):
    """Sammelt (state, next_state) Paare ueber eine Trajektorie."""
    from cgfn.tasks import make_batch
    from cgfn.model import ContinuousRNN
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    if hasattr(model, 'task_dim') and model.task_dim > 0:
        from cgfn.tasks import TASKS
        idx = TASKS.index(task)
        tv = torch.zeros(batch, model.task_dim, device=device)
        tv[:, idx] = 1.0
        _, states = model(x, tv)
    else:
        _, states = model(x)
    z = states[:, :-1, :].reshape(-1, states.shape[-1])
    z_next = states[:, 1:, :].reshape(-1, states.shape[-1])
    return z, z_next


def dsa(model_a, model_b, task, seq_len=8, device="cpu"):
    """
    Vergleicht zwei Modelle ueber die lineare Vorhersagbarkeit
    der Dynamik. Aendert sich die Dynamik, sinkt DSA.
    """
    za, za_next = collect_dynamics(model_a, task, seq_len, device=device)
    zb, zb_next = collect_dynamics(model_b, task, seq_len, device=device)

    # Ridge-Regression: z_next ~ z
    lambda_reg = 1e-3
    Ha = za.shape[1]

    # Modell A: Vorhersage von za_next aus za
    A_a = (za.T @ za + lambda_reg * torch.eye(Ha, device=device))
    W_a = torch.linalg.solve(A_a, za.T @ za_next)
    pred_a = za @ W_a

    # Cross-Prediction: sagt W_a die Dynamik von B voraus?
    pred_b_from_a = zb @ W_a
    err_cross = ((pred_b_from_a - zb_next) ** 2).mean()

    # Baseline: sagt W_b die eigene Dynamik voraus?
    A_b = (zb.T @ zb + lambda_reg * torch.eye(Ha, device=device))
    W_b = torch.linalg.solve(A_b, zb.T @ zb_next)
    pred_b = zb @ W_b
    err_self = ((pred_b - zb_next) ** 2).mean()

    # DSA: 1 - (cross_error / self_error), geclippt auf [0, 1]
    score = 1.0 - (err_cross / (err_self + 1e-12)).item()
    return max(0.0, min(1.0, score))
EOF

# ─────────────────────────────────────────────────────────────────
# 2. INSD als Simplicity-Bias-Brecher
# ─────────────────────────────────────────────────────────────────
echo
echo "[2/4] Schreibe src/cgfn/insd.py (Iterative Neural Similarity Deflation)"
cat > src/cgfn/insd.py <<'EOF'
"""
Iterative Neural Similarity Deflation (INSD) nach Qian & Pehlevan (ICLR 2026).
Penalisiert die lineare Vorhersagbarkeit neuer RNNs relativ zu bestehenden
Loesungen. Zwingt das Netz, alternative dynamische Loesungen zu finden,
statt immer wieder auf denselben Attraktor zu kollabieren.
"""
import torch
import torch.nn as nn
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch

device = "cpu"


def one_hot(task_idx, batch_size, task_dim):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train_with_insd(seed, task, reference_states=None, steps=4000,
                    lr=1e-3, seq_len=8, hidden_dim=64,
                    task_dim=len(TASKS), lambda_insd=0.5):
    """
    Trainiert ein Modell mit INSD-Regularisierung.
    reference_states: Liste von (B*T, H) Tensoren frueherer Loesungen.
    Der INSD-Term bestraft lineare Vorhersagbarkeit aus den Referenzen.
    """
    torch.manual_seed(seed)
    task_idx = TASKS.index(task)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()

    if reference_states is None:
        reference_states = []

    # Orthogonalisierungsbasis der Referenzzustaende
    if len(reference_states) > 0:
        R = torch.cat(reference_states, dim=0)  # (n_ref, H)
        # QR-Zerlegung fuer orthonormale Basis
        Q, _ = torch.linalg.qr(R.T)             # (H, k)
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
            # Lineare Vorhersagbarkeit aus Q
            proj = Z @ Q @ Q.T
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
    from cgfn.tasks import TASKS as T
    task_idx = T.index(task)
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    tv = one_hot(task_idx, batch, len(T))
    _, states = model(x, tv)
    B, T_len, H = states.shape
    return states.reshape(B * T_len, H)
EOF

# ─────────────────────────────────────────────────────────────────
# 3. Architektur-Vergleich: CT-RNN vs. GRU vs. LSTM
# ─────────────────────────────────────────────────────────────────
echo
echo "[3/4] Schreibe src/cgfn/baselines.py (GRU, LSTM, Vanilla RNN)"
cat > src/cgfn/baselines.py <<'EOF'
"""
Standard-RNN-Baselines mit gleicher Parameterzahl wie ContinuousRNN.
Entscheidet, ob die ODE-Struktur ueberhaupt eine Rolle spielt.
"""
import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32):
        super().__init__()
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim + task_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_seq, task_vec=None):
        B, T, _ = x_seq.shape
        if task_vec is None and self.task_dim > 0:
            task_vec = torch.zeros(B, self.task_dim, device=x_seq.device)
        elif task_vec is not None and self.task_dim > 0:
            task_vec = task_vec.unsqueeze(1).expand(-1, T, -1)
            x_seq = torch.cat([x_seq, task_vec], dim=-1)
        out, _ = self.gru(x_seq)
        return self.readout(out), out.unsqueeze(0)


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32):
        super().__init__()
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim + task_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_seq, task_vec=None):
        B, T, _ = x_seq.shape
        if task_vec is None and self.task_dim > 0:
            task_vec = torch.zeros(B, self.task_dim, device=x_seq.device)
        elif task_vec is not None and self.task_dim > 0:
            task_vec = task_vec.unsqueeze(1).expand(-1, T, -1)
            x_seq = torch.cat([x_seq, task_vec], dim=-1)
        out, _ = self.lstm(x_seq)
        return self.readout(out), out.unsqueeze(0)


class VanillaRNNBaseline(nn.Module):
    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32):
        super().__init__()
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim + task_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_seq, task_vec=None):
        B, T, _ = x_seq.shape
        if task_vec is None and self.task_dim > 0:
            task_vec = torch.zeros(B, self.task_dim, device=x_seq.device)
        elif task_vec is not None and self.task_dim > 0:
            task_vec = task_vec.unsqueeze(1).expand(-1, T, -1)
            x_seq = torch.cat([x_seq, task_vec], dim=-1)
        out, _ = self.rnn(x_seq)
        return self.readout(out), out.unsqueeze(0)
EOF

# ─────────────────────────────────────────────────────────────────
# 4. Haupt-Experiment
# ─────────────────────────────────────────────────────────────────
echo
echo "[4/4] Schreibe experiments/frontier.py"
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


def one_hot(task_idx, batch_size):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train_model(ModelClass, seed, task, steps=4000, lr=1e-3):
    torch.manual_seed(seed)
    model = ModelClass(task_dim=task_dim)
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

    # ─── Experiment A: Architektur-Vergleich ───
    print("=" * 60)
    print("A) Architektur-Vergleich: CT-RNN vs GRU vs LSTM vs Vanilla")
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

        # CKA zwischen Seeds auf copy
        reps = [collect_reps(m, "copy", x_fixed) for m in models]
        if len(reps) >= 2:
            c = cka(reps[0], reps[1], kernel="linear")
            print(f"  CKA cross-seed copy: {c:.4f}")

        # DSA zwischen Seeds auf copy
        if len(models) >= 2:
            d = dsa(models[0], models[1], "copy")
            print(f"  DSA cross-seed copy: {d:.4f}")

        # CKA copy vs parity (gleicher Seed)
        m = models[0]
        reps_c = collect_reps(m, "copy", x_fixed)
        reps_p = collect_reps(m, "parity", x_fixed)
        c = cka(reps_c, reps_p, kernel="linear")
        print(f"  CKA copy vs parity (same seed): {c:.4f}")

        # DSA copy vs parity (gleicher Seed, self-DSA = 1.0 trivial)
        # Stattdessen: Task-Sensitivitaet
        m2, _ = train_model(ModelClass, 0, "parity")
        d_cross = dsa(models[0], m2, "copy")
        print(f"  DSA copy-model vs parity-model: {d_cross:.4f}")

    # ─── Experiment B: INSD ───
    print("\n" + "=" * 60)
    print("B) INSD: Simplicity Bias brechen")
    print("=" * 60)

    print("\n--- Baseline (kein INSD) ---")
    for task in ["parity", "mod3"]:
        m, loss = train_with_insd(0, task, reference_states=None)
        print(f"  {task}: loss={loss:.4f}")

    print("\n--- INSD mit copy als Referenz ---")
    copy_model, _ = train_model(ContinuousRNN, 0, "copy")
    ref_states = [collect_states_for_ref(copy_model, "copy")]

    for task in ["parity", "mod3"]:
        m, loss = train_with_insd(0, task, reference_states=ref_states,
                                  lambda_insd=0.5)
        print(f"  {task}: loss={loss:.4f} (INSD)")

    print("\n--- INSD mit copy+reverse als Referenz ---")
    rev_model, _ = train_model(ContinuousRNN, 0, "reverse")
    ref_states = [
        collect_states_for_ref(copy_model, "copy"),
        collect_states_for_ref(rev_model, "reverse"),
    ]
    for task in ["parity", "mod3"]:
        m, loss = train_with_insd(0, task, reference_states=ref_states,
                                  lambda_insd=0.5)
        print(f"  {task}: loss={loss:.4f} (INSD 2 refs)")

    # ─── Experiment C: DSA vs CKA Direktvergleich ───
    print("\n" + "=" * 60)
    print("C) DSA vs CKA: Welches Mass trennt besser?")
    print("=" * 60)
    print("   Erwartung: DSA trennt Aufgaben schaerfer als CKA,")
    print("   weil es die Dynamik misst, nicht die Geometrie.")
    print()

    models = {}
    for task in ["copy", "parity", "reverse"]:
        models[task], _ = train_model(ContinuousRNN, 0, task)

    print(f"{'Paar':30s} {'CKA':>8s} {'DSA':>8s}")
    for a, b in combinations(["copy", "parity", "reverse"], 2):
        reps_a = collect_reps(models[a], a, x_fixed)
        reps_b = collect_reps(models[b], b, x_fixed)
        c = cka(reps_a, reps_b, kernel="linear")
        d = dsa(models[a], models[b], a)
        print(f"  {a:10s} vs {b:10s}      {c:8.4f} {d:8.4f}")

    print("\n=== FERTIG ===")


if __name__ == "__main__":
    main()
EOF

# ─────────────────────────────────────────────────────────────────
# Ausfuehren
# ─────────────────────────────────────────────────────────────────
echo
echo "=== Syntax-Check ==="
for f in src/cgfn/dsa.py src/cgfn/insd.py src/cgfn/baselines.py experiments/frontier.py; do
    python -c "import ast; ast.parse(open('$f').read())" 2>&1 \
      && echo "OK: $f" \
      || echo "FAIL: $f"
done

echo
echo "=== Lokaler Test ==="
python -m pytest -q 2>&1 | tail -3

echo
echo "=== Hauptexperiment (dauert 10-20 Minuten) ==="
python experiments/frontier.py 2>&1 | tee frontier_results.txt

echo
echo "=== Fertig. Ergebnisse in frontier_results.txt ==="
