#!/usr/bin/env bash
set -e

mkdir -p src/cgfn experiments tests .github/workflows
touch src/cgfn/__init__.py

cat > pyproject.toml <<'EOF'
[project]
name = "cgfn"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["torch>=2.2", "numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
EOF

cat > src/cgfn/model.py <<'EOF'
import torch
import torch.nn as nn


class ContinuousRNN(nn.Module):
    """dz/dt = -z/tau + W tanh(z) + U x(t), RK2-integriert."""

    def __init__(self, input_dim=1, hidden_dim=32, tau=1.0, dt=0.1, substeps=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau, self.dt, self.substeps = tau, dt, substeps
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.3)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def dynamics(self, z, x):
        return -z / self.tau + torch.tanh(z) @ self.W.T + self.U(x)

    def forward(self, x_seq, control=None):
        B, T, _ = x_seq.shape
        z = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        states = []
        for t in range(T):
            x = x_seq[:, t]
            for _ in range(self.substeps):
                k1 = self.dynamics(z, x)
                k2 = self.dynamics(z + 0.5 * self.dt * k1, x)
                z = z + self.dt * k2
                if control is not None:
                    z = z + self.dt * control(z, t)
            states.append(z)
        states = torch.stack(states, dim=1)
        return self.readout(states), states
EOF

cat > src/cgfn/criticality.py <<'EOF'
import torch


@torch.no_grad()
def jacobian_spectral_radius(model, z):
    W = model.W
    sech2 = 1.0 - torch.tanh(z) ** 2
    J = -torch.eye(model.hidden_dim, device=z.device) / model.tau
    J = J + W.unsqueeze(0) * sech2.unsqueeze(1)
    return torch.linalg.eigvals(J).abs().max(dim=-1).values


@torch.no_grad()
def branching_ratio(states):
    d = states[:, 1:] - states[:, :-1]
    num = d.var(dim=(1, 2))
    den = states[:, :-1].var(dim=(1, 2)) + 1e-8
    return num / den
EOF

cat > src/cgfn/geometry.py <<'EOF'
import torch


def mean_state_signature(states):
    """states: (B, T, H) -> (H,) gemittelt ueber Batch und letzte 25% der Zeit."""
    T = states.shape[1]
    tail = states[:, int(0.75 * T):, :]
    return tail.mean(dim=(0, 1))


def covariance_signature(states):
    """states: (B, T, H) -> (H, H) Kovarianz ueber Batch*Zeit."""
    B, T, H = states.shape
    X = states.reshape(B * T, H)
    X = X - X.mean(dim=0, keepdim=True)
    return (X.T @ X) / (X.shape[0] - 1)


def signature_distance(sig_a, sig_b):
    return (sig_a - sig_b).norm().item()


def steering_loss(states, target_sig):
    sig = mean_state_signature(states)
    return ((sig - target_sig) ** 2).mean()
EOF

cat > src/cgfn/tasks.py <<'EOF'
import torch

TASKS = ["copy", "reverse", "parity"]


def make_batch(task, batch_size=32, seq_len=16, device="cpu"):
    x = torch.randint(0, 2, (batch_size, seq_len), device=device).float()
    if task == "copy":
        y = x
    elif task == "reverse":
        y = x.flip(1)
    elif task == "parity":
        y = torch.cumsum(x, dim=1) % 2
    else:
        raise ValueError(task)
    return x.unsqueeze(-1), y.unsqueeze(-1)
EOF

cat > experiments/run_smoke.py <<'EOF'
import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.criticality import jacobian_spectral_radius, branching_ratio
from cgfn.geometry import mean_state_signature, signature_distance

torch.manual_seed(0)
device = "cpu"


def train_on(task, steps=2000, lr=1e-3, seq_len=8, hidden_dim=64):
    model = ContinuousRNN(hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        out, _ = model(x)
        loss = lossfn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


def main():
    print(f"{'task':10s} {'loss':>8s} {'spec_rad':>10s} {'branch':>8s}")
    signatures = {}
    for task in TASKS:
        model, loss = train_on(task)
        x, _ = make_batch(task, batch_size=64, seq_len=8, device=device)
        with torch.no_grad():
            _, states = model(x)
        sr = jacobian_spectral_radius(model, states[:, -1]).mean().item()
        br = branching_ratio(states).mean().item()
        signatures[task] = mean_state_signature(states)
        print(f"{task:10s} {loss:8.4f} {sr:10.4f} {br:8.4f}")

    print("\nSignatur-Distanzen (Mittelwert im Feature-Raum):")
    keys = list(signatures)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = signature_distance(signatures[keys[i]], signatures[keys[j]])
            print(f"  {keys[i]:8s} vs {keys[j]:8s}: {d:.4f}")

    print("\nSignatur-Normen:")
    for k, v in signatures.items():
        print(f"  {k:8s}: {v.norm().item():.4f}")


if __name__ == "__main__":
    main()
EOF

cat > tests/test_smoke.py <<'EOF'
import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import make_batch
from cgfn.criticality import jacobian_spectral_radius


def test_forward_shape():
    m = ContinuousRNN()
    x, _ = make_batch("copy", batch_size=4, seq_len=8)
    out, states = m(x)
    assert out.shape == (4, 8, 1)
    assert states.shape == (4, 8, 32)


def test_spectral_radius_finite():
    m = ContinuousRNN()
    x, _ = make_batch("copy", batch_size=4, seq_len=8)
    with torch.no_grad():
        _, s = m(x)
    sr = jacobian_spectral_radius(m, s[:, -1])
    assert torch.isfinite(sr).all()
EOF

cat > .github/workflows/smoke.yml <<'EOF'
name: cgfn-smoke

on:
  push:
  workflow_dispatch:

jobs:
  smoke:
    runs-on: ubuntu-latest
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
        run: python experiments/run_smoke.py
EOF

echo "Alle Dateien neu geschrieben."

echo
echo "=== Syntax-Check ==="
for f in src/cgfn/*.py tests/*.py experiments/*.py; do
  python -c "import ast; ast.parse(open('$f').read())" 2>&1 \
    && echo "OK: $f" \
    || echo "FAIL: $f"
done

echo
echo "=== Installation ==="
pip install -e ".[dev]" >/dev/null

echo
echo "=== Pytest ==="
python -m pytest -q

echo
echo "=== Smoke-Experiment ==="
python experiments/run_smoke.py 2>&1 | tee smoke2.txt

echo
echo "Fertig. Ausgabe auch in smoke2.txt."
