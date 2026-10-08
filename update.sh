#!/usr/bin/env bash
set -e

echo "=== Neue Datei: src/cgfn/lyap_train.py (Lyapunov-regularisiertes Training) ==="
cat > src/cgfn/lyap_train.py <<'EOF'
"""
Training mit FTLE-Regularisierung.
Statt tau manuell zu tunen, wird das Modell gezwungen,
am kritischen Punkt zu bleiben (ftle_mean ~ 0).
"""
import torch
import torch.nn as nn
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def ftle_penalty(model, z, target=0.0):
    """
    Penalty: (ftle_mean - target)^2.
    z: (B, H) letzter Zustand.
    """
    H = model.hidden_dim
    sech2 = 1.0 - torch.tanh(z) ** 2
    J = -torch.eye(H, device=z.device) / model.tau
    J = J.unsqueeze(0) + model.W.unsqueeze(0) * sech2.unsqueeze(1)
    sv = torch.linalg.svdvals(J)
    log_sv = torch.log(sv + 1e-12)
    ftle_mean = log_sv.mean()
    return (ftle_mean - target) ** 2


def train_lyap(seed, task, lambda_ftle=0.1, target_ftle=0.0,
               steps=1500, lr=1e-3, hidden_dim=32, tau=1.0,
               task_dim=None, seq_len=8, device="cpu"):
    """
    Trainiert mit FTLE-Regularisierung.
    lambda_ftle=0: normales Training.
    lambda_ftle>0: zieht ftle_mean gegen target_ftle.
    """
    if task_dim is None:
        task_dim = len(TASKS)
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim, tau=tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)

    history = []
    for step in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = torch.zeros(x.shape[0], task_dim, device=device)
        tv[:, idx] = 1.0
        out, states = model(x, tv)
        task_loss = lossfn(out, y)
        if lambda_ftle > 0:
            z_last = states[:, -1]
            ftle_loss = ftle_penalty(model, z_last, target_ftle)
            loss = task_loss + lambda_ftle * ftle_loss
        else:
            loss = task_loss
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 250 == 0:
            with torch.no_grad():
                z_last = states[:, -1]
                H = model.hidden_dim
                sech2 = 1.0 - torch.tanh(z_last) ** 2
                J = -torch.eye(H, device=device) / model.tau
                J = J.unsqueeze(0) + model.W.unsqueeze(0) * sech2.unsqueeze(1)
                sv = torch.linalg.svdvals(J)
                ftle_now = torch.log(sv + 1e-12).mean().item()
            history.append((step, task_loss.item(), ftle_now))

    return model, task_loss.item(), history
EOF

echo
echo "=== Experiment: Lyapunov-regularisiertes Training ==="
cat > experiments/lyap_training.py <<'EOF'
import torch
from cgfn.lyap_train import train_lyap
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.ftle import ftle_summary

device = "cpu"
task_dim = len(TASKS)
HIDDEN = 32
SEQ_LEN = 8


def one_hot(idx, bs):
    v = torch.zeros(bs, task_dim, device=device)
    v[:, idx] = 1.0
    return v


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

    print("=" * 74)
    print("LYAPUNOV-TRAINING: Findet das Modell den Sweet Spot selbst?")
    print(f"hidden_dim={HIDDEN}  tau=1.0 (fest!)")
    print("=" * 74)

    lambdas = [0.0, 0.01, 0.05, 0.1, 0.5]
    print(f"\n{'lambda':>8s} {'copy_loss':>10s} {'ftle':>8s} "
          f"{'CKA(c,p)':>10s} {'CKA(c,m)':>10s}")

    for lam in lambdas:
        # Copy trainieren
        mc, lc, hist_c = train_lyap(0, "copy", lambda_ftle=lam, tau=1.0,
                                     hidden_dim=HIDDEN, steps=1500)
        # Parity trainieren
        mp, lp, hist_p = train_lyap(0, "parity", lambda_ftle=lam, tau=1.0,
                                     hidden_dim=HIDDEN, steps=1500)
        # Mod3 trainieren
        mm, lm, hist_m = train_lyap(0, "mod3", lambda_ftle=lam, tau=1.0,
                                     hidden_dim=HIDDEN, steps=1500)

        # FTLE für copy
        s_c = reps(mc, "copy", x_fixed).unsqueeze(0)
        ftle = ftle_summary(mc, s_c)

        # CKA
        r_c = reps(mc, "copy", x_fixed)
        r_p = reps(mp, "parity", x_fixed)
        r_m = reps(mm, "mod3", x_fixed)
        cp = cka(r_c, r_p, kernel="linear")
        cm = cka(r_c, r_m, kernel="linear")

        print(f"{lam:8.2f} {lc:10.4f} {ftle['ftle_mean']:+8.4f} "
              f"{cp:10.4f} {cm:10.4f}")

    print("\n--- Trainingsverlauf (copy, lambda=0.1) ---")
    _, _, hist = train_lyap(0, "copy", lambda_ftle=0.1, tau=1.0,
                             hidden_dim=HIDDEN, steps=1500)
    print(f"{'step':>6s} {'task_loss':>12s} {'ftle':>10s}")
    for step, loss, ftle in hist:
        print(f"{step:6d} {loss:12.4f} {ftle:+10.4f}")

    print("\n=== FERTIG ===")
    print("Interpretation:")
    print("  - Wenn lambda>0 den Sweet Spot automatisch findet,")
    print("    sinkt CKA bei gleichbleibendem Loss.")
    print("  - Wenn nicht, ist FTLE-Regularisierung nicht ausreichend.")


if __name__ == "__main__":
    main()
EOF

echo
echo "=== Syntax-Check ==="
python -c "import ast; ast.parse(open('src/cgfn/lyap_train.py').read()); print('lyap_train OK')"
python -c "import ast; ast.parse(open('experiments/lyap_training.py').read()); print('experiment OK')"

echo
echo "=== Lauf (20-40 min) ==="
python experiments/lyap_training.py 2>&1 | tee lyap_training_results.txt
echo
echo "=== Fertig: lyap_training_results.txt ==="
