import torch
from itertools import combinations
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.effdim import effective_dimension, effective_dimension_ratio
from cgfn.inputdsa import estimate_dynamics, inputdsa

device = "cpu"
task_dim = len(TASKS)
SEQ_LEN = 8
HIDDEN = 32
STEPS = 2000
SEEDS = [0, 1, 2]
TASK_SUBSET = ["copy", "reverse", "parity", "mod3"]


def one_hot(task_idx, batch_size):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train(seed, task, tau=1.0, steps=STEPS, lr=1e-3):
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=HIDDEN, tau=tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    for _ in range(steps):
        x, y = make_batch(task, seq_len=SEQ_LEN, device=device)
        tv = one_hot(idx, x.shape[0])
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()


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

    print("=" * 70)
    print("FRONTIER 2: Effective Dimension + InputDSA + Tau-Sweep")
    print(f"seeds={SEEDS}  tasks={TASK_SUBSET}  hidden_dim={HIDDEN}")
    print("=" * 70)

    # A) Effective Dimension
    print("\n--- A) Effective Dimension (hoeher = reichere Geometrie) ---")
    print(f"{'task':10s} {'seed':>4s} {'loss':>8s} {'d_eff':>7s} {'ratio':>7s}")
    all_d_eff = {t: [] for t in TASK_SUBSET}
    for task in TASK_SUBSET:
        for seed in SEEDS:
            m, loss = train(seed, task)
            r = reps(m, task, x_fixed)
            d = effective_dimension(r)
            ratio = effective_dimension_ratio(r)
            all_d_eff[task].append(d)
            print(f"{task:10s} {seed:4d} {loss:8.4f} {d:7.2f} {ratio:7.3f}")

    print("\n  Mittelwerte:")
    for task in TASK_SUBSET:
        vals = all_d_eff[task]
        mean = sum(vals) / len(vals)
        print(f"    {task:10s}: d_eff={mean:.2f}")

    # B) InputDSA
    print("\n--- B) InputDSA: Normen der intrinsischen und input-getriebenen Dynamik ---")
    print(f"{'task':10s} {'seed':>4s} {'loss':>8s} {'||A||':>8s} {'||B||':>8s}")
    for task in TASK_SUBSET:
        for seed in SEEDS:
            m, loss = train(seed, task)
            try:
                A, B = estimate_dynamics(m, task, device=device)
                d_int = A.norm().item()
                d_inp = B.norm().item()
                print(f"{task:10s} {seed:4d} {loss:8.4f} {d_int:8.4f} {d_inp:8.4f}")
            except Exception as e:
                print(f"{task:10s} {seed:4d} {loss:8.4f} ERROR: {e}")

    # C) Tau-Sweep
    print("\n--- C) Tau-Sweep: Effekt der Rueckstellkraft auf Kollaps ---")
    taus = [0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"{'tau':>6s} {'copy_loss':>10s} {'parity_loss':>12s} {'mod3_loss':>10s} {'CKA(copy,parity)':>18s}")
    for tau in taus:
        losses = {}
        reps_dict = {}
        for task in ["copy", "parity", "mod3"]:
            m, loss = train(0, task, tau=tau, steps=1500)
            losses[task] = loss
            reps_dict[task] = reps(m, task, x_fixed)
        cka_cp = cka(reps_dict["copy"], reps_dict["parity"], kernel="linear")
        print(f"{tau:6.1f} {losses['copy']:10.4f} {losses['parity']:12.4f} "
              f"{losses['mod3']:10.4f} {cka_cp:18.4f}")

    print("\n=== FERTIG ===")


if __name__ == "__main__":
    main()
