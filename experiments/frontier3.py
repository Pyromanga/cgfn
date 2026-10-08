import torch
from itertools import combinations
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.effdim import effective_dimension, effective_dimension_ratio
from cgfn.ftle import ftle_summary
from cgfn.phase import collapse_phase, criticality_distance

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


def get_states(model, task, fixed_x):
    idx = TASKS.index(task)
    tv = one_hot(idx, fixed_x.shape[0])
    with torch.no_grad():
        _, s = model(fixed_x, tv)
    if s.dim() == 4:
        s = s.squeeze(0)
    return s


def main():
    torch.manual_seed(999)
    x_fixed = torch.randint(0, 2, (32, SEQ_LEN, 1), device=device).float()

    print("=" * 70)
    print("FRONTIER 3: Phasenübergangs-Analyse des Repräsentationskollaps")
    print(f"seeds={SEEDS}  tasks={TASK_SUBSET}  hidden_dim={HIDDEN}")
    print("=" * 70)

    # Teil A: FTLE + d_eff + Kollaps-Typ
    print("\n--- A) FTLE-Spektrum, d_eff und Kollaps-Typ ---")
    print(f"{'task':10s} {'seed':>4s} {'loss':>8s} {'ftle_mean':>10s} "
          f"{'d_eff':>7s} {'phase':>22s}")
    for task in TASK_SUBSET:
        for seed in SEEDS:
            m, loss = train(seed, task)
            s = get_states(m, task, x_fixed)
            ftle = ftle_summary(m, s)
            d = effective_dimension(s.reshape(-1, s.shape[-1]))
            ratio = effective_dimension_ratio(s.reshape(-1, s.shape[-1]))
            phase = collapse_phase(ftle, ratio)
            print(f"{task:10s} {seed:4d} {loss:8.4f} {ftle['ftle_mean']:10.4f} "
                  f"{d:7.2f} {phase:>22s}")

    # Teil B: Tau-Sweep mit FTLE
    print("\n--- B) Tau-Sweep: FTLE und Kollaps-Typ ---")
    taus = [0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"{'tau':>6s} {'task':10s} {'loss':>8s} {'ftle_mean':>10s} "
          f"{'d_eff':>7s} {'phase':>22s}")
    for tau in taus:
        for task in ["copy", "parity", "mod3"]:
            m, loss = train(0, task, tau=tau, steps=1500)
            s = get_states(m, task, x_fixed)
            ftle = ftle_summary(m, s)
            d = effective_dimension(s.reshape(-1, s.shape[-1]))
            ratio = effective_dimension_ratio(s.reshape(-1, s.shape[-1]))
            phase = collapse_phase(ftle, ratio)
            print(f"{tau:6.1f} {task:10s} {loss:8.4f} {ftle['ftle_mean']:10.4f} "
                  f"{d:7.2f} {phase:>22s}")

    # Teil C: Korrelation FTLE <-> d_eff <-> Loss
    print("\n--- C) Korrelationen (Pearson) ---")
    ftles, ds, losses = [], [], []
    for task in TASK_SUBSET:
        for seed in SEEDS:
            m, loss = train(seed, task)
            s = get_states(m, task, x_fixed)
            ftle = ftle_summary(m, s)
            d = effective_dimension(s.reshape(-1, s.shape[-1]))
            ftles.append(ftle["ftle_mean"])
            ds.append(d)
            losses.append(loss)

    import statistics
    def pearson(x, y):
        n = len(x)
        mx, my = sum(x)/n, sum(y)/n
        cov = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y)) / n
        sx = (sum((xi-mx)**2 for xi in x)/n)**0.5
        sy = (sum((yi-my)**2 for yi in y)/n)**0.5
        return cov / (sx*sy + 1e-12)

    print(f"  corr(ftle, d_eff)  = {pearson(ftles, ds):+.4f}")
    print(f"  corr(ftle, loss)   = {pearson(ftles, losses):+.4f}")
    print(f"  corr(d_eff, loss)  = {pearson(ds, losses):+.4f}")

    print("\n=== FERTIG ===")


if __name__ == "__main__":
    main()
