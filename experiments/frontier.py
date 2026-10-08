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
