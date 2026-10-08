import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.ftle import ftle_summary
from cgfn.adversarial import adversarial_sensitivity

device = "cpu"
task_dim = len(TASKS)
HIDDEN = 32
SEQ_LEN = 8


def one_hot(task_idx, bs):
    v = torch.zeros(bs, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train(seed, task, tau, steps=1500, lr=1e-3):
    torch.manual_seed(seed)
    m = ContinuousRNN(task_dim=task_dim, hidden_dim=HIDDEN, tau=tau)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    for _ in range(steps):
        x, y = make_batch(task, seq_len=SEQ_LEN, device=device)
        tv = one_hot(idx, x.shape[0])
        out, _ = m(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m, loss.item()


def reps(m, task, fixed_x):
    idx = TASKS.index(task)
    tv = one_hot(idx, fixed_x.shape[0])
    with torch.no_grad():
        _, s = m(fixed_x, tv)
    if s.dim() == 4:
        s = s.squeeze(0)
    B, T, H = s.shape
    return s.reshape(B * T, H)


def main():
    torch.manual_seed(999)
    x_fixed = torch.randint(0, 2, (32, SEQ_LEN, 1), device=device).float()
    taus = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    eps_values = [0.01, 0.05, 0.1, 0.2]

    print("=" * 74)
    print("PFAD 2: Tau vs. adversarial Robustheit")
    print(f"hidden_dim={HIDDEN}  taus={taus}")
    print("=" * 74)

    print(f"\n{'tau':>6s} {'copy_loss':>10s} {'ftle':>8s} "
          f"{'CKA(c,p)':>10s} {'adv_sens(0.1)':>15s}")

    for tau in taus:
        # Copy + Parity trainieren
        mc, lc = train(0, "copy", tau)
        mp, lp = train(0, "parity", tau)

        # FTLE für copy
        s_c = reps(mc, "copy", x_fixed).unsqueeze(0)
        ftle = ftle_summary(mc, s_c)

        # CKA(copy, parity)
        r_c = reps(mc, "copy", x_fixed)
        r_p = reps(mp, "parity", x_fixed)
        c = cka(r_c, r_p, kernel="linear")

        # Adversariale Sensitivität für copy
        adv = adversarial_sensitivity(mc, "copy", task_dim, eps=0.1, device=device)

        print(f"{tau:6.2f} {lc:10.4f} {ftle['ftle_mean']:+8.4f} "
              f"{c:10.4f} {adv:15.4f}")

    print("\n=== FERTIG ===")
    print("Interpretation:")
    print("  - Wenn adv_sens am Sweet Spot (tau~2) minimal ist,")
    print("    dann korreliert Repräsentationstrennung mit Robustheit.")
    print("  - Wenn nicht, ist Repräsentationstrennung unabhaengig von Robustheit.")


if __name__ == "__main__":
    main()
