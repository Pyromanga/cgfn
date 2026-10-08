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
