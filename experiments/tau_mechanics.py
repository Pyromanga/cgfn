import torch
import math
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.cka import cka
from cgfn.mechanics import jacobian_spectral_radius_at_fixed_point, untrained_state_stats

device = "cpu"
task_dim = len(TASKS)
HIDDEN = 32
SEQ_LEN = 8


def one_hot(task_idx, batch_size):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train(seed, task, tau, steps=1500, lr=1e-3):
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

    # Log-spaced taus: fein um den vermuteten Sweet Spot herum
    taus = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    print("=" * 74)
    print("TAU-MECHANIK: Warum gibt es einen Sweet Spot?")
    print(f"hidden_dim={HIDDEN}  taus={len(taus)} Werte")
    print("=" * 74)

    # ─── Teil 1: Untrained Dynamics ───
    print("\n--- Teil 1: Untrained Dynamics (kein Training) ---")
    print(f"{'tau':>6s} {'|z|':>8s} {'z_std':>8s} {'ftle_max':>10s} "
          f"{'ftle_min':>10s} {'|rho(J)|':>10s}")
    for tau in taus:
        torch.manual_seed(42)  # gleiche Initialisierung
        m = ContinuousRNN(task_dim=0, hidden_dim=HIDDEN, tau=tau)
        stats = untrained_state_stats(m, input_scale=1.0, seq_len=SEQ_LEN,
                                      batch=32, device=device)
        rho = jacobian_spectral_radius_at_fixed_point(m.W, tau, z=None)
        print(f"{tau:6.2f} {stats['state_norm']:8.3f} {stats['state_std']:8.3f} "
              f"{stats['ftle_max']:10.4f} {stats['ftle_min']:10.4f} {rho:10.4f}")

    # ─── Teil 2: Trained Sweet Spot ───
    print("\n--- Teil 2: Trainiertes Modell, CKA(copy, parity) ---")
    print("   2 Seeds gemittelt. Niedrige CKA = Aufgaben getrennt.")
    print(f"{'tau':>6s} {'copy_loss':>10s} {'parity_loss':>12s} "
          f"{'mod3_loss':>10s} {'CKA(c,p)':>10s} {'CKA(c,m)':>10s}")

    seeds = [0, 1]
    results = []
    for tau in taus:
        c_losses, p_losses, m_losses = [], [], []
        cka_cp, cka_cm = [], []
        for seed in seeds:
            mc, lc = train(seed, "copy", tau)
            mp, lp = train(seed, "parity", tau)
            mm, lm = train(seed, "mod3", tau)
            rc = reps(mc, "copy", x_fixed)
            rp = reps(mp, "parity", x_fixed)
            rm = reps(mm, "mod3", x_fixed)
            c_losses.append(lc); p_losses.append(lp); m_losses.append(lm)
            cka_cp.append(cka(rc, rp, kernel="linear"))
            cka_cm.append(cka(rc, rm, kernel="linear"))

        c_m = sum(c_losses)/len(c_losses)
        p_m = sum(p_losses)/len(p_losses)
        m_m = sum(m_losses)/len(m_losses)
        cp_m = sum(cka_cp)/len(cka_cp)
        cm_m = sum(cka_cm)/len(cka_cm)
        results.append({"tau": tau, "cka_cp": cp_m, "cka_cm": cm_m,
                        "copy_loss": c_m, "parity_loss": p_m, "mod3_loss": m_m})
        print(f"{tau:6.2f} {c_m:10.4f} {p_m:12.4f} {m_m:10.4f} "
              f"{cp_m:10.4f} {cm_m:10.4f}")

    # ─── Teil 3: Sweet-Spot-Analyse ───
    print("\n--- Teil 3: Sweet-Spot-Analyse ---")
    # Definiere Sweet Spot als Minimum von CKA(copy, parity)
    best = min(results, key=lambda r: r["cka_cp"])
    print(f"  Sweet Spot (min CKA(copy,parity)): tau = {best['tau']:.2f}")
    print(f"    CKA(copy,parity) = {best['cka_cp']:.4f}")
    print(f"    CKA(copy,mod3)   = {best['cka_cm']:.4f}")
    print(f"    copy_loss        = {best['copy_loss']:.4f}")
    print(f"    parity_loss      = {best['parity_loss']:.4f}")

    # Qualitaet: Kann copy noch gelernt werden?
    print("\n  Trade-off: Wann wird copy schlecht?")
    for r in results:
        flag = " <-- copy bricht ein" if r["copy_loss"] > 0.05 else ""
        print(f"    tau={r['tau']:5.2f}: copy_loss={r['copy_loss']:.4f}{flag}")

    print("\n=== FERTIG ===")
    print("Interpretation:")
    print("  - Wenn CKA(copy,parity) ein Minimum hat, existiert ein Sweet Spot.")
    print("  - Wenn gleichzeitig copy_loss klein bleibt, ist der Sweet Spot nutzbar.")
    print("  - Der Vergleich mit Teil 1 zeigt, ob der Sweet Spot mit")
    print("    dem kritischen Punkt (|rho(J)| ~ 1) zusammenfaellt.")


if __name__ == "__main__":
    main()
