"""
F7 — Spektrale Kritikalitaets-Regel: Der "Spectral Ruler" (nach F6-Befund).

Aus F6 (results/f6_spectrum_results.txt):
  - Spektrum-Korrelation 0.88-1.00: die Trajektorien-Lyapunov-Struktur ist
    fast vollstaendig durch die STATISCHE Jacobi J0 = -I/tau + W bestimmt.
  - Sompolinsky-Kriterium 76.7% Trefferquote, ABER einseitig exakt:
    KEIN Fall von "Chaos vorhergesagt, keins gemessen". Alle 21 Misses
    sind konservativ (pred=kein Chaos, gemessen Chaos), alle mit kleinem Q.

F7 klärt die Ursache und testet DREI Regeln gegeneinander:
  (a) Mean-Field:      tau * g * sqrt(Q) > 1
  (b) Eigenwert:       tau * Re lambda_max(W) > 1   (exakte lineare Stabilität)
  (c) Singulärwert:    tau * [log sigma_max(W) + log sqrt(Q)] > 1
                       (nicht-normale spektrale Kante)
gegen das gemessene SVD-FTLE_max — untrainiert und kurz trainiert.

Zusaetzlich: Nicht-Normalitaet kappa = sigma_max/|lambda_max| —
Hypothese: die Mean-Field-Misses korrelieren mit kappa (transientes
Wachstum nicht-normaler Matrizen, das Mean-Field nicht sieht).

 Wenn (b) die Misses eliminieren kann, gilt:
**Kritikalitaet ist aus der Gewichtsmatrix ALLEIN berechenbar — ohne
Rollout.** tau* haette dann die geschlossene Form  tau* ~ 1/Re lambda_max(W).
Das waere der "Spectral Ruler": Hyperparameter-Kalibrierung per Formel.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.cavity import chaos_criterion, nonnormality
from cgfn.ftle import ftle_summary
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def make_model(seed, tau, g_scale, hidden_dim=64):
    """g_scale multipliziert die W-Initialisierung (Basis-std 0.3)."""
    torch.manual_seed(seed)
    m = ContinuousRNN(task_dim=len(TASKS), hidden_dim=hidden_dim, tau=tau)
    with torch.no_grad():
        m.W.copy_(m.W * g_scale)
    return m


def train_copy(model, steps=300, lr=1e-3, seq_len=8):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index("copy")
    for _ in range(steps):
        x, y = make_batch("copy", seq_len=seq_len)
        tv = torch.zeros(x.shape[0], len(TASKS))
        tv[:, idx] = 1.0
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def probe(model, seq_len=16, batch=16):
    idx = TASKS.index("copy")
    x, _ = make_batch("copy", batch_size=batch, seq_len=seq_len)
    tv = torch.zeros(x.shape[0], len(TASKS))
    tv[:, idx] = 1.0
    _, states = model(x, tv)
    ftle = ftle_summary(model, states)
    crit = chaos_criterion(model, states)
    kappa, sv, lam = nonnormality(model)
    return {
        "ftle_max": ftle["ftle_max"],
        "Q": crit["Q"],
        "g": crit["g"],
        "mf": crit["tau_g_sqrtQ"],
        "eig": crit["eig_max"],
        "sv": crit["sv_max"],
        "pred_mf": crit["pred_mf"],
        "pred_eig": crit["pred_eig"],
        "pred_sv": crit["pred_sv"],
        "kappa": kappa,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--g-scales", default="0.5,1.0,1.5,2.0,3.0")
    p.add_argument("--taus", default="0.5,1.0,2.0")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--out", default="results/f7_spectral_results.txt")
    args = p.parse_args()

    g_scales = [float(v) for v in args.g_scales.split(",")]
    taus = [float(v) for v in args.taus.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 104)
    lines.append("F7 SPECTRAL RULER: Drei Chaos-Regeln aus W vs. gemessenes SVD-FTLE_max")
    lines.append("(a) Mean-Field tau*g*sqrt(Q)>1   (b) Eigenwert tau*Re lam_max(W)>1   (c) Singulärwert tau*(log sv_max + log sqrt Q)>1")
    lines.append(f"g_scales={g_scales}  taus={taus}  seeds={seeds}  hidden={args.hidden_dim}  train_steps={args.train_steps}")
    lines.append("=" * 104)
    lines.append(f"{'g_scale':>8s} {'tau':>5s} {'mode':>9s} {'ftle_max':>9s} {'Q':>6s} "
                 f"{'mf':>7s} {'lam_max':>8s} {'sv_max':>7s} {'kappa':>6s} "
                 f"{'pMF':>4s} {'pEig':>5s} {'pSV':>4s} {'meas':>5s}")

    stats = {"mf": [0, 0], "eig": [0, 0], "sv": [0, 0]}  # [hits, total]
    misses = []  # (kappa, mf, ftle) für Miss-Analyse

    for g_scale in g_scales:
        for tau in taus:
            for mode in ("untrained", "trained"):
                acc: dict = {}
                for seed in seeds:
                    m = make_model(seed, tau, g_scale, args.hidden_dim)
                    if mode == "trained":
                        train_copy(m, steps=args.train_steps)
                    r = probe(m)
                    for k in ("ftle_max", "Q", "g", "mf", "eig", "sv", "kappa"):
                        acc.setdefault(k, []).append(r[k])
                    meas = r["ftle_max"] > 0.0
                    for rule, pred in (("mf", r["pred_mf"]), ("eig", r["pred_eig"]), ("sv", r["pred_sv"])):
                        stats[rule][1] += 1
                        if pred == meas:
                            stats[rule][0] += 1
                    if not r["pred_mf"] and meas:
                        misses.append((r["kappa"], r["mf"], r["ftle_max"]))
                fm = sum(acc["ftle_max"]) / len(acc["ftle_max"])
                qm = sum(acc["Q"]) / len(acc["Q"])
                mfm = sum(acc["mf"]) / len(acc["mf"])
                em = sum(acc["eig"]) / len(acc["eig"])
                sm = sum(acc["sv"]) / len(acc["sv"])
                km = sum(acc["kappa"]) / len(acc["kappa"])
                meas = fm > 0.0
                pmf = mfm > 1.0
                peig = args.hidden_dim and (em * tau > 1.0)
                psv = (tau * (math_log(sm) + 0.5 * math_log(max(qm, 1e-9))) ) > 1.0
                line = (f"{g_scale:8.2f} {tau:5.2f} {mode:>9s} {fm:+9.4f} {qm:6.3f} "
                        f"{mfm:7.3f} {em:8.3f} {sm:7.3f} {km:6.2f} "
                        f"{str(pmf):>4s} {str(peig):>5s} {str(psv):>4s} {str(meas):>5s}")
                lines.append(line)
                print(line, flush=True)

    lines.append("")
    lines.append("--- Trefferquoten (gemittelt ueber seeds; Vorzeichen-Abgleich) ---")
    for rule, (h, t) in stats.items():
        lines.append(f"Regel {rule:>4s}: {h}/{t} = {h / max(1, t):.1%}")
    lines.append("")
    if misses:
        kappas = [k for k, _, _ in misses]
        lines.append(f"Mean-Field-Misses: {len(misses)} (alle konservativ, s. F6). "
                     f"kappa-Range der Misses: {min(kappas):.2f}..{max(kappas):.2f}")
    lines.append("")
    lines.append("LESEART:")
    lines.append("- pEig bei ~100% => lineare Stabilitaet von W erklaert Chaos vollstaendig;")
    lines.append("  Kritikalitaet ist aus W allein berechenbar => 'Spectral Ruler':")
    lines.append("  tau* ~ 1 / Re lambda_max(W)  — Hyperparameter ohne Rollout.")
    lines.append("- pEig > pMF und kappa korreliert mit den MF-Misses => die F6-Luecke")
    lines.append("  ist Nicht-Normalitaet (Bulk vs. spektraler Rand), kein Widerspruch")
    lines.append("  zur Theorie, sondern eine praegnantere Regel darunter.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


def math_log(x):
    import math
    return math.log(max(x, 1e-12))


if __name__ == "__main__":
    main()
