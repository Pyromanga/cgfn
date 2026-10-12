"""
F6 — Theorie-Anschluss: Chaos-Kriterium & Lyapunov-Spektrum (ROADMAP).

Zwei Tests gegen die Mean-Field-Theorie (Sompolinsky/Van Vreeswijk,
Cavity-Methode, s. Clark arXiv:2610.12426):

(1) Chaos-Kriterium  tau * g * sqrt(Q) > 1
    g-Sweep (W-Skalierung) x tau-Sweep am UNTRAINIERTEN und kurz
    TRAINIERTEN Netz: Vorhersage (Kriterium) gegen Vorzeichen des
    gemessenen FTLE_max. Trefferquote über alle Configs.

(2) Volles Spektrum: linearisierte Jacobi J0 = -I/tau + W (Fixpunkt)
    gegen Trajektorien-FTLE-Spektrum — Korrelation der sortierten
    Spektren je Config.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.cavity import chaos_criterion, compare_spectrum, effective_gain, q_factor
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
def probe(model, seq_len=16, batch=16, task="copy"):
    idx = TASKS.index(task)
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len)
    tv = torch.zeros(x.shape[0], len(TASKS))
    tv[:, idx] = 1.0
    _, states = model(x, tv)
    ftle = ftle_summary(model, states)
    crit = chaos_criterion(model, states)
    spec = compare_spectrum(model, states)
    return {
        "g": crit["g"],
        "Q": crit["Q"],
        "criterion": crit["tau_g_sqrtQ"],
        "pred": crit["chaos_predicted"],
        "ftle_max": ftle["ftle_max"],
        "spec_corr": spec["spectrum_corr"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--g-scales", default="0.5,1.0,1.5,2.0,3.0")
    p.add_argument("--taus", default="0.5,1.0,2.0")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--out", default="results/f6_spectrum_results.txt")
    args = p.parse_args()

    g_scales = [float(v) for v in args.g_scales.split(",")]
    taus = [float(v) for v in args.taus.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 92)
    lines.append("F6 THEORIE: Sompolinsky-Kriterium tau*g*sqrt(Q)>1 vs. gemessenes FTLE_max;")
    lines.append("Spektrum: J0 = -I/tau + W vs. Trajektorien-FTLE (sortierte Korrelation)")
    lines.append(f"g_scales={g_scales}  taus={taus}  seeds={seeds}  hidden={args.hidden_dim}  "
                 f"train_steps(copy)={args.train_steps}")
    lines.append("=" * 92)
    lines.append(f"{'g_scale':>8s} {'tau':>5s} {'mode':>9s} {'g':>6s} {'Q':>6s} "
                 f"{'tau*g*sqrtQ':>12s} {'pred':>5s} {'ftle_max':>10s} {'meas':>5s} "
                 f"{'hit':>4s} {'spec_corr':>9s}")

    hits, total = 0, 0
    for g_scale in g_scales:
        for tau in taus:
            for mode in ("untrained", "trained"):
                g_acc, q_acc, crit_acc, ftle_acc, corr_acc = [], [], [], [], []
                for seed in seeds:
                    m = make_model(seed, tau, g_scale, args.hidden_dim)
                    if mode == "trained":
                        train_copy(m, steps=args.train_steps)
                    r = probe(m)
                    g_acc.append(r["g"]); q_acc.append(r["Q"])
                    crit_acc.append(r["criterion"]); ftle_acc.append(r["ftle_max"])
                    corr_acc.append(r["spec_corr"])
                    total += 1
                    if r["pred"] == (r["ftle_max"] > 0.0):
                        hits += 1
                gm = sum(g_acc) / len(g_acc)
                qm = sum(q_acc) / len(q_acc)
                cm = sum(crit_acc) / len(crit_acc)
                fm = sum(ftle_acc) / len(ftle_acc)
                rm = sum(corr_acc) / len(corr_acc)
                pred = cm > 1.0
                meas = fm > 0.0
                hit = "Y" if pred == meas else "N"
                line = (f"{g_scale:8.2f} {tau:5.2f} {mode:>9s} {gm:6.3f} {qm:6.3f} "
                        f"{cm:12.3f} {str(pred):>5s} {fm:+10.4f} {str(meas):>5s} "
                        f"{hit:>4s} {rm:9.3f}")
                lines.append(line)
                print(line, flush=True)

    acc_rate = hits / max(1, total)
    lines.append("")
    lines.append(f"Trefferquote Sompolinsky-Kriterium vs. FTLE-Vorzeichen: "
                 f"{hits}/{total} = {acc_rate:.1%}")
    lines.append("Leseart: >=80% => Mean-Field-Kriterium ist am CT-RNN mit -z/tau-Decay "
                 "tragfaehig (nach tau-Reskalierung); spec_corr nahe +1 => die "
                 "linearisierte Jacobi erklaert das Trajektorien-Spektrum.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
