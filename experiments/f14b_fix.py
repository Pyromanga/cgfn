"""
F14b — Fix it, richtig gemessen: Benettin-Regler im Regelkreis.

F14-Befund: Der SVD-basierte Pinning-Regler scheiterte KATASTROPHAL —
der Messartefakt (SVD-Kurzfenster +2..+3 bei kappa>>1) liess den
Regler endlos "korrigieren", tau rannte auf den Clamp, die Netze wurden
Benettin-SUPERkritikal (+0.37), MC kollabierte. F14b testet dieselbe
Rezept-Idee mit dem ARTEFAKTFREIEN Messgeraet im Regelkreis:
BenettinRegulator (src/cgfn/regulator.py) misst getrieben per
Standard-Benettin-QR und regelt tau auf target=-0.2 (subkritisch nahe
der Kante, F12-Lektion).

Arme (identisches Design wie F14 fuer Vergleichbarkeit):
  free    : plain Adam (Kontrolle, identisch zu F14)
  kcap    : kappa-Cap (identisch zu F14)
  pin     : SVD-Regler (F14-Fail-Arm — MITLAUFEND als direkter A/B-Beweis)
  pinben  : Benettin-Regler — DIE F14b-Frage: rettet artefaktfreie Messung
            das Kanten-Pinning?
  bothben : kcap + pinben (volles Rezept)

Fragen:
  (1) Konvergiert pinben zu einem sinnvollen tau (nicht Clamp)?
  (2) Liegt der Benettin-FTLE am Ende nahe target (-0.2)?
  (3) Schaedigt Kanten-Pinning mit richtiger Messung MC/Retention
      — oder gewinnt es sogar gegen free/kcap?
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.model import ContinuousRNN
from cgfn.tasks import make_batch
from cgfn.ftle import ftle_summary
from cgfn.regulator import FtleRegulator, BenettinRegulator
from cgfn.kcap import kappa_of, kappa_cap_penalty
from cgfn.memory_capacity import memory_capacity
from cgfn.benettin import benettin_ftle_max, stability_audit

ARMS = ("free", "kcap", "pin", "pinben", "bothben")


def evaluate(model, task, seq_len, batches=8):
    lossfn = torch.nn.BCEWithLogitsLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(task, batch_size=32, seq_len=seq_len, device="cpu")
            out, _ = model(x, None)
            total += lossfn(out, y).item()
    return total / batches


def train_phase(model, task, steps, lr, seq_len, arm, kappa_target,
                kappa_weight, regulator, track, eval_every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for step in range(steps):
        x, y = make_batch(task, batch_size=32, seq_len=seq_len, device="cpu")
        out, states = model(x, None)
        loss = lossfn(out, y)
        if arm in ("kcap", "bothben"):
            loss = loss + kappa_weight * kappa_cap_penalty(model, kappa_target)
        opt.zero_grad(); loss.backward(); opt.step()
        if regulator is not None:
            regulator.step(step, model, states)
        if (step + 1) % eval_every == 0:
            track.append(kappa_of(model))


def run_arm(arm, seed, args):
    torch.manual_seed(seed)
    model = ContinuousRNN(input_dim=1, task_dim=0,
                          hidden_dim=args.hidden_dim, tau=args.tau_init)
    regulator = None
    if arm == "pin":
        regulator = FtleRegulator(target=args.pin_target, gain=args.gain,
                                   interval=args.interval)
    elif arm in ("pinben", "bothben"):
        regulator = BenettinRegulator(target=args.pin_target, gain=args.gain,
                                      interval=args.interval)
    track = [kappa_of(model)]

    train_phase(model, args.task_a, args.steps, args.lr, args.seq_len,
                arm, args.kappa_target, args.kappa_weight, regulator, track)
    loss_a_after = evaluate(model, args.task_a, args.seq_len)
    train_phase(model, args.task_b, args.steps, args.lr, args.seq_len,
                arm, args.kappa_target, args.kappa_weight, regulator, track)
    loss_a_retention = evaluate(model, args.task_a, args.seq_len)
    loss_b_final = evaluate(model, args.task_b, args.seq_len)

    mc, _ = memory_capacity(model, max_lag=args.max_lag, input_scale=0.5)
    x = torch.randn(4, 48, 1) * 0.5
    _, states = model(x, None)
    ftle_svd = float(ftle_summary(model, states)["ftle_max"])
    ftle_ben = benettin_ftle_max(model, x)
    audit = stability_audit(model)
    return {
        "arm": arm, "seed": seed,
        "loss_a": loss_a_after, "loss_b": loss_b_final,
        "retent_a": loss_a_retention,
        "mc": mc, "ben": ftle_ben, "svd": ftle_svd,
        "kappa0": track[0], "kappa1": track[-1],
        "tau": float(model.tau), "rk2ok": audit["rk2_ok"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--task-a", default="copy")
    p.add_argument("--task-b", default="parity")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--tau-init", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=16)
    p.add_argument("--kappa-target", type=float, default=4.0)
    p.add_argument("--kappa-weight", type=float, default=0.02)
    p.add_argument("--pin-target", type=float, default=-0.2)
    p.add_argument("--gain", type=float, default=0.8)
    p.add_argument("--interval", type=int, default=100)
    p.add_argument("--out", default="results/f14b_fix_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 120)
    lines.append("F14b FIX-IT RICHTIG GEMESSEN: Benettin-Regler im Regelkreis vs. SVD-Regler (pin, F14-Fail) — direkter A/B-Beweis")
    lines.append(f"arms={list(ARMS)} seeds={seeds} steps={args.steps} lr={args.lr} tau_init={args.tau_init} H={args.hidden_dim}")
    lines.append(f"kappa_target={args.kappa_target} kappa_weight={args.kappa_weight} pin_target={args.pin_target} (subkritisch, F12-Lektion)")
    lines.append("Messung am Ende: MC (Jaeger), Benettin getrieben, SVD (Illusions-Kontrolle), kappa, tau, RK2-Audit")
    lines.append("=" * 120)
    lines.append(f"{'arm':>8s} {'seed':>4s} {'lossA':>7s} {'lossB':>7s} {'retentA':>8s} "
                 f"{'MC':>6s} {'ben':>8s} {'svd':>7s} {'kap1':>6s} {'tau':>5s} {'rk2ok':>5s}")

    results = []
    for seed in seeds:
        for arm in ARMS:
            r = run_arm(arm, seed, args)
            results.append(r)
            line = (f"{r['arm']:>8s} {r['seed']:4d} {r['loss_a']:7.4f} {r['loss_b']:7.4f} "
                    f"{r['retent_a']:8.4f} {r['mc']:6.2f} {r['ben']:+8.3f} {r['svd']:+7.3f} "
                    f"{r['kappa1']:6.2f} {r['tau']:5.2f} {str(r['rk2ok']):>5s}")
            lines.append(line)
            print(line, flush=True)

    lines.append("")
    lines.append("--- Aggregat (Mittel pro Arm) ---")
    lines.append(f"{'arm':>8s} {'lossA':>8s} {'lossB':>8s} {'retentA':>8s} "
                 f"{'MC':>6s} {'ben':>8s} {'kap1':>6s} {'tau':>5s}")
    agg = {}
    for arm in ARMS:
        rs = [r for r in results if r["arm"] == arm]
        def mean(key):
            vals = [r[key] for r in rs]
            return sum(vals) / len(vals) if vals else float("nan")
        agg[arm] = {k: mean(k) for k in ("loss_a", "loss_b", "retent_a", "mc", "ben", "kappa1", "tau")}
        a = agg[arm]
        lines.append(f"{arm:>8s} {a['loss_a']:8.4f} {a['loss_b']:8.4f} {a['retent_a']:8.4f} "
                     f"{a['mc']:6.2f} {a['ben']:+8.3f} {a['kappa1']:6.2f} {a['tau']:5.2f}")

    # --- (1) Konvergenz: pinben vs pin ---
    lines.append("")
    lines.append("--- (1) Regler-Konvergenz: pin (SVD-Artefakt) vs. pinben (Benettin) ---")
    lines.append(f"pin:    tau={agg['pin']['tau']:.2f} (F14: Clamp 0.05), Benettin-Ende={agg['pin']['ben']:+.3f}")
    lines.append(f"pinben: tau={agg['pinben']['tau']:.2f}, Benettin-Ende={agg['pinben']['ben']:+.3f} (Ziel {args.pin_target:+.1f})")
    hit = abs(agg["pinben"]["ben"] - args.pin_target)
    lines.append(f"pinben Ziel-Abweichung: {hit:+.3f} — {'KANTE GETROFFEN' if hit < 0.15 else 'nicht konvergiert (gain/interval tunen)'}")

    # --- (2/3) Wer gewinnt? ---
    lines.append("")
    lines.append("--- (2/3) Rezept-Arme vs. free ---")
    for arm in ARMS:
        if arm == "free":
            continue
        dl = agg[arm]["loss_a"] - agg["free"]["loss_a"]
        dm = agg[arm]["mc"] - agg["free"]["mc"]
        dr = agg[arm]["retent_a"] - agg["free"]["retent_a"]
        lines.append(f"{arm:>8s} vs free: dlossA={dl:+.4f}  dMC={dm:+.2f}  dretentA={dr:+.2f}")
    lines.append("")
    lines.append("Lesart: pinben konvergiert (tau im sinnvollen Bereich, Benettin ~ -0.2) und")
    lines.append("schlaegt pin auf ALLEN Metriken => der F14-Fail war der Artefakt, nicht die")
    lines.append("Rezept-Idee. Gewinnt pinben/bothben gegen free/kcap (MC, Retention) => ist")
    lines.append("Kanten-Pinning mit richtigem Messgeraet das vollstaendige Bau-Rezept.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
