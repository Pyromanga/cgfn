"""
F2 — Kausaler Test: FTLE-Zwang waehrend des Trainings (ROADMAP F2).

Design:
  Arme:
    free     : tau fix (kein Regler) — Kontrollarm
    reg-T    : FtleRegler mit target T (default grid: 0.0, +0.3, -0.3)
  Pro Arm:
    1. Phase A: Task 'copy' trainieren → Learning-Speed (Steps bis loss<thresh),
       Endloss, FTLE-Ende, tau-Verlauf (bei reg).
    2. Phase B: Task 'parity' WEITERtrainieren (continual) → Loss auf 'copy'
       DANACH (Retention) + Loss auf 'parity'.
  Vergleichstabelle: Arm vs. SpeedA / LossA / LossParityB / RetentionA.

Kausale Lesart: Wenn Edge-of-Chaos URSACHE (nicht nur Korrelat) guten Lernens
ist, muss reg-0.0 gegen free mit schlechtem Start-tau gewinnen (Sicherheit:
free startet absichtlich suboptimal, default tau_init=4.0 → weit ueber Sweet
Spot). Und reg mit target weit weg von 0 sollte schlechter sein als reg-0.0.

Selbst-contained wie scaling.py: nur model/tasks/ftle/regulator aus src/cgfn.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.ftle import ftle_summary
from cgfn.regulator import FtleRegulator


def evaluate(model, task, task_dim, seq_len, device, batches=8):
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(task, batch_size=32, seq_len=seq_len, device=device)
            tv = torch.zeros(x.shape[0], task_dim, device=device)
            tv[:, idx] = 1.0
            out, _ = model(x, tv)
            total += lossfn(out, y).item()
    return total / batches


def train_phase(model, task, steps, lr, seq_len, device, regulator=None,
                speed_threshold=0.15, eval_every=50):
    """Trainiert `steps` Schritte auf `task`; rueckt Dict mit Metriken."""
    task_dim = len(TASKS)
    idx = TASKS.index(task)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    steps_to_thresh = -1
    last_ftle = float("nan")
    for step in range(steps):
        x, y = make_batch(task, batch_size=32, seq_len=seq_len, device=device)
        tv = torch.zeros(x.shape[0], task_dim, device=device)
        tv[:, idx] = 1.0
        out, states = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if regulator is not None:
            regulator.step(step, model, states)
        if (step + 1) % eval_every == 0:
            with torch.no_grad():
                last_ftle = float(ftle_summary(model, states)["ftle_mean"])
            if steps_to_thresh < 0 and loss.item() < speed_threshold:
                steps_to_thresh = step + 1
    return {
        "final_train_loss": loss.item(),
        "steps_to_thresh": steps_to_thresh,
        "ftle_final": last_ftle,
        "tau_final": float(model.tau),
    }


def run_arm(name, seed, task_a, task_b, steps, lr, seq_len, device,
            tau_init, regulator_factory):
    task_dim = len(TASKS)
    torch.manual_seed(seed)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=32, tau=tau_init)
    reg = regulator_factory() if regulator_factory else None

    phase_a = train_phase(model, task_a, steps, lr, seq_len, device, reg)
    loss_a_after = evaluate(model, task_a, task_dim, seq_len, device)
    phase_b = train_phase(model, task_b, steps, lr, seq_len, device, reg)
    loss_a_retention = evaluate(model, task_a, task_dim, seq_len, device)
    loss_b_final = evaluate(model, task_b, task_dim, seq_len, device)

    return {
        "arm": name, "seed": seed,
        "speed_a": phase_a["steps_to_thresh"],
        "loss_a_end": phase_a["final_train_loss"],
        "loss_a_after": loss_a_after,
        "loss_b_end": loss_b_final,
        "loss_a_retention": loss_a_retention,
        "ftle_final": phase_b["ftle_final"],
        "tau_final": phase_b["tau_final"],
        "tau_init": tau_init,
        "reg_history": list(reg.history) if reg else [],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--task-a", default="copy")
    p.add_argument("--task-b", default="parity")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--tau-init", type=float, default=4.0,
                   help="absichtlich suboptimal: weit ueber dem Sweet Spot")
    p.add_argument("--targets", default="0.0,0.3,-0.3",
                   help="FTLE-Ziele der Regler-Arme; free immer dabei")
    p.add_argument("--gain", type=float, default=0.8)
    p.add_argument("--interval", type=int, default=100)
    p.add_argument("--out", default="results/f2_regulator_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    targets = [float(t) for t in args.targets.split(",")]
    device = "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 78)
    lines.append("F2 REGULATOR: Kausaltest — bringt FTLE~0-Zwang besseres Lernen?")
    lines.append(f"task_a={args.task_a}  task_b={args.task_b}  seeds={seeds}")
    lines.append(f"steps={args.steps}  lr={args.lr}  tau_init={args.tau_init} "
                 f"(suboptimal)  regler: interval={args.interval} gain={args.gain}")
    lines.append(f"targets={targets} (+ free-Kontrollarm)")
    lines.append("=" * 78)
    lines.append(f"{'arm':>10s} {'seed':>4s} {'speedA':>7s} {'lossA':>7s} "
                 f"{'lossB':>7s} {'retentA':>8s} {'ftle':>7s} {'tau':>6s}")

    results = []
    for seed in seeds:
        # Kontrollarm: tau fix
        r = run_arm("free", seed, args.task_a, args.task_b, args.steps,
                    args.lr, args.seq_len, device, args.tau_init, None)
        results.append(r)
        lines.append(f"{r['arm']:>10s} {r['seed']:4d} {r['speed_a']:7d} "
                      f"{r['loss_a_after']:7.4f} {r['loss_b_end']:7.4f} "
                      f"{r['loss_a_retention']:8.4f} {r['ftle_final']:+7.3f} "
                      f"{r['tau_final']:6.2f}")
        # Regler-Arme
        for tgt in targets:
            def factory(t=tgt):
                return FtleRegulator(target=t, gain=args.gain,
                                     interval=args.interval)
            name = f"reg{tgt:+.1f}"
            r = run_arm(name, seed, args.task_a, args.task_b, args.steps,
                        args.lr, args.seq_len, device, args.tau_init, factory)
            results.append(r)
            lines.append(f"{r['arm']:>10s} {r['seed']:4d} {r['speed_a']:7d} "
                          f"{r['loss_a_after']:7.4f} {r['loss_b_end']:7.4f} "
                          f"{r['loss_a_retention']:8.4f} {r['ftle_final']:+7.3f} "
                          f"{r['tau_final']:6.2f}")

    # Aggregat pro Arm (Mittel ueber Seeds)
    lines.append("")
    lines.append("--- Aggregat (Mittel pro Arm) ---")
    lines.append(f"{'arm':>10s} {'speedA':>8s} {'lossA':>8s} {'lossB':>8s} "
                  f"{'retentA':>8s} {'ftle':>7s}")
    arms = ["free"] + [f"reg{t:+.1f}" for t in targets]
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        def mean(key):
            vals = [r[key] for r in rs if isinstance(r[key], (int, float))]
            if not vals:
                return float("nan")
            good = [v for v in vals if v == v and v != -1]
            return sum(good) / len(good) if good else -1
        lines.append(f"{arm:>10s} {mean('speed_a'):8.1f} {mean('loss_a_after'):8.4f} "
                     f"{mean('loss_b_end'):8.4f} {mean('loss_a_retention'):8.4f} "
                     f"{mean('ftle_final'):+7.3f}")

    # Kausale Lesart
    lines.append("")
    lines.append("Lesart: Wenn Edge-of-Chaos URSACHE ist, gilt:")
    lines.append("  reg+0.0 schneller/kleiner LossA als free (Regler rettet den")
    lines.append("  suboptimalen Start-tau); retentA nicht signifikant schlechter;")
    lines.append("  reg-0.3 (zuvor kontraktiv) und reg+0.3 (chaotisch) schlechter")
    lines.append("  als reg+0.0. Sonst: Kritikalitaet nur Korrelat.")
    lines.append("")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
