"""
F10 — Sättigungs-korrigierter Spectral Ruler: das eine Gesetz?

F9-Fund (kappa=1, s=0): das FTLE-Crossing liegt NICHT bei 1/lam_max,
sondern deutlich darüber (1.25->1.84, 0.87->2.54). Verdacht: Der reine
W-Ruler nimmt die Ursprungs-Näherung d=sech^2(0)=1 an, aber das Netz ist
input-getrieben — entlang der Trajektorie gilt J(z) = -I/tau + W diag(sech^2(z)),
d.h. der effektive Gain schrumpft auf d = <sech^2(z)> = 1 - Q.

F10-These (das candidates Gesetz):
    Chaos  <=>  tau * d * Re lambda_max(W) > 1,   d = <sech^2(z)>
    Sweet-Spot:  tau* = 1 / (lam_max * d(tau*))   — self-consistent.

Zwei Prüfteile:
  (A) Designed Networks (kappa=1, Schur-Bau): empirisches FTLE-Crossing
      vs. 1/lam (unkorrigiert) vs. self-consistent 1/(lam*d). Wenn die
      korrigierte Vorhersage das Crossing trifft, ist das Gesetz geschlossen.
  (B) F8-Replikat (trainierte Netze, Checkpoints): Regel-Güte
      MF / Eig / EigQbar / SV vs. FTLE-Vorzeichen. Frage: schließt die
      Korrektur die 5 hit_eig-Misses aus F8 — und wie verhält sie sich
      gegen die 100%-SVD-Regel?
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.cavity import chaos_criterion, eig_max
from cgfn.continuous import make_continuous_batch
from cgfn.design import make_designed_model
from cgfn.ftle import ftle_summary
from cgfn.ruler import spectral_report


def rollout_states(model, batch=16, seq_len=24):
    x, _ = make_continuous_batch("sinus_memory", batch, seq_len)
    _, states = model(x)
    return states


def measure_ftle(lam, s, tau, hidden_dim, seed, batch=16, seq_len=24):
    model = make_designed_model(hidden_dim, tau, lam, s, seed, task_dim=0)
    states = rollout_states(model, batch, seq_len)
    return ftle_summary(model, states)["ftle_max"]


def tau_crossing(lam, s, taus, hidden_dim, seed):
    """FTLE ueber tau-Grid; interpoliertes Crossing (Vorzeichenwechsel)."""
    pts = [(tau, measure_ftle(lam, s, tau, hidden_dim, seed)) for tau in taus]
    for (t1, f1), (t2, f2) in zip(pts, pts[1:]):
        if (f1 <= 0.0 <= f2) or (f2 <= 0.0 <= f1):
            if f2 == f1:
                return t1, pts
            frac = (0.0 - f1) / (f2 - f1)
            return t1 + frac * (t2 - t1), pts
    return None, pts


def self_consistent_tau(lam, s, hidden_dim, seed, iters=4, tau_min=0.2, tau_max=6.0):
    """
    F10-Kern: tau* = 1/(lam * d(tau*)) iterativ.
    Start bei der Ursprungs-Vorhersage 1/lam; pro Iteration EIN Rollout
    (kein Training, kein FTLE), d messen, tau nachziehen. Konvergenz in
    wenigen Schritten erwartet (d aendert sich langsam mit tau).
    """
    tau_it = min(max(1.0 / lam, tau_min), tau_max)
    trace = [tau_it]
    for _ in range(iters):
        model = make_designed_model(hidden_dim, tau_it, lam, s, seed, task_dim=0)
        states = rollout_states(model)
        crit = chaos_criterion(model, states)
        d = crit["Qbar"]
        tau_new = 1.0 / max(lam * max(d, 1e-9), 1e-9)
        tau_new = min(max(tau_new, tau_min), tau_max)
        trace.append(tau_new)
        if abs(tau_new - tau_it) < 0.01:
            tau_it = tau_new
            break
        tau_it = tau_new
    return tau_it, trace


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lams", default="0.8,1.15,1.6")
    p.add_argument("--taus", default="0.5,0.7,0.9,1.1,1.4,1.7,2.0,2.4,2.8,3.3")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--b-steps", type=int, default=600)
    p.add_argument("--b-ckpt", type=int, default=150)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--out", default="results/f10_qbar_results.txt")
    args = p.parse_args()

    lams = [float(v) for v in args.lams.split(",")]
    grid = [float(v) for v in args.taus.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 108)
    lines.append("F10 SAETTIGUNGS-KORRIGIERTER RULER: tau* = 1/(lam_max * d), d = <sech^2(z)> = 1-Q")
    lines.append(f"lams={lams}  taus={grid}  seeds={seeds}  iters={args.iters}  hidden={args.hidden_dim}")
    lines.append("=" * 108)

    # ── Teil A: Designed Networks (kappa=1) ──────────────────────────────
    lines.append("")
    lines.append("--- (A) Designed W, s=0 (kappa=1): Crossing vs. Vorhersagen ---")
    lines.append(f"{'lam':>5s} {'tRaw':>6s} {'tSC':>6s} {'tCross':>7s} "
                 f"{'|dRaw|':>7s} {'|dSC|':>7s} {'d_final':>7s} {'winner':>7s} {'SC-Trace':>34s}")
    errs_raw, errs_sc = [], []
    wins = {"raw": 0, "sc": 0, "none": 0}
    for lam in lams:
        crosses = []
        for seed in seeds:
            cross, _ = tau_crossing(lam, 0.0, grid, args.hidden_dim, seed)
            if cross is not None:
                crosses.append(cross)
        tau_sc, trace = self_consistent_tau(lam, 0.0, args.hidden_dim, seeds[0],
                                            iters=args.iters)
        # final d am self-consistenten Punkt fuer die Tabelle
        model = make_designed_model(args.hidden_dim, tau_sc, lam, 0.0, seeds[0], task_dim=0)
        crit = chaos_criterion(model, rollout_states(model))
        d_final = crit["Qbar"]
        if not crosses:
            wins["none"] += 1
            lines.append(f"{lam:5.2f} {1.0/lam:6.3f} {tau_sc:6.3f} {'kein Wechsel':>12s}   d={d_final:.3f}")
            continue
        cross = sum(crosses) / len(crosses)
        d_raw, d_sc = abs(cross - 1.0 / lam), abs(cross - tau_sc)
        errs_raw.append(d_raw); errs_sc.append(d_sc)
        winner = "raw" if d_raw <= d_sc else "SC"
        wins[winner] += 1
        line = (f"{lam:5.2f} {1.0/lam:6.3f} {tau_sc:6.3f} {cross:7.3f} "
                f"{d_raw:7.3f} {d_sc:7.3f} {d_final:7.3f} {winner:>7s} "
                f"{' -> '.join(f'{t:.2f}' for t in trace):>34s}")
        lines.append(line)
        print(line, flush=True)
    if errs_raw:
        lines.append(f"mean |cross - 1/lam|   = {sum(errs_raw)/len(errs_raw):.3f}   (unkorrigiert)")
        lines.append(f"mean |cross - tau_SC| = {sum(errs_sc)/len(errs_sc):.3f}   (self-consistent)")
        lines.append(f"Sieger: raw={wins['raw']}, SC={wins['sc']}, kein Crossing={wins['none']}")

    # ── Teil B: F8-Replikat mit 4. Regel ─────────────────────────────────
    lines.append("")
    lines.append("--- (B) Trainierte Netze (F8-Design): Regel-Güte MF / Eig / EigQbar / SV ---")
    lines.append(f"tasks=copy+sinus_memory  g_scales=0.8,1.6  taus=1.0  seeds={seeds}  "
                 f"steps={args.b_steps}  ckpt={args.b_ckpt}")
    hits = {"mf": [0, 0], "eig": [0, 0], "eigqbar": [0, 0], "sv": [0, 0]}
    from cgfn.model import ContinuousRNN
    from cgfn.tasks import TASKS, make_batch

    def train_block(model, task, steps, lr=2e-3):
        opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=lr)
        if task == "copy":
            lossfn = torch.nn.BCEWithLogitsLoss()
            idx = TASKS.index("copy")
            for _ in range(steps):
                x, y = make_batch("copy", seq_len=8)
                tv = torch.zeros(x.shape[0], len(TASKS)); tv[:, idx] = 1.0
                out, _ = model(x, tv)
                loss = lossfn(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            lossfn = torch.nn.MSELoss()
            for _ in range(steps):
                x, y = make_continuous_batch("sinus_memory", 16, 16)
                out, _ = model(x)
                loss = lossfn(out, y)
                opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def rollout_task(model, task, batch=16, seq_len=16):
        if task == "copy":
            idx = TASKS.index("copy")
            x, _ = make_batch("copy", batch_size=batch, seq_len=seq_len)
            tv = torch.zeros(batch, len(TASKS)); tv[:, idx] = 1.0
            _, st = model(x, tv)
        else:
            x, _ = make_continuous_batch("sinus_memory", batch, seq_len)
            _, st = model(x)
        return st

    for task in ("copy", "sinus_memory"):
        for g_scale in (0.8, 1.6):
            for seed in seeds:
                torch.manual_seed(seed)
                task_dim = len(TASKS) if task == "copy" else 0
                model = ContinuousRNN(input_dim=1, task_dim=task_dim,
                                       hidden_dim=args.hidden_dim, tau=1.0)
                with torch.no_grad():
                    model.W.copy_(model.W * g_scale)
                for step in range(0, args.b_steps + 1, args.b_ckpt):
                    if step > 0:
                        train_block(model, task, args.b_ckpt)
                    states = rollout_task(model, task)
                    crit = chaos_criterion(model, states)
                    ftle = ftle_summary(model, states)["ftle_max"]
                    meas = ftle > 0.0
                    for rule, pred in (("mf", crit["pred_mf"]),
                                       ("eig", crit["pred_eig"]),
                                       ("eigqbar", crit["pred_eig_qbar"]),
                                       ("sv", crit["pred_sv"])):
                        hits[rule][1] += 1
                        if pred == meas:
                            hits[rule][0] += 1
    for rule, (h, t) in hits.items():
        lines.append(f"Regel {rule:>8s}: {h}/{t} = {h/max(1,t):.1%}")

    lines.append("")
    lines.append("LESEAT (A): |dSC| << |dRaw| => das Gesetz tau* = 1/(lam*d) ist")
    lines.append("geschlossen — der Ruler braucht genau EINEN Rollout, kein Training,")
    lines.append("keinen FTLE-Sweep. Der F9-Twist waere aufgeklaert (Sättigung, nicht")
    lines.append("Widerspruch). LESEAT (B): eigqbar nahe 100% => auch die trainierten")
    lines.append("Netze folgen dem korrigierten Gesetz; eigqbar vs. sv entscheidet,")
    lines.append("ob der Ruler mit d-Korrektur die SVD-Regel ersetzt (eine Regel statt")
    lines.append("dreier) — das waeere der Durchbruch.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
