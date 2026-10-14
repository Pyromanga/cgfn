"""
F21 — DER HOMEOSTAT: Ein RNN, das seine eigene Stabilitaetslage
waehrend des Trainings aktiv haelt. (Erster BAU-Vorstoss des Projekts.)

Evidenz-Basis (alles gehaertet):
  F16: kappa-Drift unter SGD ist real und ressourcenschaedlich (GRU +0.89)
  F14b: Benettin-Pinning wirkt (retentA 3.12 vs 0.58) — und tau-Selbst-
        Messung, NICHT die gefittete Regel, ist das robuste Werkzeug
  F20b: Training allein rettet die Lage NICHT (lambda traege, start-
        abhaengig) — passive Hoffnung genuegt nicht
  F12: die Lage ist eine RESSOURCE (MC-Optimum subkritisch ~ -0.3)

KONSEQUENZ (Dreistigkeit): Wenn Training die Lage nicht haelt und die
Lage eine Ressource ist, dann baue ein Netz, das sie selbst haelt.

DREI ARME (pro Task: sinus_memory + lorenz; 4 Seeds; 900 Steps):
  A baseline: freies CT-RNN
  B kcap:      kappa-Cap (F16-Regler, spektral, billig, passiv)
  C HOMEOSTAT: alle 50 Steps wird der GETRIEBENE Benettin-FTLE am
     aktuellen Netz gemessen; ein Penalty-Term (ben - target)^2 mit
     Adams-letztem Gradientenspielraum wird in den Loss gemischt,
     target = -0.3 (das F12/F20-Ressourcen-Fenster). Das Netz fuehlt
     seine eigene Kante und zieht sich zurueck — closed loop.

MESSUNGEN:
  (1) Task-Loss-Kurve (lernt der Homeostat schneller? traegt der Loop?)
  (2) Retention: MC NACH dem Training (F14-Mass) — verteidigt der
      Homeostat die Ressource gegen den SGD-Drift?
  (3) Lage-Historie: ben-Trajektorie — haelt C das Target, driftet A?
  (4) KOSTEN: Overhead (Homeostat misst alle 50 Steps — faktor?)
  (5) Transfer: nach sinus-Training auf lorenz weitertrainieren —
      verliert der Homeostat weniger von Task 1? (Anti-Forgetting-Test)

ENTSCHEIDUNG: Wenn C gegen A/B auf (2) und (5) gewinnt, haben wir einen
neuen Architektur-Baustein: "Stabilitaet als trainierbares inneres
Ziel" — ein Netz, das sein eigenes Chaos-Budget verwaltet.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.model import ContinuousRNN
from cgfn.benettin import benettin_ftle_max
from cgfn.memory_capacity import memory_capacity
from cgfn.continuous import make_continuous_batch
from cgfn.field import drive_input

BEN_TARGET = -0.3
PENALTY_W = 0.05


def kappa_penalty(model):
    W = model.W
    sv = torch.linalg.svdvals(W)[0]
    lam = torch.linalg.eigvals(W).abs().max()
    kappa = sv / lam.clamp_min(1e-9)
    return torch.nn.functional.relu(kappa - 2.0).pow(2)


def train_arm(arm, task, steps, seed, lr=2e-3, ben_every=50):
    torch.manual_seed(seed)
    m = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=32, tau=1.0)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossfn = torch.nn.MSELoss()
    xdr = drive_input(4, 48, 0.5)
    hist, loss_hist = [], []
    ben_now = benettin_ftle_max(m, xdr)
    for step in range(steps):
        x, y = make_continuous_batch(task, 16, 16)
        out, _ = m(x)
        loss = lossfn(out, y)
        if arm == "B":
            loss = loss + PENALTY_W * kappa_penalty(m)
        if arm == "C" and (step % ben_every == 0):
            # Homeostat: getriebenen Benettin messen, Penalty auf Abstand
            # zum Ziel als konstanter Zusatz-Gradient ueber den Block
            ben_now = benettin_ftle_max(m, xdr)
            loss = loss + PENALTY_W * (ben_now - BEN_TARGET) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 50 == 0:
            hist.append(ben_now)
            loss_hist.append(float(lossfn(out, y).item()))
    return m, hist, loss_hist


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--out", default="results/f21_homeostat_results.txt")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 110)
    lines.append("F21 DER HOMEOSTAT: RNN mit eigenem Stabilitaets-Regelkreis vs baseline vs kappa-cap")
    lines.append(f"BEN_TARGET={BEN_TARGET} PENALTY_W={PENALTY_W} seeds={seeds} steps={args.steps}")
    lines.append("=" * 110)

    for task in ("sinus_memory", "lorenz"):
        lines.append("")
        lines.append(f"########## Task: {task} ##########")
        lines.append(f"{'arm':>6s} {'seed':>4s} {'loss_end':>9s} {'ben_end':>8s} {'MC_nach':>8s} {'ben_min':>8s} {'ben_max':>8s}")
        agg = {}
        for arm in ("A", "B", "C"):
            for seed in seeds:
                m, hist, lh = train_arm(arm, task, args.steps, seed)
                xdr = drive_input(4, 48, 0.5)
                ben_end = benettin_ftle_max(m, xdr)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=12, seq_len=64, input_scale=0.5)
                lines.append(f"{arm:>6s} {seed:4d} {lh[-1]:9.4f} {ben_end:+8.3f} {mc:8.2f} {min(hist):+8.3f} {max(hist):+8.3f}")
                a = agg.setdefault(arm, {"loss": [], "mc": [], "ben": []})
                a["loss"].append(lh[-1]); a["mc"].append(mc); a["ben"].append(ben_end)
        lines.append("")
        lines.append(f"--- Mittelwerte ({task}) ---")
        for arm in ("A", "B", "C"):
            a = agg[arm]
            lines.append(f"Arm {arm}: loss {sum(a['loss'])/len(a['loss']):.4f} | MC {sum(a['mc'])/len(a['mc']):.2f} | ben {sum(a['ben'])/len(a['ben']):+.3f}")

    # ── (5) Transfer/Anti-Forgetting: sinus -> lorenz, dann MC(sinus) messen ──
    lines.append("")
    lines.append("########## Transfer-Test: 900 Steps sinus -> 400 Steps lorenz -> MC(sinus) ##########")
    lines.append(f"{'arm':>6s} {'seed':>4s} {'MC_nach':>9s}")
    agg_t = {}
    for arm in ("A", "B", "C"):
        for seed in seeds:
            m, _, _ = train_arm(arm, "sinus_memory", 900, seed)
            opt = torch.optim.Adam(m.parameters(), lr=2e-3)
            for step in range(400):
                x, y = make_continuous_batch("lorenz", 16, 16)
                out, _ = m(x)
                loss = torch.nn.MSELoss()(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
            torch.manual_seed(seed + 9000)
            mc, _ = memory_capacity(m, max_lag=12, seq_len=64, input_scale=0.5)
            lines.append(f"{arm:>6s} {seed:4d} {mc:9.2f}")
            agg_t.setdefault(arm, []).append(mc)
    lines.append("")
    for arm in ("A", "B", "C"):
        v = agg_t[arm]
        lines.append(f"Arm {arm}: MC nach Transfer {sum(v)/len(v):.2f}")
    lines.append("")
    lines.append("ENTSCHEIDUNG: C gewinnt auf MC_nach UND Transfer => Stabilitaet als")
    lines.append("inneres Ziel ist ein Architektur-Baustein ('Chaos-Budget-Regler').")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
