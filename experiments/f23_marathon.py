"""
F23 — PLASTIZITAETS-MARATHON: 10 Tasks hintereinander, ein Netz.
Der Test, der aus dem Homeostat einen Architektur-Baustein macht —
oder ihn begräbt. Katastrophales Vergessen ist DER Plastizitaets-Killer;
F16 zeigt: kappa-Drift frisst die Ressource, F14b: Regeln hilft.

SEQUENZ (10 Bloecke, 300 Steps je Block, dann Umschalten):
  sinus_memory -> lorenz -> ou_noise -> sinus_memory (Retest!) -> lorenz
  -> ou_noise -> sinus_memory -> lorenz -> ou_noise -> sinus_memory (Endtest)
  => sinus_memory erscheint 4x (Bloecke 1, 4, 7, 10): Misst die
  Retentions-Kurve ueber die Sequenz — verliert Arm C langsamer?

ARME: A frei / B kappa-cap / C1 Homeostat-Block (F21) — C2 kommt
aus F22 dazu, wenn F22 positiv entscheidet (hier: C1 als Repraesentant).

MESSUNG pro sinus-Block: Loss auf frischem sinus-Batch (immer
derselbe Eval-Seed => vergleichbar ueber Bloecke). Kennzahl:
Retention_10/1 = Loss(sinus, Block 10) / Loss(sinus, Block 1 nach
Training). Und: Anzahl der Bloecke bis Loss > 2x Block-1-Wert.

ZIEL: Wenn C signifikant langsamer vergisst (Retention naeher an 1,
mehr Bloecke bis Schwelle), ist der Homeostat ein echter
Kontinuierliches-Lernen-Baustein — DER AGI-relevante Satz.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.model import ContinuousRNN
from cgfn.benettin import benettin_ftle_max
from cgfn.continuous import make_continuous_batch
from cgfn.field import drive_input

BEN_TARGET = -0.3
PENALTY_W = 0.05
SEQ = ["sinus_memory", "lorenz", "ou_noise", "sinus_memory", "lorenz",
       "ou_noise", "sinus_memory", "lorenz", "ou_noise", "sinus_memory"]
BLOCK = 300


def kappa_penalty(model):
    W = model.W
    sv = torch.linalg.svdvals(W)[0]
    lam = torch.linalg.eigvals(W).abs().max()
    kappa = sv / lam.clamp_min(1e-9)
    return torch.nn.functional.relu(kappa - 2.0).pow(2)


def eval_sinus(model):
    torch.manual_seed(4242)
    x, y = make_continuous_batch("sinus_memory", 16, 32)
    out, _ = model(x)
    return torch.nn.MSELoss()(out, y).item()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--out", default="results/f23_marathon_results.txt")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 110)
    lines.append("F23 PLASTIZITAETS-MARATHON: 10 Task-Bloecke, sinus-Retention-Kurve (Bloecke 1,4,7,10)")
    lines.append(f"sequenz={SEQ} block={BLOCK} seeds={seeds}")
    lines.append("=" * 110)

    agg = {}
    for arm in ("A", "B", "C1"):
        for seed in seeds:
            torch.manual_seed(seed)
            m = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=32, tau=1.0)
            opt = torch.optim.Adam(m.parameters(), lr=2e-3)
            lossfn = torch.nn.MSELoss()
            xdr = drive_input(4, 48, 0.5)
            curve = []
            for bi, task in enumerate(SEQ):
                for step in range(BLOCK):
                    x, y = make_continuous_batch(task, 16, 16)
                    out, _ = m(x)
                    loss = lossfn(out, y)
                    if arm == "B":
                        loss = loss + PENALTY_W * kappa_penalty(m)
                    elif arm == "C1" and step % 50 == 0:
                        ben = benettin_ftle_max(m, xdr)
                        loss = loss + PENALTY_W * (ben - BEN_TARGET) ** 2
                    opt.zero_grad(); loss.backward(); opt.step()
                if task == "sinus_memory":
                    curve.append((bi + 1, eval_sinus(m)))
            lines.append(f"arm {arm} seed {seed}: " + " | ".join(f"B{b}:{l:.4f}" for b, l in curve))
            agg.setdefault(arm, []).append(curve)
    lines.append("")
    lines.append("--- Mittel: sinus-Retention (Loss je sinus-Block, gemittelt ueber Seeds) ---")
    summary = {}
    for arm in ("A", "B", "C1"):
        curves = agg[arm]
        n_blocks = len(curves[0])
        means = [sum(c[i][1] for c in curves) / len(curves) for i in range(n_blocks)]
        blocks = [curves[0][i][0] for i in range(n_blocks)]
        lines.append(f"arm {arm}: " + " | ".join(f"B{b}:{m:.4f}" for b, m in zip(blocks, means)))
        ret = means[-1] / max(means[0], 1e-9)
        thresh = None
        for i, m in enumerate(means):
            if m > 2 * means[0]:
                thresh = blocks[i]; break
        summary[arm] = (ret, thresh)
        lines.append(f"arm {arm}: Retention(End/Start) {ret:.2f} | 2x-Schwelle bei Block {thresh}")
    lines.append("")
    lines.append("Urteil: C1 mit Retention deutlich naeher 1 UND spaeterer 2x-Schwelle")
    lines.append("=> Homeostat = Kontinuierliches-Lernen-Baustein (AGI-relevanter Satz).")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
