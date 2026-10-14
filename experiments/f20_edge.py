"""
F20 — WARUM SITZT LERNEN AN DER KANTE? Der Mechanismus-Test.
(Die grosse Frage: Edge of Chaos als Zufall, Zwang, Selektion oder Optimum)

Hypothesen (gegenseitig ausschliesslich testbar):
  H1 GRADIENTEN-ZWANG: Der Task-Gradient zieht lambda Richtung 0.
      Test: designed Netze mit START-lambda in {-0.8,-0.4,-0.1,+0.1}
      (beidseits der Kante), 400 Steps Task-Training, TRAJEKTORIE
      von lambda Benettin. Zwang => alle laufen AUF 0 zu (von beiden
      Seiten). Random-Walk => Diffusion um Start.
  H2 SELEKTION: Lerngeschwindigkeit (Loss-Abfall) ist maximal nahe
      lambda=0. Test: pro Start-lambda die Zeit bis Loss-Schwelle.
      Wenn t*(lambda) ein Minimum bei 0 hat, lernen Kanten-Netze
      schneller => Selektionsmechanismus moeglich.
  H3 RANDOM-WALK-BARRIERE: lambda ist ein Drift mit Barriere: im
      Chaos (lambda>0) explodiert der Loss => Gradient drueckt zurueck.
      Test: Start bei lambda=+0.2 OHNE Trainingsabbruch: kehrt der
      Drift um? + Verteilung der Endpunkte nach langem Training
      (Barrieren-Walk: Anhaeufung knapp unter 0, nicht AT 0).
  H4 OPTIMUM: Der Endpunkt des Trainings koinzidiert mit dem
      MC-Optimum (F12: subkritisch!). Test: End-lambda vs. argmax MC
      auf demselben Netz-Gitter. Wenn Endpunkt << 0 = MC-Optimum,
      ist die Kante eine VERWECHSLUNG mit dem Ressourcen-Optimum.

MASSNAHME (unsere USP): lambda BENETTIN getrieben (F11-artefaktfrei),
Trajektorie alle 50 Steps; Start-lambda per Design praezise gesetzt.
Task: sinus_memory UND copy (Task-Abhaengigkeit der Antwort).
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.design import make_designed_model
from cgfn.benettin import benettin_ftle_max
from cgfn.memory_capacity import memory_capacity
from cgfn.continuous import make_continuous_batch
from cgfn.field import drive_input

LAM_TARGETS = (-0.8, -0.4, -0.1, 0.1)


def make_net_at_lambda(lam_target, seed, H=32):
    """Designed Netz, lam so gewaehlt, dass getriebener Benettin ~ lam_target.
    Wir nutzen lam (Design-Spektralradius) als Hebel und suchen per Bisection
    aus einem kleinen lam-Gitter den naechsten Treffer (grob reicht)."""
    best = None
    for lam in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        m = make_designed_model(H, 1.0, lam, 0.0, seed, task_dim=0)
        x = drive_input(4, 48, 0.5)
        ben = benettin_ftle_max(m, x)
        if best is None or abs(ben - lam_target) < best[2]:
            best = (lam, m, abs(ben - lam_target), ben)
    return best[1], best[0], best[3]


def train_track(m, steps, lr, task, track_every=50):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossfn = torch.nn.MSELoss()
    traj = []
    for step in range(steps):
        x, y = make_continuous_batch(task, 16, 16)
        out, _ = m(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % track_every == 0:
            xdr = drive_input(4, 48, 0.5)
            traj.append((step + 1, float(loss.item()), benettin_ftle_max(m, xdr)))
    return traj


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--out", default="results/f20_edge_results.txt")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 110)
    lines.append("F20 WARUM AN DER KANTE? Mechanismus-Test: Zwang vs Selektion vs Barriere vs Optimum")
    lines.append(f"seeds={seeds} steps={args.steps} lr={args.lr} — lambda Benettin getrieben (artefaktfrei), Start beidseits der Kante")
    lines.append("=" * 110)

    # ── H1 + H3: Trajektorien von beidseitigen Starts ──
    lines.append("")
    lines.append("--- H1/H3: lambda-Trajektorie pro Startpunkt (Task: sinus_memory) ---")
    lines.append(f"{'start':>7s} {'seed':>4s} {'lam0':>7s} {'lam1':>7s} {'dlam':>7s} {'loss_end':>9s}")
    outcomes = []
    for lam_t in LAM_TARGETS:
        for seed in seeds:
            m, lam_used, ben0 = make_net_at_lambda(lam_t, seed)
            traj = train_track(m, args.steps, args.lr, "sinus_memory")
            lam1 = traj[-1][2]
            lines.append(f"{lam_t:+7.1f} {seed:4d} {ben0:+7.3f} {lam1:+7.3f} {lam1-ben0:+7.3f} {traj[-1][1]:9.4f}")
            outcomes.append({"start": lam_t, "lam0": ben0, "lam1": lam1, "loss": traj[-1][1]})

    # H1-Statistik: ziehen alle auf 0?
    lines.append("")
    lines.append("--- H1: ZWANG? (dlam Richtung 0 von beiden Seiten?) ---")
    for lam_t in LAM_TARGETS:
        sub = [o for o in outcomes if o["start"] == lam_t]
        d = sum(o["lam1"] - o["lam0"] for o in sub) / len(sub)
        richtung = "auf 0 zu" if (lam_t < 0 and d > 0) or (lam_t > 0 and d < 0) else "weg von 0/diffus"
        lines.append(f"start {lam_t:+.1f}: mittel dlam {d:+.3f} ({richtung})")
    end_dist = [o["lam1"] for o in outcomes]
    near = sum(1 for l in end_dist if -0.15 < l < 0.15)
    lines.append(f"Endpunkte in (-0.15, 0.15): {near}/{len(end_dist)} — Anhaeufung AT 0 => Zwang; knapp UNTER 0 => Barriere")

    # ── H2: Lerngeschwindigkeit vs Start-lambda ──
    lines.append("")
    lines.append("--- H2: SELEKTION? Zeit bis Loss<0.02 pro Start-lambda ---")
    for lam_t in LAM_TARGETS:
        times = []
        for seed in seeds[:2]:
            m, _, _ = make_net_at_lambda(lam_t, seed)
            t = None
            opt = torch.optim.Adam(m.parameters(), lr=args.lr)
            for step in range(args.steps):
                x, y = make_continuous_batch("sinus_memory", 16, 16)
                out, _ = m(x)
                loss = torch.nn.MSELoss()(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
                if loss.item() < 0.02 and t is None:
                    t = step + 1
                    break
            times.append(t if t is not None else args.steps + 1)
        mt = sum(times) / len(times)
        lines.append(f"start {lam_t:+.1f}: mittel Zeit {mt:5.0f} Schritte")
    lines.append("Minimum bei start ~ -0.1/nahe 0 => Kanten-Netze lernen schneller (Selektion moeglich)")

    # ── H4: Endpunkt vs MC-Optimum ──
    lines.append("")
    lines.append("--- H4: OPTIMUM? MC vs lambda auf designed Gitter (s=0) ---")
    best = None
    for lam in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        mcs, bs = [], []
        for seed in seeds[:2]:
            m = make_designed_model(32, 1.0, lam, 0.0, seed, task_dim=0)
            x = drive_input(4, 48, 0.5)
            bs.append(benettin_ftle_max(m, x))
            torch.manual_seed(seed + 2000)
            mc, _ = memory_capacity(m, max_lag=12, seq_len=64, input_scale=0.5)
            mcs.append(mc)
        mc, ben = sum(mcs) / 2, sum(bs) / 2
        lines.append(f"lam {lam:.1f}: ben {ben:+.3f} | MC {mc:5.2f}")
        if best is None or mc > best[1]:
            best = (ben, mc)
    end_mittel = sum(end_dist) / len(end_dist)
    lines.append(f"MC-Optimum bei ben ~ {best[0]:+.3f} | Trainings-Endpunkte mittel {end_mittel:+.3f}")
    lines.append("Koinzidenz (<0.1 Abstand) => Kante ist Ressourcen-Optimum (H4); sonst eigenstaendig")

    lines.append("")
    lines.append("Lesart: Nur EINE Hypothese sollte das Muster erklaeren. Zwang => Kante ist")
    lines.append("notwendig (AGI-Architekturen muessen sie halten); Selektion => Kante ist")
    lines.append("optimierbar weg; Barriere => Kante ist trivial (Drift-Wand); Optimum =>")
    lines.append("Kante ist verwechselte Ressource. Jede Antwort ist ein anderer AGI-Bauplan.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
