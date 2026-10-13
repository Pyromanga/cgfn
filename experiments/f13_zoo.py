"""
F13 — Zoo-Audit: bestehende Architekturen (GRU, coRNN, CT-RNN) unter
unserem getriebenen Messgeraet — wo irren die Design-Kriterien der Praxis?

Fragen:
  (1) Design-Fehlurteile: Wie oft besteht ein Modell das klassische
      Design-Kriterium (CT: tau*rho(W)<1, diskret: rho(W)<1 — Echo-State/
      Reservoir-Standard), ist unter TREIBUNG aber lyapunov-positiv
      (chaotisch im Einsatz)? Und der umgekehrte Fall (konservativ
      verworfen, aber stabil unter Treibung)?
  (2) Artefakt-Generalisierung: corr(kappa_traj, |lyap_svd - lyap_ben|)
      ueber ALLE Architekturen — gilt der F11-Befund (SVD-Messung
      blaeht bei Nicht-Normalitaet auf) modell-uebergreifend?
  (3) Ressource: MC (Jaeger) pro Modell vs. getriebenem Kanten-Abstand —
      tragen auch fremde Architekturen ihre Kapazitaet nahe der
      GETRIEBENEN Kante (statt der Design-Kante)?

Design: pro (modell, seed, phase): design() (ungetrieben, das was
Designer sehen) vs. driven_measures() (Benettin-QR auf exakten
Schritt-Jacobians, weisses Rauschen) + memory_capacity(). Phase
'init' = zufaellige Initialisierung, 'trained' = 200 Adam-Steps
auf sinus_memory. Zusaetzlich Skalen-Check: ct mit H=16/32/64.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.continuous import make_continuous_batch
from cgfn.driven_lyap import driven_measures
from cgfn.memory_capacity import memory_capacity
from cgfn.zoo import make_zoo_model


def train_briefly(model, steps, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.MSELoss()
    for _ in range(steps):
        x, y = make_continuous_batch("sinus_memory", 16, 16)
        out, _ = model(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--models", default="ct,gru,cornn")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--max-lag", type=int, default=12)
    p.add_argument("--scale-check", action="store_true",
                   help="zusaetzlich ct bei H=16/64 (Allgemeingueltigkeit)")
    p.add_argument("--out", default="results/f13_zoo_results.txt")
    args = p.parse_args()

    seeds = [int(v) for v in args.seeds.split(",")]
    models = args.models.split(",")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 132)
    lines.append("F13 ZOO-AUDIT: Design-Kriterien der Praxis vs. getriebene Lyapunov-Realitaet (GRU / coRNN / CT-RNN)")
    lines.append(f"models={models} seeds={seeds} H={args.hidden_dim} train={args.train_steps} max_lag={args.max_lag}")
    lines.append("design_ok = klassisches Kriterium (CT: tau*rho<1, diskret: rho<1); lyap_ben = Benettin-QR auf exakten Schritt-Jacobians (weisses Rauschen)")
    lines.append("=" * 132)
    lines.append(f"{'model':>8s} {'H':>4s} {'seed':>4s} {'phase':>7s} {'rho_W':>6s} {'sv_W':>6s} {'kap_d':>6s} "
                 f"{'designOK':>8s} {'lyap_ben':>9s} {'lyap_svd':>9s} {'kap_tr':>6s} {'agree':>6s} {'MC':>6s}")

    rows = []  # dicts

    def audit_model(name, hidden_dim, seed):
        torch.manual_seed(seed)
        model = make_zoo_model(name, hidden_dim=hidden_dim, seed=seed)
        for phase in ("init", "trained"):
            if phase == "trained":
                torch.manual_seed(seed + 1000)
                train_briefly(model, args.train_steps)
            d = model.design()
            x_drive = torch.randn(4, 48, 1) * 0.5
            m = driven_measures(model, x_drive)
            torch.manual_seed(seed + 2000)
            mc, _ = memory_capacity(model, max_lag=args.max_lag, seq_len=64,
                                    input_scale=0.5)
            agree = (m["lyap_ben"] > 0) == (m["lyap_svd"] > 0)
            row = {"model": name, "H": hidden_dim, "seed": seed, "phase": phase,
                   **d, **m, "agree": agree, "mc": mc}
            rows.append(row)
            line = (f"{name:>8s} {hidden_dim:4d} {seed:4d} {phase:>7s} "
                    f"{d['rho_W']:6.2f} {d['sv_W']:6.2f} {d['kappa']:6.2f} "
                    f"{str(d['design_ok']):>8s} {m['lyap_ben']:+9.4f} "
                    f"{m['lyap_svd']:+9.4f} {m['kappa_traj']:6.2f} "
                    f"{str(agree):>6s} {mc:6.2f}")
            lines.append(line)
            print(line, flush=True)

    for name in models:
        for seed in seeds:
            audit_model(name, args.hidden_dim, seed)

    if args.scale_check and "ct" in models:
        lines.append("--- Skalen-Check (Allgemeingueltigkeit): ct bei H=16/64 ---")
        for H in (16, 64):
            for seed in seeds:
                audit_model("ct", H, seed)

    # ── Aggregate ──
    lines.append("")
    lines.append("--- (1) Design-Fehlurteile vs. getriebene Realitaet ---")
    for name in models:
        sub = [r for r in rows if r["model"] == name]
        n = len(sub)
        false_stable = [r for r in sub if r["design_ok"] and r["lyap_ben"] > 0]
        false_chaotic = [r for r in sub if not r["design_ok"] and r["lyap_ben"] <= 0]
        lines.append(f"{name:>7s}: {len(false_stable)}/{n} 'Design sagt stabil, getrieben chaotisch' | "
                     f"{len(false_chaotic)}/{n} 'Design verwirft, getrieben stabil'")

    lines.append("")
    lines.append("--- (2) Artefakt-Generalisierung: corr(kappa_traj, |lyap_svd - lyap_ben|) ---")
    if len(rows) > 2:
        ks = torch.tensor([r["kappa_traj"] for r in rows])
        ds = torch.tensor([abs(r["lyap_svd"] - r["lyap_ben"]) for r in rows])
        if ks.std() > 1e-6 and ds.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([ks, ds]))[0, 1].item()
            lines.append(f"ueber ALLE Architekturen/Zeilen: corr = {corr:+.2f} "
                         f"(F11-Anker auf CT: +0.51)")
    dis = sum(1 for r in rows if not r["agree"])
    lines.append(f"Vorzeichen-Diskordanz SVD vs. Benettin: {dis}/{len(rows)} Zeilen")

    lines.append("")
    lines.append("--- (3) Ressource: MC vs. getriebenem Kanten-Abstand ---")
    for name in models:
        sub = [r for r in rows if r["model"] == name]
        best = max(sub, key=lambda r: r["mc"])
        lines.append(f"{name:>7s}: bestes MC={best['mc']:.2f} bei {best['phase']} seed={best['seed']} "
                     f"(lyap_ben={best['lyap_ben']:+.3f}, design rho={best['rho_W']:.2f})")

    lines.append("")
    lines.append("LESEAT: (1) hohe Fehlurteile => die Praxis-Design-Kriterien")
    lines.append("messen die ungetriebene Strecke — genau der F11-Befund, jetzt")
    lines.append("an fremden Architekturen. (2) positive Korrelation ueber ALLE")
    lines.append("Modelle => SVD-Kurzfenster-Messung ist architektur-uebergreifend")
    lines.append("verzerrt (nicht nur ein CT-RNN-Artefakt). (3) MC-Gipfel nahe")
    lines.append("lyap_ben~0 => getriebene Kante ist die universelle Ressourcen-Lage.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
