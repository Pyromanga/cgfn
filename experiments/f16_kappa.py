"""
F16 — kappa-Drift-Audit ueber Architekturen: SGD-Selbst-Sabotage der
Nicht-Normalitaet und der kappa-Cap als Gegenmittel (ROADMAP Kern-Claim).

F14 zeigte am CT-RNN: freies Task-Training treibt kappa 2.27 -> 7.82 in
den F12-schaedlichen Bereich; der gradientenbasierte kappa-Cap haelt es
kostenguenstig und verbessert MC UND Retention. F13 zeigte: GRU hat
dasselbe Symptom (rho(W_hh) 0.65 -> 1.88 unter Training). F16 testet den
Claim ARCHITEKTUR-UEBERGREIFEND (der entscheidende Schritt fuer Paper-
Relevanz — nnRNN/commutativity-Regularisierung sind benachbart, aber
"kappa-Drift unter freiem SGD + doppelter Cap-Nutzen" ist nirgends
explizit gezeigt):

Arme (pro Modell: ct / gru / cornn):
  free : plain Adam auf sinus_memory
  kcap : + kappa_weight * relu(kappa(W_aktuell) - kappa_target)^2
         mit W_aktuell = das REKURRENTE Gewicht des Modells
         (ct: W; gru: weight_hh-Stack; cornn: linearisierte J-Bloecke)

Fragen:
  (1) Drift: steigt kappa unter free bei ALLEN Architekturen?
      (cornn ist der Kontrast-Anker: von Konstruktion ~1 — haelt SGD das?)
  (2) Cap: haelt kcap kappa unten — und was kostet es am Task-Loss?
  (3) Nutzen: aendert der Cap MC / getriebene Lyapunov-Lage messbar?

Messung pro Arm/Seed: kappa-Trajektorie (alle K Steps), finaler
Holdout-Loss, MC (Jaeger), getriebener Benettin-FTLE (driven_lyap).
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.continuous import make_continuous_batch
from cgfn.driven_lyap import driven_measures
from cgfn.memory_capacity import memory_capacity
from cgfn.zoo import make_zoo_model


# ── kappa-Masse pro Architektur (differentierbar wo noetig) ──

def _blocks_eigmax(W):
    """max |lambda| ueber die 3 Gatter-Bloecke (detached, Skalar)."""
    return max(torch.linalg.eigvals(B).abs().max().item() for B in W.chunk(3, dim=0))


def kappa_value(model):
    """kappa = sv_max / |lam_max| (numerisch, fuer Logging)."""
    W = _recurrent_weight(model)
    sv = torch.linalg.svdvals(W.detach())[0].item()
    lam = _spectral_radius(model)
    return sv / max(abs(lam), 1e-9)


def _recurrent_weight(model):
    m = model.core
    if model.name == "ct":
        return m.W
    if model.name == "gru":
        return m.weight_hh
    # cornn: linearisierte Schritt-Map als (2H, 2H)-Matrix
    H, dt, gam = m.hidden_dim, m.dt, m.gamma
    I = torch.eye(H)
    Jxx = (1 - dt) * I + dt * m.Wx
    Jxy = dt * m.Wy - dt * gam * I
    Jyx = dt * m.Wyx
    Jyy = (1 - gam * dt) * I
    return torch.cat([torch.cat([Jxx, Jxy], dim=1),
                      torch.cat([Jyx, Jyy], dim=1)], dim=0)


def _spectral_radius(model):
    m = model.core
    if model.name == "ct":
        return torch.linalg.eigvals(m.W.detach()).abs().max().item()
    if model.name == "gru":
        return _blocks_eigmax(m.weight_hh.detach())
    return torch.linalg.eigvals(_recurrent_weight(model).detach()).abs().max().item()


def kappa_penalty(model, kappa_target):
    """relu(sv_max/|lam_max| - kappa_target)^2 — differentierbar in W.

    Eigenwert als detached Skalar (robust, s. kcap.py-Rationale).
    """
    W = _recurrent_weight(model)
    sv = torch.linalg.matrix_norm(W, ord=2)
    lam = max(abs(_spectral_radius(model)), 1e-9)
    return torch.relu(sv / lam - kappa_target) ** 2


# ── Training ──

def train(model, steps, lr, arm, kappa_target, kappa_weight, kap_track,
          track_every=50, holdout=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.MSELoss()
    for step in range(steps):
        x, y = make_continuous_batch("sinus_memory", 16, 16)
        out, _ = model(x)
        loss = lossfn(out, y)
        if arm == "kcap":
            loss = loss + kappa_weight * kappa_penalty(model, kappa_target)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % track_every == 0 or step == 0:
            kap_track.append(kappa_value(model))


def holdout_loss(model, batches=8):
    lossfn = torch.nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_continuous_batch("sinus_memory", 16, 16)
            out, _ = model(x)
            total += lossfn(out, y).item()
    return total / batches


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--models", default="ct,gru,cornn")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=12)
    p.add_argument("--kappa-target", type=float, default=4.0)
    p.add_argument("--kappa-weight", type=float, default=0.02)
    p.add_argument("--out", default="results/f16_kappa_results.txt")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    lines.append("=" * 118)
    lines.append("F16 KAPPA-DRIFT-AUDIT: SGD-Selbst-Sabotage der Nicht-Normalitaet ueber Architekturen + kappa-Cap")
    lines.append(f"models={models} seeds={seeds} steps={args.steps} lr={args.lr} H={args.hidden_dim}")
    lines.append(f"kappa_target={args.kappa_target} kappa_weight={args.kappa_weight} — Task: sinus_memory (MSE), kappa-Tracking alle 50 Steps")
    lines.append("kappa = sv_max/|lam_max| des REKURRENTEN Gewichts (gru: weight_hh-Stack, cornn: linearisierte Schritt-Map)")
    lines.append("=" * 118)
    lines.append(f"{'model':>7s} {'arm':>5s} {'seed':>4s} {'kap0':>6s} {'kap1':>6s} {'dkap':>6s} "
                 f"{'loss':>7s} {'MC':>6s} {'ben':>8s} {'svd':>7s}")

    results = []
    for name in models:
        for arm in ("free", "kcap"):
            for seed in seeds:
                torch.manual_seed(seed)
                model = make_zoo_model(name, hidden_dim=args.hidden_dim, seed=seed)
                track = [kappa_value(model)]
                torch.manual_seed(seed + 1000)
                train(model, args.steps, args.lr, arm, args.kappa_target,
                      args.kappa_weight, track)
                loss = holdout_loss(model)
                torch.manual_seed(seed + 2000)
                mc, _ = memory_capacity(model, max_lag=args.max_lag,
                                        seq_len=64, input_scale=0.5)
                x_drive = torch.randn(4, 48, 1) * 0.5
                m = driven_measures(model, x_drive)
                r = {"model": name, "arm": arm, "seed": seed,
                     "kap0": track[0], "kap1": track[-1],
                     "dkap": track[-1] - track[0],
                     "loss": loss, "mc": mc,
                     "ben": m["lyap_ben"], "svd": m["lyap_svd"]}
                results.append(r)
                line = (f"{name:>7s} {arm:>5s} {seed:4d} {r['kap0']:6.2f} {r['kap1']:6.2f} "
                        f"{r['dkap']:+6.2f} {r['loss']:7.4f} {r['mc']:6.2f} "
                        f"{r['ben']:+8.3f} {r['svd']:+7.3f}")
                lines.append(line)
                print(line, flush=True)

    # ── Aggregate ──
    lines.append("")
    lines.append("--- (1) kappa-Drift unter freiem SGD, pro Architektur ---")
    for name in models:
        sub = [r for r in results if r["model"] == name and r["arm"] == "free"]
        if not sub:
            continue
        dk = sum(r["dkap"] for r in sub) / len(sub)
        lines.append(f"{name:>7s} free: d_kappa={dk:+.2f} ({'DRIFT' if dk > 0.5 else 'stabil'})")
    lines.append("")
    lines.append("--- (2) Cap-Wirkung: kcap vs free ---")
    lines.append(f"{'model':>7s} {'dkap_free':>9s} {'dkap_kcap':>10s} {'dloss':>7s} {'dMC':>6s} {'dben':>7s}")
    for name in models:
        fr = [r for r in results if r["model"] == name and r["arm"] == "free"]
        kc = [r for r in results if r["model"] == name and r["arm"] == "kcap"]
        if not fr or not kc:
            continue
        def mean(rs, key): return sum(r[key] for r in rs) / len(rs)
        dkap_f, dkap_k = mean(fr, "dkap"), mean(kc, "dkap")
        dloss = mean(kc, "loss") - mean(fr, "loss")
        dmc = mean(kc, "mc") - mean(fr, "mc")
        dben = mean(kc, "ben") - mean(fr, "ben")
        lines.append(f"{name:>7s} {dkap_f:+9.2f} {dkap_k:+10.2f} {dloss:+7.4f} "
                     f"{dmc:+6.2f} {dben:+7.3f}")
    lines.append("")
    lines.append("Lesart: (1) d_kappa > 0 bei allen Architekturen unter free => der")
    lines.append("F14-Befund ist generisch: Task-SGD selbst sabotiert die Nicht-")
    lines.append("Normalitaets-Ressource. cornn ist der Konstruktionstest: von Hand")
    lines.append("bei kappa~1 gepinnt — haelt SGD das? (2) kcap haelt d_kap klein")
    lines.append("ohne Loss-Schaden (dloss ~ 0) => billiges universelles Gegenmittel;")
    lines.append("dMC/dben positiv => Cap sichert die Ressource, nicht nur die Zahl.")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
