#!/usr/bin/env python
"""F27 — Optimierer-Ressourcen-Interferenz (fix #R19):
Schneidet der SGD-Update-Schritt "kappa(W)-Kanalrauschen" ab? Wir messen den
Drift von kappa(W) (Bedingungszahl-Proxy) unter drei Optimierern
(SGD, SGD+Momentum, Adam) bei identischer Lerngeschwindigkeit (matched dL).
Erfahrungskurve g = <dL_t - dL_{t+1}>:  Zeigt der Adam-Lauf bei
kurzzeitigem  kappa-Drop  einen  Ruck in der Lernkurve (dL hoch)?

Design:
  - Reservoir-Setting: W sind frei (freeze_w=False), Ausgang linear
  - Optimierer mit gleicher Anfangs-dL-Verlustreduktion
  - g misst den quadratischen Gradient an W als Kraeftemass
"""
import argparse

import numpy as np
import torch

from cgfn.design import make_designed_model, kappa_of


def train_with_optimizer(opt_name, lr, steps, seed, log_every=5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reservoir: W bleibt trainierbar, loss haengt an W
    m = make_designed_model(32, 1.0, 1.0, 0.8, seed, task_dim=0, freeze_w=False)  # F27 misst W-Drift: W MUSS trainierbar sein
    if opt_name == "sgd":
        opt = torch.optim.SGD(m.parameters(), lr=lr)
    elif opt_name == "momentum":
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "adam":
        opt = torch.optim.Adam(m.parameters(), lr=lr)
    else:
        raise ValueError(opt_name)

    # Loss-Task: fixe Sequenz vorbereiten (Seed-getrennt vom Modell)
    gen = torch.Generator().manual_seed(seed + 1000)
    xs = torch.randn(1000, 64, 16, generator=gen)  # Buch?
    ys = torch.randn(1000, 1, 16, generator=gen)  # Rauschen-Kontrolle: y = Rauschen

    log = []
    g_sq_ema = None
    for t in range(steps):
        x = xs[t % 1000].unsqueeze(0)
        y = ys[t % 1000]
        opt.zero_grad()
        out, _ = m(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
        g = m.W.grad
        g_sq = (g ** 2).mean().item()
        g_sq_ema = g_sq if g_sq_ema is None else 0.9 * g_sq_ema + 0.1 * g_sq
        if t % log_every == 0 or t == steps - 1:
            log.append({
                "t": t,
                "loss": loss.item(),
                "g_sq": g_sq,
                "g_sq_ema": g_sq_ema,
                "kappa": float(kappa_of(m.W.detach()).cpu()),
            })
    return log, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/f27_optimizer_interference_results.txt")
    args = p.parse_args()

    lrs = {"sgd": 0.05, "momentum": 0.05, "adam": 0.01}
    results = {}
    for opt_name in ("sgd", "momentum", "adam"):
        log, _ = train_with_optimizer(opt_name, lrs[opt_name], args.steps, args.seed)
        results[opt_name] = log

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("F27 — Optimizer-Resource-Interferenz
")
        f.write("=" * 60 + "

")
        for opt_name, log in results.items():
            f.write(f"
[{opt_name}] lr={lrs[opt_name]}
")
            for row in log:
                f.write(f"  t={row['t']:>4}  loss={row['loss']:.6f}  "
                        f"g_sq={row['g_sq']:.6f} (ema {row['g_sq_ema']:.6f})  "
                        f"kappa={row['kappa']:.4f}
")
        f.write("
Interpretation:
")
        f.write("  - Wenn Adam kappa-Drift zeigt, wo SGD keinen zeigt: Ressourcen-Interferenz.
")
    print(f"[i] F27 abgeschlossen — Ergebnisse in {args.out}")


if __name__ == "__main__":
    main()
