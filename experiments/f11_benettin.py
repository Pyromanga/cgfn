"""
F11 — Messplatz-Validierung: SVD-FTLE vs. Benettin-QR + Stabilitaets-Audit.

Fragen (s. src/cgfn/benettin.py):
  (1) Messwuestigkeit: |FTLE_svd - FTLE_benettin| als Funktion von kappa?
  (2) Regel-Guete unter Benettin: Eigenwert- oder SVD-Regel?
  (3) Stabilitaets-Audit: welche Konfigurationen sind RK2-vertrauenswuerdig?

Design: dieselben Netze wie F8/F9 (designed s=0/3 + trainierte g=0.8/1.6),
pro Netz: ein Rollout, beide Messgeraete, beide Regeln, Audit-Flags.
"""
import argparse
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_ftle_max, stability_audit
from cgfn.cavity import chaos_criterion
from cgfn.continuous import make_continuous_batch
from cgfn.design import make_designed_model
from cgfn.ftle import ftle_summary
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def rollout_x(task, batch, seq_len):
    """Eingabe-Sequenz (mit Task-Vec falls noetig) fuer beide Messgeraete."""
    if task == "copy":
        x, _ = make_batch("copy", batch_size=batch, seq_len=seq_len)
        tv = torch.zeros(batch, len(TASKS))
        tv[:, TASKS.index("copy")] = 1.0
        return x, tv
    x, _ = make_continuous_batch("sinus_memory", batch, seq_len)
    return x, None


def run_config(model, x, tv):
    _, states = model(x, tv)
    ftle_svd = ftle_summary(model, states)["ftle_max"]
    ftle_ben = benettin_ftle_max(model, x, tv)
    crit = chaos_criterion(model, states)
    audit = stability_audit(model)
    return ftle_svd, ftle_ben, crit, audit


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--taus", default="0.5,1.0,2.0")
    p.add_argument("--lams", default="1.15")
    p.add_argument("--g-scales", default="0.8,1.6")
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--out", default="results/f11_benettin_results.txt")
    args = p.parse_args()

    seeds = [int(v) for v in args.seeds.split(",")]
    taus = [float(v) for v in args.taus.split(",")]
    lams = [float(v) for v in args.lams.split(",")]
    g_scales = [float(v) for v in args.g_scales.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 116)
    lines.append("F11 MESSPLATZ-VALIDIERUNG: SVD-FTLE vs. Benettin-QR — folgt die wahre Lyapunov-Kante Eig oder SV?")
    lines.append(f"seeds={seeds} taus={taus} lams={lams} g_scales={g_scales} steps={args.train_steps} hidden={args.hidden_dim}")
    lines.append("=" * 116)
    lines.append(f"{'config':>26s} {'tau':>5s} {'kappa':>6s} {'svdt':>7s} {'benettin':>9s} "
                 f"{'agree':>6s} {'pEig':>5s} {'pSV':>5s} {'svJ0dt':>7s} {'rk2ok':>5s}")

    hits_svd_meas = {"eig": [0, 0], "sv": [0, 0]}   # Regel vs. Benettin-Vorzeichen
    hits_svd_rule = {"eig": [0, 0], "sv": [0, 0]}   # Regel vs. SVD-Vorzeichen
    agree_total = [0, 0]
    diffs = []  # (kappa, |ftle_svd - ftle_ben|)

    def eval_config(name, model, tau):
        for seed in seeds:
            torch.manual_seed(seed)
        x, tv = rollout_x("sinus_memory", 8, 32)
        ftle_svd, ftle_ben, crit, audit = run_config(model, x, tv)
        meas_ben = ftle_ben > 0.0
        meas_svd = ftle_svd > 0.0
        agree = meas_ben == meas_svd
        agree_total[0] += int(agree); agree_total[1] += 1
        for rule, pred in (("eig", crit["pred_eig"]), ("sv", crit["pred_sv"])):
            hits_svd_meas[rule][1] += 1
            if pred == meas_ben:
                hits_svd_meas[rule][0] += 1
            hits_svd_rule[rule][1] += 1
            if pred == meas_svd:
                hits_svd_rule[rule][0] += 1
        kappa = crit["sv_max"] / max(abs(crit["eig_max"]), 1e-9)
        diffs.append((kappa, abs(ftle_svd - ftle_ben)))
        line = (f"{name:>26s} {tau:5.2f} {kappa:6.2f} {ftle_svd:+7.3f} {ftle_ben:+9.4f} "
                f"{str(agree):>6s} {str(crit['pred_eig']):>5s} {str(crit['pred_sv']):>5s} "
                f"{audit['svJ0_times_dt']:7.2f} {str(audit['rk2_ok']):>5s}")
        lines.append(line)
        print(line, flush=True)

    # (a) Designed Networks: kappa-Serje s=0 vs s=3
    for lam in lams:
        for s_nn in (0.0, 3.0):
            for tau in taus:
                model = make_designed_model(args.hidden_dim, tau, lam, s_nn,
                                            seeds[0], task_dim=0)
                eval_config(f"designed lam={lam} s={s_nn}", model, tau)

    # (b) Trainierte Netze (kurz, wie F7 'trained'-Mode)
    for g_scale in g_scales:
        for tau in taus:
            torch.manual_seed(seeds[0])
            model = ContinuousRNN(input_dim=1, task_dim=0,
                                   hidden_dim=args.hidden_dim, tau=tau)
            with torch.no_grad():
                model.W.copy_(model.W * g_scale)
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            lossfn = torch.nn.MSELoss()
            for _ in range(args.train_steps):
                x, y = make_continuous_batch("sinus_memory", 16, 16)
                out, _ = model(x)
                loss = lossfn(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
            eval_config(f"trained g={g_scale}", model, tau)

    lines.append("")
    lines.append("--- Aggregat ---")
    lines.append(f"Vorzeichen-Einheit SVD vs. Benettin: {agree_total[0]}/{agree_total[1]}")
    lines.append(f"Regel-Guete vs. BENETTIN-Vorzeichen: eig={hits_svd_meas['eig'][0]}/{hits_svd_meas['eig'][1]} "
                 f"({hits_svd_meas['eig'][0]/max(1,hits_svd_meas['eig'][1]):.0%}), "
                 f"sv={hits_svd_meas['sv'][0]}/{hits_svd_meas['sv'][1]} "
                 f"({hits_svd_meas['sv'][0]/max(1,hits_svd_meas['sv'][1]):.0%})")
    lines.append(f"Regel-Guete vs. SVD-Vorzeichen (Replikat): eig={hits_svd_rule['eig'][0]}/{hits_svd_rule['eig'][1]}, "
                 f"sv={hits_svd_rule['sv'][0]}/{hits_svd_rule['sv'][1]}")
    if diffs:
        ds = sorted(diffs)
        lines.append(f"|FTLE_svd - FTLE_benettin|: min={min(d for _,d in ds):.3f} "
                      f"median={ds[len(ds)//2][1]:.3f} max={max(d for _,d in ds):.3f}")
        # Korrelation kappa vs Differenz
        kappas = torch.tensor([k for k, _ in diffs])
        dd = torch.tensor([d for _, d in diffs])
        if kappas.numel() > 2 and kappas.std() > 1e-6 and dd.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([kappas, dd]))[0, 1].item()
            lines.append(f"corr(kappa, |Differenz|) = {corr:+.2f}  (>0: Messabweichung waechst mit Nicht-Normalitaet)")
    lines.append("")
    lines.append("LESEAT: Wenn eig unter Benettin deutlich aufholt (und sv einbricht),")
    lines.append("war die SVD-Dominanz von F7/F8 ein Messartefakt — der Ruler braucht")
    lines.append("das Benettin-Messgeraet als Referenz. Wenn sv auch unter Benettin")
    lines.append("gewinnt, ist die SVD-Kante physikalisch real und das Zwei-Regime-")
    lines.append("Gesetz Ruler=f(kappa) besteht ECHT, nicht nur messgeraetbedingt.")
    lines.append("rk2ok=False-Zeilen sind numerisch mindergueltig (Literatur-Punkt 7).")
    lines.append("=== FERTIG ===")
    out = "\n".join(lines) + "\n"
    Path(args.out).write_text(out, encoding="utf-8")
    print(out)
    print(f"[i] Ergebnisse nach {args.out} geschrieben.")


if __name__ == "__main__":
    main()
