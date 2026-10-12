"""
F8 — Spectral Ruler: Diagnose-Tool.

Eine Funktion, ein Befund: `spectral_report(model, states)` liest aus der
Gewichtsmatrix ALLEIN (plus optional einer Trajektorie fuer die Saettigung
Q), ob das Netz am aktuellen tau chaotisch ist, wo der Kritikalitaets-
Sweet-Spot liegen MUSS (tau* ~ 1/Re lambda_max bzw. 1/(sv_max*sqrt(Q)))
und wie nicht-normal die Dynamik ist (kappa).

Diagnose-Fragen, die das Tool beantwortet:
  - Ist das Netz am aktuellen tau chaotisch / subkritisch / kritisch?
  - Welches tau waere der Sweet-Spot (ohne einen einzigen Rollout)?
  - Ist die Dynamik stark nicht-normal (transientes Wachstum moeglich)?
  - Wie viele Units sind dormant (Plasticity-Loss-Fruehwarnung)?
  - Wie breit ist der Abstand zwischen Eigenwert- und SVD-Vorhersage
    (= die Nische, in der Mean-Field versagt, s. F6)?
"""
import torch

from cgfn.cavity import (
    chaos_criterion,
    eig_max,
    effective_gain,
    nonnormality,
    q_factor,
    spectral_edge,
)


@torch.no_grad()
def activation_stats(states, dormant_threshold=0.1):
    """Sättigung, d_eff-Ratio und Dormant-Anteil aus Zuständen (B,T,H)."""
    if states.dim() == 4:
        states = states.squeeze(0)
    z = states.reshape(-1, states.shape[-1])
    tanh_z = torch.tanh(z)
    saturation = tanh_z.pow(2).mean().item()  # Q
    # d_eff (Participation Ratio der Kovarianz)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    d_eff = ((eig.sum() ** 2) / (eig ** 2).sum()).item()
    # dormant: Unit im Mittel fast null-aktiv
    unit_act = tanh_z.abs().mean(dim=0)  # (H,)
    dormant = (unit_act < dormant_threshold).float().mean().item()
    return {
        "saturation_Q": saturation,
        "d_eff_ratio": d_eff / states.shape[-1],
        "dormant_frac": dormant,
    }


@torch.no_grad()
def spectral_report(model, states=None):
    """
    Der Spectral Ruler als Ein-Aufruf-Diagnose.

    states=None => Ursprungs-Naeherung (Q=1): reine W-Analyse ohne Rollout.
    states=(B,T,H) => mit Saettigungs-Korrektur aus der Trajektorie.

    Liefert u.a.:
      tau_pred_eig = 1 / Re lambda_max(W)      — lineare Sweet-Spot-Vorhersage
      tau_pred_sv  = 1 / (sv_max(W)*sqrt(Q))   — nicht-normale Kante
      verdict      — Klartext-Diagnose
    """
    if states is not None:
        crit = chaos_criterion(model, states)
        act = activation_stats(states)
    else:
        # Ursprungs-Naeherung: Q=1, keine Dormant-Info
        g = effective_gain(model)
        lam = eig_max(model)
        sv = spectral_edge(model)
        kappa, _, _ = nonnormality(model)
        crit = {
            "g": g, "Q": 1.0, "tau_g_sqrtQ": model.tau * g,
            "eig_max": lam, "sv_max": sv,
            "pred_mf": model.tau * g > 1.0,
            "pred_eig": model.tau * lam > 1.0,
            "pred_sv": model.tau * sv > 1.0,
        }
        act = {"saturation_Q": 1.0, "d_eff_ratio": None, "dormant_frac": None}

    lam = crit["eig_max"]
    sv = crit["sv_max"]
    Q = crit["Q"]
    tau_pred_eig = 1.0 / max(lam, 1e-9)
    tau_pred_sv = 1.0 / max(sv * max(Q, 1e-9) ** 0.5, 1e-9)

    if crit["pred_eig"]:
        phase = "chaotisch (Eigenwert-Regel: instabil am Ursprung)"
    elif crit["pred_sv"]:
        phase = "nicht-normal transient (SVD-Kante instabil, Eigenwert stabil)"
    else:
        phase = "subkritisch (stabil)"
    margin = model.tau * lam - 1.0
    verdict = (
        f"tau={model.tau:g}: {phase}. "
        f"Sweet-Spot-Vorhersage: tau*~{tau_pred_eig:.2f} (Eigenwert) / "
        f"{tau_pred_sv:.2f} (SVD-Kante). "
        f"Stabilitaetsrand tau*lam_max-1 = {margin:+.3f}."
    )

    return {
        "tau": model.tau,
        "lam_max": lam,
        "sv_max": sv,
        "kappa": sv / max(abs(lam), 1e-9),
        "Q": Q,
        "g": crit["g"],
        "pred_mf": crit["pred_mf"],
        "pred_eig": crit["pred_eig"],
        "pred_sv": crit["pred_sv"],
        "tau_pred_eig": tau_pred_eig,
        "tau_pred_sv": tau_pred_sv,
        "stability_margin": margin,
        "phase": phase,
        "verdict": verdict,
        **act,
    }
