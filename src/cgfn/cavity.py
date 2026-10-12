"""
F6 — Mean-Field / Cavity-Theorie-Anschluss (ROADMAP).

Sompolinsky / Van Vreeswijk Chaos-Kriterium für rekurrente tanh-Netze
    dz/dt = -z/tau + W tanh(z) + U x,   W_ij ~ N(0, g^2 / N)
lautet (nach Zeitreskalierung t -> tau*t, W -> tau*W):
    Chaos  <=>  tau * g * sqrt(Q) > 1,   Q = <tanh(z)^2>

F7-Erweiterung: DREI konkurrierende Chaos-Regeln aus der Gewichtsmatrix:
    (a) Mean-Field:     tau * g * sqrt(Q) > 1            (Sompolinsky, Bulk)
    (b) Eigenwert:      tau * Re lambda_max(W) > 1       (exakte lineare
                        Stabilitaet am Ursprung: J0 = -I/tau + W)
    (c) Singulärwert:   tau * sv_max(W) * sqrt(Q) > 1
                        (nicht-normale spektrale Kante; nahe an der
                        SVD-basierten FTLE-Messung)
plus Nicht-Normalitaet  kappa = sigma_max(W) / |lambda_max(W)|  —
Verdacht aus F6: die 21 Kriterium-Misses (alle konservativ, alle mit
kleinem Q) kommen von Nicht-Normalitaet + SVD-basierter FTLE-Messung.

Anschluss an Clark (arXiv:2610.12426): volles Spektrum der linearisierten
Jacobi gegen Trajektorien-FTLE (Spektrum-Korrelation).
"""
import math

import torch


@torch.no_grad()
def effective_gain(model):
    """g = std(W) * sqrt(H) — Sompolinsky-Konvention W_ij ~ N(0, g^2/N)."""
    return model.W.detach().std().item() * (model.hidden_dim ** 0.5)


@torch.no_grad()
def q_factor(model, states):
    """Q = <tanh(z)^2> über Batch, Zeit und Units — Sättigungsmaß."""
    if states.dim() == 4:
        states = states.squeeze(0)
    return torch.tanh(states).pow(2).mean().item()


@torch.no_grad()
def linearized_spectrum(model):
    """
    Singulärwerte der Jacobi am Fixpunkt z=0 (sech^2(0)=1):
        J0 = -I/tau + W
    (Batch-frei, da W nicht von der Trajektorie abhängt.)
    """
    H = model.hidden_dim
    J0 = model.W.detach() - torch.eye(H, device=model.W.device) / model.tau
    return torch.linalg.svdvals(J0)


@torch.no_grad()
def spectral_edge(model):
    """sv_max(W) — größter Singulärwert der Gewichtsmatrix."""
    return torch.linalg.svdvals(model.W.detach())[0].item()


@torch.no_grad()
def eig_max(model):
    """Re lambda_max(W) — größter Realteil des Eigenwertspektrums."""
    ev = torch.linalg.eigvals(model.W.detach())
    return ev.real.max().item()


@torch.no_grad()
def nonnormality(model):
    """
    kappa = sv_max(W) / max(|lambda_i(W)|, eps).
    kappa=1: normale Matrix; kappa>>1: starke Nicht-Normalitaet
    (transientes Wachstum moeglich, SVD-FTLE > Eigenwert-Lyapunov).
    """
    sv = spectral_edge(model)
    ev = torch.linalg.eigvals(model.W.detach())
    lam = ev.abs().max().item()
    return sv / max(lam, 1e-9), sv, lam


@torch.no_grad()
def chaos_criterion(model, states):
    """
    Alle drei Chaos-Regeln als Referenz. Liefert die Größen als dict.
    'chaos_predicted' bleibt Mean-Field (Backward-kompatibel zu F6).
    """
    g = effective_gain(model)
    Q = q_factor(model, states)
    lin = linearized_spectrum(model)
    sv_W = spectral_edge(model)
    lam_W = eig_max(model)
    mf = model.tau * g * math.sqrt(Q)           # (a) Mean-Field
    eig_rule = model.tau * lam_W                # (b) > 1 <=> instabil am Ursprung
    sv_rule = model.tau * sv_W * math.sqrt(max(Q, 1e-9))  # (c) nicht-normale Kante
    return {
        "g": g,
        "Q": Q,
        "tau_g_sqrtQ": mf,
        "eig_max": lam_W,
        "sv_max": sv_W,
        "tau_eig": eig_rule,
        "tau_sv": sv_rule,
        "pred_mf": mf > 1.0,
        "pred_eig": eig_rule > 1.0,
        "pred_sv": sv_rule > 1.0,
        "chaos_predicted": mf > 1.0,
        "lin_sv_max": lin.max().item(),
        "lin_log_sv_max": torch.log(lin.max() + 1e-12).item(),
    }


@torch.no_grad()
def compare_spectrum(model, states, dt=1.0):
    """
    Volles Spektrum: linearisierte Jacobi (Fixpunkt-Theorie) gegen
    gemessenes Trajektorien-FTLE (cgfn.ftle.ftle_spectrum) — Anschluss
    an Clark (arXiv:2610.12426): Spektrum gegen Cavity-Referenz stellen.
    """
    from cgfn.ftle import ftle_spectrum

    lin_sv = torch.log(linearized_spectrum(model) + 1e-12)
    traj = ftle_spectrum(model, states, dt=dt)
    lin_sorted, _ = torch.sort(lin_sv, descending=True)
    traj_sorted, _ = torch.sort(traj, descending=True)
    corr = torch.corrcoef(torch.stack([lin_sorted, traj_sorted]))[0, 1].item()
    return {
        "lin_spectrum": lin_sorted,
        "traj_spectrum": traj_sorted,
        "spectrum_corr": corr,
        "lin_max": lin_sorted[0].item(),
        "traj_max": traj_sorted[0].item(),
    }
