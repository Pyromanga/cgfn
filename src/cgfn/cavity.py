"""
F6 — Mean-Field / Cavity-Theorie-Anschluss (ROADMAP).

Sompolinsky / Van Vreeswijk Chaos-Kriterium für rekurrente tanh-Netze
    dz/dt = -z/tau + W tanh(z) + U x,   W_ij ~ N(0, g^2 / N)
lautet (nach Zeitreskalierung t -> tau*t, W -> tau*W):
    Chaos  <=>  tau * g * sqrt(Q) > 1,   Q = <tanh(z)^2>

Hier: analytische Referenzwerte + empirischer Abgleich gegen die
gemessenen FTLE-Werte aus cgfn.ftle. Anschluss an arXiv:2610.12426
(Clark, vollständiges Lyapunov-Spektrum random-asymmetrischer RNNs):
das volle Spektrum der linearisierten Jacobi wird mit dem
Trajektorien-FTLE-Spektrum verglichen.
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
def chaos_criterion(model, states):
    """
    Sompolinsky-Kriterium (mit tau-Reskalierung) + linearisierte Jacobi
    als Referenz. Liefert alle Größen als dict.
    """
    g = effective_gain(model)
    Q = q_factor(model, states)
    lin = linearized_spectrum(model)
    criterion = model.tau * g * math.sqrt(Q)
    return {
        "g": g,
        "Q": Q,
        "tau_g_sqrtQ": criterion,
        "chaos_predicted": criterion > 1.0,
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
