"""
F11 — Messplatz-Validierung: SVD-FTLE vs. Benettin-QR + Stabilitaets-Audit.

Kontext: F7/F8 zeigten die SVD-Regel (tau*sv_max*sqrt(Q)>1) als Champion
(93% / 100%), F10 die d-korrigierte Eigenwert-Regel als KEIN Ersatz (47.5%).
Aber unsere FTLE-Messung ist selbst SVD-basiert — Regel und Messgeraet
teilen dieselbe mathematische Struktur (Zirkularitaet, s. ROADMAP
Validierungs-Dreieck). Die Literatur (Benettin-Standard, Pseudospektren
bei Nicht-Normalitaet) sagt: SVD-FTLEs ueber kurze Rollouts sind bei
kappa>>1 von transientem Wachstum verzerrt.

F11-Fragen:
  (1) Messwuestigkeit: Wie gross ist |FTLE_svd - FTLE_benettin| als
      Funktion von kappa?
  (2) Regel-Guete unter dem Standard-Messgeraet: Folgt die WAHRE
      Lyapunov-Kante (Benettin) der Eigenwert- oder der SVD-Regel?
  (3) Stabilitaets-Audit: sv_max(J0)*dt vs. RK2-Grenze (~2) pro
      Konfiguration — welche Daten sind numerisch vertrauenswuerdig?

Benettin: Tangenten-Dynamik d(dz)/dt = (-I/tau + W diag(sech^2(z))) dz
wird SIMULTAN mit der Zustandsdynamik per RK2 (gleiche Substeps)
integriert; periodische QR-Reorthonormalisierung; akkumulierte
log-R-Diagonale / Zeit = Lyapunov-Spektrum (Standard-Verfahren).
"""
import torch


def _jac_diag(z):
    """diag(sech^2(z)) — lokale tanh-Ableitung entlang der Trajektorie."""
    return 1.0 - torch.tanh(z).pow(2)


def _dyn_pair(model, z, dz, x_t, task_vec):
    """Zustands- und Tangenten-Rechte-Seite als Paar (gleiche Input-Injektion).

    dz: (B, H, K) Tangenten-Buendel (K Tangentenvektoren pro Batch).
    Tangente:  d(dz)/dt = -dz/tau + W @ (diag(sech^2(z)) * dz)
    """
    if model.task_dim > 0:
        u = torch.cat([x_t, task_vec * model.task_scale], dim=-1)
    else:
        u = x_t
    f = -z / model.tau + torch.tanh(z) @ model.W.T + model.U(u)
    g = -dz / model.tau + torch.einsum("ij,bjk->bik", model.W,
                                       _jac_diag(z).unsqueeze(-1) * dz)
    return f, g


@torch.no_grad()
def benettin_spectrum(model, x_seq, task_vec=None, renorm_every=1):
    """
    Lyapunov-Spektrum via Benettin-QR (Standard-Messgeraet).

    Integriert (z, dz) gemeinsam mit dem SELBEN RK2/substeps-Schema wie
    ContinuousRNN.forward (Fairness: identische Diskretisierung), QR-
    Reorthonormalisierung pro Substep (renorm_every=1) oder seltener.
    Rueckgabe:
      spectrum  — (H,) absteigend sortiert, gemittelt ueber Batch,
                  in 1/ODE-Zeit
      t_total   — gesamte ODE-Zeit des Rollouts
    """
    B, T, _ = x_seq.shape
    H = model.hidden_dim
    device = x_seq.device
    if task_vec is None:
        task_vec = torch.zeros(B, 0, device=device)
    dt = model.dt

    z = torch.zeros(B, H, device=device)
    dz = torch.eye(H, device=device).unsqueeze(0).expand(B, H, H).contiguous()
    log_sum = torch.zeros(H, device=device)
    t_total = 0.0

    for t in range(T):
        x_t = x_seq[:, t]
        for s in range(model.substeps):
            f1, g1 = _dyn_pair(model, z, dz, x_t, task_vec)
            z2 = z + 0.5 * dt * f1
            dz2 = dz + 0.5 * dt * g1
            f2, g2 = _dyn_pair(model, z2, dz2, x_t, task_vec)
            z = z + dt * f2
            dz = dz + dt * g2
            t_total += dt
            if (s + 1) % renorm_every == 0:
                Q_, R_ = torch.linalg.qr(dz)
                diag_r = torch.diagonal(R_, dim1=-2, dim2=-1).abs()
                # Batch-Mittelung: jede Batch-Trajektorie ist ein Sample
                log_sum += torch.log(diag_r.clamp_min(1e-30)).mean(dim=0)
                dz = Q_
    spectrum = (log_sum / max(t_total, 1e-9)).sort(descending=True).values
    return spectrum, t_total


@torch.no_grad()
def benettin_ftle_max(model, x_seq, task_vec=None, renorm_every=1):
    """Groesster Lyapunov-Exponent (Benettin) — Vergleichswert zu ftle_summary."""
    spectrum, _ = benettin_spectrum(model, x_seq, task_vec, renorm_every)
    return spectrum[0].item()


@torch.no_grad()
def stability_audit(model, dt=None):
    """
    Numerisches Stabilitaets-Audit (Literatur-Punkt 7):
    Explizite RK-Verfahren haben eine endliche Stabilitaetsregion;
    fuer RK2 gilt grob |lambda*dt| < 2. Wir nehmen sv_max(J0) als
    konservative obere Schranke des Jacobi-Spektralrands (sv >= |lam|)
    und pruefen sv_max(J0)*dt < 2.

    Rueckgabe: dict mit sv_max_J0, svJ0_times_dt, rk2_ok.
    """
    if dt is None:
        dt = model.dt
    H = model.hidden_dim
    J0 = model.W.detach() - torch.eye(H, device=model.W.device) / model.tau
    sv_J0 = torch.linalg.svdvals(J0)[0].item()
    return {
        "sv_max_J0": sv_J0,
        "svJ0_times_dt": sv_J0 * dt,
        "rk2_ok": sv_J0 * dt < 2.0,
    }
