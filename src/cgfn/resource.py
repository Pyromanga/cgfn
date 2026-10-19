"""
Runde XI — Der Ressourcen-Kern: DMFT-Selbstschluss, Sigma-Buchhaltung,
Kappa-Zoo und Optimierer-Kokkyklus.

Tiefenanalyse XI (docs/tiefenanalyse_2026-10-20_XI.md) findet vier
Zirkularitaeten im eigenen Messgeraet-Stapel. Dieses Modul liefert die
mathematisch geschlossenen Gegenstuecke:

  (1) dmft_fixed_point: schliesst die Linearisierungs-Schleife
      d <-> Sigma. d = E[sech^2(z)] als Funktion der Zustandsvarianz
      (1D-Quadratur, geschlossen), Sigma = Loesung der
      Lyapunov-Gleichung der linearisierten getriebenen Dynamik.
  (2) mc_linear_prediction: EXAKTE lineare Theorie fuer Memory
      Capacity (optimaler linearer Readout pro Lag, Mahalanobis-
      Form) plus H2-Rauschverstaerkung Tr(Sigma)/sigma_u^2 — beide
      Funktionale aus (J, B) allein, ohne Rollout.
      ACHTUNG (R3 der Tiefenanalyse XI): MC_opt und Tr(Sigma)/sigma^2
      sind VERSCHIEDENE Funktionale (Skalar-Beispiel: MC=1 fuer alle
      b, a — Tr Sigma/sigma^2 = b^2/(1-a^2)). Beide sind Ressourcen-
      Projektionen; keines ist die 'Identitaet' des anderen.
  (3) kappa_zoo / transient_growth: die vier ehrlichen Nicht-
      Normalitaets-Invarianten (Kondition, Eigenvektor-Kondition,
      Henrici-Abweichung, Kreiss-Untergrenze) und das direkt
      gemessene transiente Wachstum g = sup_t sv_max(exp(J t)).
      Kreiss-Schranke: K_Kreiss <= g <= e*n*K_Kreiss.
  (4) update_cocycle_growth: transienter Verstaerkungsfaktor des
      Heavy-Ball-Kompagnon-Operators. EoS-Sharpness ist Fixpunkt
      dieses Operators, kein Landschafts-Eigentum; der Kompagnon
      wird bei (1+beta-eta*h)^2 = 4*beta defekt (pro Modus).

Alle Funktionen deterministisch (keine Zufallsziehung ausser in
explizit seeds-gesteuerten Proben).
"""
import math

import torch


@torch.no_grad()
def sech2_gauss_moment(s2, n_grid=161, span=8.0):
    """E[sech^2(z)] fuer z ~ N(0, s2), elementweise, via Trapez-Gitter.

    Geschlossene 1D-Quadratur (kein Sampling) — der Baustein, der die
    DMFT-Schleife d <-> Sigma zu einer echten Fixpunkt-Iteration macht.
    """
    s2 = torch.as_tensor(s2, dtype=torch.float64)
    shape = s2.shape
    s = s2.reshape(-1).clamp_min(1e-10).sqrt()
    t = torch.linspace(-span, span, n_grid, dtype=torch.float64)
    pdf = torch.exp(-0.5 * (t.unsqueeze(0) / s.unsqueeze(1)) ** 2) / (
        (math.sqrt(2.0 * math.pi) * s).unsqueeze(1))
    vals = (1.0 - torch.tanh(t) ** 2).unsqueeze(0)
    m = torch.trapezoid(vals * pdf, t, dim=1)
    return m.clamp(1e-6, 1.0).reshape(shape).to(torch.float32)


def _lyapunov_ct(J, Q):
    """Loese  J S + S J^T + Q = 0  (row-major vec, Kron-Linearisierung)."""
    H = J.shape[0]
    I = torch.eye(H, dtype=J.dtype)
    A = torch.kron(J, I) + torch.kron(I, J)
    return torch.linalg.solve(A, -Q.reshape(-1)).reshape(H, H)


def _lyapunov_dt(A, C):
    """Loese  S = A S A^T + C  (diskret, stabil)."""
    H = A.shape[0]
    M = torch.eye(H * H, dtype=A.dtype) - torch.kron(A, A)
    return torch.linalg.solve(M, C.reshape(-1)).reshape(H, H)


@torch.no_grad()
def dmft_fixed_point(model, input_scale=0.5, damp=0.5, tol=1e-4,
                     max_iter=300, dtype=torch.float64):
    """Selbst-konsistenter DMFT-Fixpunkt d <-> Sigma (R1-Schnitt).

    Modell (vgl. benettin._dyn_pair):
        dz/dt = -z/tau + tanh(z) W^T + U u(t)
    Input: pro Sample weisses Rauschen, Varianz input_scale^2, gehalten
    fuer Delta = substeps*dt  (effektive PSD ~ input_scale^2 * Delta).
    Mean-Field: z_i ~ N(0, Sigma_ii), Korrelationen vernachlaessigt;
        d_i = E[sech^2(z_i)]  (Quadratur, geschlossen)
        J = -I/tau + W diag(d)
        J Sigma + Sigma J^T = -Q,  Q = q_eff * B B^T

    Rueckgabe dict oder None (kein stabiler Fixpunkt: supra-lineare
    Kante — auch das ist eine Aussage: dort existiert kein stationaerer
    Betriebspunkt, den der Homeostat halten koennte).
    """
    H = model.hidden_dim
    W = model.W.detach().to(dtype)
    B = model.U.weight.detach().to(dtype)          # (H, in_dim)
    Delta = float(model.substeps) * float(model.dt)
    q = float(input_scale) ** 2 * Delta
    Q = q * (B @ B.T)
    eye = torch.eye(H, dtype=dtype)

    d = torch.ones(H, dtype=dtype)
    Sigma = None
    converged = False
    it = 0
    for it in range(max_iter):
        J = -eye / float(model.tau) + W * d.unsqueeze(0)
        if torch.linalg.eigvals(J).real.max().item() >= -1e-6:
            return None
        try:
            Sigma = _lyapunov_ct(J, Q)
        except Exception:
            return None
        var = torch.diagonal(Sigma).clamp_min(0.0)
        d_new = sech2_gauss_moment(var).to(dtype)
        if (d_new - d).abs().max().item() < tol:
            d = d_new
            converged = True
            break
        d = damp * d + (1.0 - damp) * d_new
    J = -eye / float(model.tau) + W * d.unsqueeze(0)
    lam_field = torch.linalg.eigvals(J).real.max().item()
    return {
        "d": d.to(torch.float32), "Sigma": Sigma, "J": J, "Q": Q,
        "lam_field": lam_field,
        "tr_sigma": Sigma.diagonal().sum().item(),
        "converged": converged, "iters": it + 1,
    }


def mc_linear_prediction(J, B, input_scale=0.5, Delta=0.1, max_lag=16,
                        dtype=torch.float64):
    """Exakte lineare Theorie fuer MC und H2 aus (J, B) allein (R3-Schnitt).

    Diskretisierung bei stueckweise-konstantem Input (exakt fuer die
    Linearisierung):
        A = exp(J Delta),  b = J^{-1} (A - I) B
        Sigma = A Sigma A^T + sigma^2 b b^T
        corr2_max(lag k) = sigma^2 * Tr( m_k^T Sigma^{-1} m_k ),
                          m_k = A^{k-1} B
        MC_opt = Summe k=1..max_lag   (optimaler linearer Readout pro
        Lag — genau das, wogegen Jaeger-MC mit Ridge Readout konvergiert)
        h2 = Tr(Sigma)/sigma^2  (Rauschverstaerkung, NICHT MC-Decke!)
    Rueckgabe dict oder None (falls J instabil).
    """
    H = J.shape[0]
    eye = torch.eye(H, dtype=dtype)
    if torch.linalg.eigvals(J).real.max().item() >= -1e-6:
        return None
    A = torch.matrix_exp(J * float(Delta))
    b = torch.linalg.solve(J, (A - eye) @ B.to(dtype))
    sig2 = float(input_scale) ** 2
    S = _lyapunov_dt(A, sig2 * (b @ b.T))
    Sinv = torch.linalg.inv(S + 1e-9 * eye)
    per_lag = []
    m = b.clone()
    for _k in range(max_lag):
        per_lag.append((sig2 * torch.trace(m.T @ Sinv @ m)).item())
        m = A @ m
    return {
        "mc_per_lag": per_lag, "mc_opt": sum(per_lag),
        "h2": (S.diagonal().sum() / sig2).item(),
        "A": A, "b": b, "Sigma_dt": S,
    }


def kappa_zoo(J):
    """Die vier ehrlichen Nicht-Normalitaets-Invarianten von J (R2-Schnitt).

    kappa_cond : sv_max/sv_min (Kondition — was field.py fälschlich
                 'Kreiss' nennt)
    kappa_T    : cond(V) der Eigenvektoren (die 'echte' Nicht-Normalitaet
                 diagonalisierbarer Matrizen)
    delta_H    : Henrici-Abweichung sqrt(||J||_F^2 - sum |lam_i|^2)
    kreiss_lb  : Untergrenze der Kreiss-Konstante (nur reelle Achse):
                 max_s 1/sv_min(J - s I)
    """
    J = J.to(torch.float64)
    H = J.shape[0]
    sv = torch.linalg.svdvals(J)
    kappa_cond = (sv[0] / sv[-1].clamp_min(1e-12)).item()
    eig = torch.linalg.eig(J)
    lam, V = eig.eigenvalues, eig.eigenvectors
    kappa_T = torch.linalg.cond(V).item()
    delta_H = (J.norm() ** 2 - (lam.abs() ** 2).sum()).clamp_min(0.0).sqrt().item()
    re = lam.real
    grid = torch.linspace(re.min().item(), re.max().item(), 33)
    kreiss_lb = 0.0
    for s in grid:
        r = torch.linalg.svdvals(J - s * torch.eye(H, dtype=J.dtype))[-1].item()
        if r > 1e-12:
            kreiss_lb = max(kreiss_lb, 1.0 / r)
    return {"kappa_cond": kappa_cond, "kappa_T": kappa_T,
            "delta_H": delta_H, "kreiss_lb": kreiss_lb}


def transient_growth(J, n_t=41):
    """g = sup_t sv_max(exp(J t)) (J stabil) und Zeitpunkt t*.

    Direkt gemessen (keine Abschaetzung). Einordnung:
    K_Kreiss <= g <= e*n*K_Kreiss (Trefethen-Embree).
    """
    J = J.to(torch.float64)
    lam = torch.linalg.eigvals(J).real.max().item()
    t_max = min(60.0, max(4.0, 4.0 / max(-lam, 1e-3)))
    best, t_star = 1.0, 0.0
    for t in torch.linspace(0.0, t_max, n_t).tolist():
        g = torch.linalg.svdvals(torch.matrix_exp(J * float(t)))[0].item()
        if g > best:
            best, t_star = g, float(t)
    return best, t_star


def normal_excess(J, Q):
    """E = Tr(Sigma)/Tr(Sigma_norm): gleiche Spektrum, normale Koordinaten.

    Sigma_norm: Loesung mit J_norm = diag(Re lambda_i) — das Referenz-
    system, das die Fenster-These als 'lambda-Achse allein' vorhersagt.
    E > 1 ist der exakte kappa-Beitrag zur Ressource Tr(Sigma).
    """
    J = J.to(torch.float64)
    lam = torch.linalg.eigvals(J).real
    S = _lyapunov_ct(J, Q)
    Sn = _lyapunov_ct(torch.diag(lam), Q)
    tr = S.diagonal().sum().item()
    trn = Sn.diagonal().sum().item()
    return tr / max(trn, 1e-12), S, Sn


@torch.no_grad()
def empirical_sigma(model, T=3000, input_scale=0.5, burn=200, seed=1234):
    """Rollierende Kovarianz aus EINEM Rollout — das billige Ressourcen-
    Messgeraet: Tr(Sigma_emp) ist ohne QR und ohne Tangenten-Integration
    direkt beobachtbar (Homeostat-v3-Zielgroesse).
    Rueckgabe: (Sigma (H,H) float64, d_emp (H,) float64).
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, 1, generator=gen) * input_scale
    _, states = model(x, None)
    z = states[0, burn:].to(torch.float64)
    z = z - z.mean(dim=0, keepdim=True)
    S = (z.T @ z) / z.shape[0]
    d_emp = (1.0 - torch.tanh(z) ** 2).mean(dim=0)
    return S, d_emp


def update_cocycle_growth(h, eta, beta, t_max=600, n_probe=4, seed=0):
    """Transient-Verstaerkung des Zwillings-Perturbations-Deltas unter
    Heavy-Ball auf quadratischem Verlust mit Hessian diag(h) (R4-Schnitt).

    Kompagnon-Operator (pro Modus 2x2, nicht-normal auch fuer
    diagonales H):
        M_i = [[1+beta-eta*h_i, -beta], [1, 0]]
    Defektpunkt pro Modus: (1+beta-eta*h_i)^2 = 4*beta — dort divergiert
    die Eigenvektor-Kondition des Update-Kokkyklus. g_opt ist der direkt
    gemessene sup-Verstaerkungsfaktor (Zwillings-Proben-Analogon, hier
    analytisch gepowert statt teuer simuliert).
    """
    h = torch.as_tensor(h, dtype=torch.float64)
    n = h.numel()
    M = torch.zeros(2 * n, 2 * n, dtype=torch.float64)
    M[:n, :n] = (1.0 + beta) * torch.eye(n, dtype=torch.float64) - eta * torch.diag(h)
    M[:n, n:] = -beta * torch.eye(n, dtype=torch.float64)
    M[n:, :n] = torch.eye(n, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    g_opt = 1.0
    for _ in range(n_probe):
        delta = torch.randn(2 * n, generator=gen)
        delta = delta / delta.norm()
        cur = delta.clone()
        for _ in range(t_max):
            cur = M @ cur
            g_opt = max(g_opt, cur.norm().item())
            if cur.norm() > 1e12:
                break
    return g_opt
