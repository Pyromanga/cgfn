"""
F13 — Getriebene Lyapunov-Messung fuer BELIEBIGE Schritt-Maps.

Generalisierung des F11-Benettin-Messgeraets: Statt die CT-Form
dz/dt = -z/tau + W tanh(z) + Ux vorauszusetzen, messen wir die
WAHRE Schritt-Jacobi J_t = d z_{t+1} / d z_t entlang der getriebenen
Trajektorie — exakt via Autograd (torch.func.jacrev) — und betreiben
darauf diskreten Benettin-QR:

    M_t = J_t Q_t;  Q_{t+1}, R = QR(M_t);  lambda = (1/T) sum log|diag R|

Das ist modell-agnostisch (GRU, coRNN, CT-RNN, transformer-artige
Zellzustands-Maps — alles mit step(z,x)->z' messbar) und damit die
Allgemeingueltigkeits-Antwort: Unser Messvertrag ist nicht an unsere
Dynamik gebunden.

Zusätzlich das SVD-Pendant (F11-Vergleich): mittleres log sv_max(J_t)
entlang derselben Trajektorie — wenn die beiden Messungen
systematisch auseinanderlaufen (kappa-abhaengig), gilt der
F11-Artefakt-Befund FUER DIE PRAXIS-ARCHITEKTUREN.
"""
import torch
from torch.func import jacrev


def _jacobian_step(step_fn, z, x_in):
    """Exakte (H,H)-Jacobi der Schritt-Map bei (z, x_in) — autograd."""
    return jacrev(lambda zz: step_fn(zz, x_in))(z)


@torch.no_grad()
def driven_measures(model, x_seq, device="cpu"):
    """Benettin-QR + SVD-Fenster entlang der getriebenen Trajektorie.

    model braucht nur: step(z, x) -> z' (Zoo-Vertrag).
    x_seq: (B, T, D) Antrieb (z.B. weisses Rauschen — dieselben
    Statistiken wie memory_capacity).

    Rueckgabe dict:
      lyap_ben   — groesster diskreter Lyapunov-Exponent (1/Schritt),
                   Batch-Mittel — Standard-Messgeraet (ehrlich).
      lyap_svd   — mean_t log sv_max(J_t), Batch-Mittel — das, was
                   SVD-basierte Kurzfenster-Messung sieht.
      kappa_traj — mittleres sv_max/|eig|max(J_t) entlang der Fahrt
                   (Nicht-Normalitaet UNTER GETRIEB).
      max_abs_eig_t — groesstes |eig(J_t)| (diagnose).
    """
    B, T, D = x_seq.shape
    H = model.state_dim
    step_fn = model.step

    lyap_ben_batch = []
    lyap_svd_batch = []
    kappa_traj_batch = []
    max_eig_batch = []

    for b in range(B):
        z = torch.zeros(H, device=x_seq.device)
        Q = torch.eye(H, device=x_seq.device)
        log_sum = torch.zeros(H, device=x_seq.device)
        sv_log_sum = 0.0
        kap_sum = 0.0
        big_eig = 0.0
        for t in range(T):
            x_in = x_seq[b, t]
            J = _jacobian_step(step_fn, z, x_in)
            # diskreter Benettin: Q-R-Zerlegung des Produkt-Schritts
            M = J @ Q
            Q_, R_ = torch.linalg.qr(M)
            diag_r = torch.diagonal(R_).abs().clamp_min(1e-30)
            log_sum = log_sum + torch.log(diag_r)
            Q = Q_
            # SVD-Fenster-Mass + Trajektorien-Nicht-Normalitaet
            sv = torch.linalg.svdvals(J)
            sv_log_sum += torch.log(sv[0].clamp_min(1e-30)).item()
            ev = torch.linalg.eigvals(J).abs().max().item()
            big_eig = max(big_eig, ev)
            kap_sum += (sv[0] / max(ev, 1e-9)).item()
            z = step_fn(z, x_in)
        lyap_ben_batch.append((log_sum / T).max().item())
        lyap_svd_batch.append(sv_log_sum / T)
        kappa_traj_batch.append(kap_sum / T)
        max_eig_batch.append(big_eig)

    def mean(xs):
        return sum(xs) / len(xs)

    return {
        "lyap_ben": mean(lyap_ben_batch),
        "lyap_svd": mean(lyap_svd_batch),
        "kappa_traj": mean(kappa_traj_batch),
        "max_abs_eig_t": mean(max_eig_batch),
    }
