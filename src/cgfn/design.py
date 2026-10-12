"""
F9 — Generatives Pendant zum Spectral Ruler: Kritikalitaet by construction.

Die Umkehrung der Diagnose: Statt Spektrum aus gegebenen Gewichten zu
messen, KONSTRUIEREN wir Gewichte mit vorgeschriebenen spektralen
Eigenschaften und testen, ob der Ruler das Verhalten vorhersagt,
BEVOR es gemessen wird (Validierungs-Dreieck: Messung/Theorie/Konstruktion).

Konstruktion via Schur-Zerlegung:
    W = Q T Q^T,  T = diag(eigenwerte) + s * strict-upper-noise
    - Re lambda_max(W) = lam_max exakt vorgegeben (dominanter Eigenwert)
    - Rest-Eigenwerte im Inneren der Scheibe |lambda| <= 0.9 * lam_max
    - s steuert die Nicht-Normalitaet: s=0 => normal (kappa~1),
      s gross => transientes Wachstum (kappa >> 1)

Damit ist die Vorhersage VORHER bekannt:
    tau* (Eigenwert-Regel)  = 1 / lam_max        — exakt, per Konstruktion
    tau* (SVD-Kante)        = 1 / sv_max(W)      — wächst mit s
Die Frage von F9: Welche Regel sagt das GEMESSENE Verhalten voraus?
Und: Liegt das Loss-Optimum (Memory-Task) dort, wo der Ruler es hinlegt?
"""
import torch

from cgfn.model import ContinuousRNN


def recurrent_matrix(hidden_dim, lam_max, s_nonnormal=1.0, seed=0):
    """
    W mit Re lambda_max(W) = lam_max (exakt), Nicht-Normalitaet ~ s_nonnormal.

    hidden_dim   : H
    lam_max      : dominanter reeller Eigenwert (Vorzeichen egal,
                   >0 => chaotisch ab tau > 1/lam_max moeglich)
    s_nonnormal  : Staerke der streng-obren Dreiecks-Stoerungen (kappa-Regler)
    """
    gen = torch.Generator().manual_seed(seed)
    # orthonormale Basis
    A = torch.randn(hidden_dim, hidden_dim, generator=gen)
    Q, _ = torch.linalg.qr(A)
    # Eigenwerte: 1x lam_max, Rest reell im Inneren (|lam| <= 0.9 lam_max)
    others = lam_max * 0.9 * (2.0 * torch.rand(hidden_dim - 1, generator=gen) - 1.0)
    eigs = torch.cat([torch.tensor([lam_max], dtype=torch.float32), others])
    T = torch.diag(eigs)
    if s_nonnormal > 0:
        upper = torch.triu(torch.randn(hidden_dim, hidden_dim, generator=gen), diagonal=1)
        T = T + s_nonnormal * lam_max * upper
    return Q @ T @ Q.T


def make_designed_model(hidden_dim, tau, lam_max, s_nonnormal=1.0, seed=0,
                        task_dim=0, freeze_w=True):
    """
    CT-RNN mit konstruiertem W. freeze_w=True => W bleibt waehrend des
    Trainings fix (Reservoir-Setting): die Dynamik ist vollstaendig
    designed, gelernt wird nur die Peripherie (U, readout). So bleibt
    die spektrale Vorhersage ueber das Training hinweg gueltig.
    """
    model = ContinuousRNN(input_dim=1, task_dim=task_dim,
                          hidden_dim=hidden_dim, tau=tau)
    W = recurrent_matrix(hidden_dim, lam_max, s_nonnormal, seed)
    with torch.no_grad():
        model.W.copy_(W)
    if freeze_w:
        model.W.requires_grad_(False)
    return model
