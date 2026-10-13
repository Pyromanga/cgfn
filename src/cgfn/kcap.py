"""
F14 — kappa-Cap: Nicht-Normalitaet als gradientenbasiertes Bau-Rezept.

F12 zeigte: kappa ist ein NICHT-MONOTONER Ressourcen-Regler — moderates
kappa (~4) kann helfen, kappa>>1 loescht die Memory Capacity fast aus
(corr(kappa, MC) bis -0.99). Der kappa-Cap macht aus diesem Befund ein
Trainingsziel: eine differentierbare Penalty, die sv_max(W) relativ
zum dominanten Eigenwert deckelt:

    pen = relu( sv_max(W) / |Re lam_max(W)|  -  kappa_target )^2

Eigenschaften:
  - Differentierbar in W (sv_max via 2-Norm; Eigenwert als SKALAR
    detached — Eigenwert-Gradienten sind bei fast-entarteten Spektren
    numerisch unzuverlaessig, der Skalar-Level ist robust genug).
  - kappa-invariant unter Skalierung von W: das Verhaeltnis (nicht die
    Groesse) ist das regulierte Objekt — der Cap bremst die NICHT-
    NORMALITAET, nicht die Kapazitaet.
  - Bei kappa <= kappa_target exakt 0: der Cap greift nur, wenn das
    Training von selbst in den schaedlichen Bereich laeuft (F14-Frage 1).
"""
import torch

from .cavity import eig_max


@torch.no_grad()
def kappa_of(model):
    """kappa = sv_max(W) / |Re lam_max(W)| — dasselbe Mass wie F7/F9/F12."""
    sv = torch.linalg.svdvals(model.W)[0].item()
    lam = abs(float(eig_max(model)))
    return sv / max(lam, 1e-9)


def kappa_cap_penalty(model, kappa_target=4.0):
    """relu(sv_max/|lam_max| - kappa_target)^2 — differentierbar in W.

    Der Eigenwert geht als detacheter Skalar ein (s. Modul-Docstring).
    Rueckgabe: 0-dim Tensor (zum Addieren auf den Task-Loss).
    """
    sv = torch.linalg.matrix_norm(model.W, ord=2)
    lam = max(abs(float(eig_max(model))), 1e-9)  # Skalar, detached
    ratio = sv / lam
    return torch.relu(ratio - kappa_target) ** 2
