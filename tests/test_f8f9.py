"""Smoke-Tests für F8/F9-Module (design, ruler) — kein langes Training."""
import pytest
import torch

from cgfn.design import make_designed_model, recurrent_matrix
from cgfn.model import ContinuousRNN
from cgfn.ruler import spectral_report


def test_design_eig_max_exact():
    """Re lambda_max(W) muss vorgeschrieben sein (Kern des F9-Baus)."""
    for lam in (0.8, 1.15, 1.6):
        W = recurrent_matrix(16, lam, s_nonnormal=0.0, seed=0)
        ev = torch.linalg.eigvals(W).real.max().item()
        assert abs(ev - lam) < 1e-4, (lam, ev)


def test_kappa_grows_with_nonnormality():
    """s=0 => normal (kappa~1); s gross => kappa >> 1."""
    from cgfn.cavity import nonnormality
    k0, _, _ = nonnormality(_model_with(16, 1.15, 0.0))
    k3, _, _ = nonnormality(_model_with(16, 1.15, 3.0))
    assert k3 > k0
    assert k0 < 2.0  # normal


def _model_with(hidden_dim, lam, s):
    return make_designed_model(hidden_dim, 1.0, lam, s, seed=0, task_dim=0)


def test_freeze_w_flag():
    m = make_designed_model(16, 1.0, 1.15, 1.0, 0, task_dim=0, freeze_w=True)
    assert not m.W.requires_grad
    m2 = make_designed_model(16, 1.0, 1.15, 1.0, 0, task_dim=0, freeze_w=False)
    assert m2.W.requires_grad


def test_ruler_report_keys_and_prediction():
    """Ruler-Vorhersage ohne Rollout: tau_pred_eig = 1/lam_max per Bau."""
    model = _model_with(16, 1.15, 0.0)
    rep = spectral_report(model)
    required = {"tau", "lam_max", "sv_max", "kappa", "Q", "pred_eig", "pred_sv",
                "tau_pred_eig", "tau_pred_sv", "phase", "verdict"}
    assert all(k in rep for k in required), set(rep)
    assert abs(rep["lam_max"] - 1.15) < 1e-3
    assert abs(rep["tau_pred_eig"] - 1.0 / 1.15) < 1e-2
    assert isinstance(rep["verdict"], str) and len(rep["verdict"]) > 10


def test_ruler_with_states():
    model = _model_with(16, 1.15, 1.0)
    x = torch.randn(4, 16, 1)
    _, states = model(x)
    rep = spectral_report(model, states)
    assert 0.0 <= rep["Q"] <= 1.0
    assert 0.0 <= rep["dormant_frac"] <= 1.0
