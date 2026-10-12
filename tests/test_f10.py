"""Smoke-Tests für F10 (Sättigungs-korrektur: qbar_factor, eig_qbar-Regel)."""
import torch

from cgfn.cavity import chaos_criterion, q_factor, qbar_factor
from cgfn.design import make_designed_model
from cgfn.model import ContinuousRNN
from cgfn.ruler import spectral_report
from cgfn.tasks import TASKS, make_batch


def test_qbar_plus_q_equals_one():
    """d = <sech^2> und Q = <tanh^2> sind elementweise Komplemente."""
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=1.5)
    idx = TASKS.index("parity")
    x, _ = make_batch("parity", batch_size=4, seq_len=8)
    tv = torch.zeros(4, len(TASKS)); tv[:, idx] = 1.0
    _, states = model(x, tv)
    Q = q_factor(model, states)
    d = qbar_factor(model, states)
    assert abs((Q + d) - 1.0) < 1e-5, (Q, d)
    assert 0.0 <= d <= 1.0


def test_chaos_criterion_has_qbar_keys():
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=1.5)
    idx = TASKS.index("parity")
    x, _ = make_batch("parity", batch_size=4, seq_len=8)
    tv = torch.zeros(4, len(TASKS)); tv[:, idx] = 1.0
    _, states = model(x, tv)
    crit = chaos_criterion(model, states)
    for k in ("Qbar", "tau_eig_qbar", "pred_eig_qbar"):
        assert k in crit, k
    assert isinstance(crit["pred_eig_qbar"], bool)
    # Korrigierte Regel ist konservativer als die Ursprungs-Regel (d<=1)
    assert crit["tau_eig_qbar"] <= crit["tau_eig"] + 1e-6


def test_ruler_report_qbar_prediction():
    """Mit states: tau_pred_eig_qbar = 1/(lam*d) muss im Report stehen."""
    model = make_designed_model(16, 1.0, 1.15, 0.0, seed=0, task_dim=0)
    x = torch.randn(4, 16, 1)
    _, states = model(x)
    rep = spectral_report(model, states)
    assert "tau_pred_eig_qbar" in rep and "Qbar" in rep and "pred_eig_qbar" in rep
    lam, d = rep["lam_max"], rep["Qbar"]
    assert abs(rep["tau_pred_eig_qbar"] - 1.0 / (lam * d)) < 1e-3
    assert rep["tau_pred_eig_qbar"] >= rep["tau_pred_eig"] - 1e-6  # d<=1


def test_ruler_without_states_qbar_defaults():
    """Ohne states: d=1 => qbar-Vorhersage == Ursprungs-Vorhersage."""
    model = make_designed_model(16, 1.0, 1.15, 0.0, seed=0, task_dim=0)
    rep = spectral_report(model)  # states=None
    assert abs(rep["tau_pred_eig_qbar"] - rep["tau_pred_eig"]) < 1e-9
    assert rep["Qbar"] == 1.0
