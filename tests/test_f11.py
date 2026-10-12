"""Smoke-Tests für F11 (Benettin-QR-Messgeraet + Stabilitaets-Audit)."""
import torch

from cgfn.benettin import benettin_ftle_max, benettin_spectrum, stability_audit
from cgfn.continuous import make_continuous_batch
from cgfn.design import make_designed_model
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def test_benettin_spectrum_finite_and_sorted():
    model = ContinuousRNN(task_dim=0, hidden_dim=8, tau=1.5)
    x, _ = make_continuous_batch("sinus_memory", 2, 12)
    spec, t_total = benettin_spectrum(model, x)
    assert spec.shape == (8,)
    assert torch.isfinite(spec).all()
    assert spec[0] >= spec[-1] - 1e-6  # absteigend sortiert
    assert t_total > 0


def test_benettin_decaying_net_negative():
    """Kleine Kopplung, tau gross => alles kontrahiert => FTLE < 0."""
    model = ContinuousRNN(task_dim=0, hidden_dim=8, tau=2.0)
    with torch.no_grad():
        model.W.copy_(model.W * 0.01)
    x, _ = make_continuous_batch("sinus_memory", 2, 24)
    assert benettin_ftle_max(model, x) < 0.0


def test_benettin_unstable_net_positive():
    """Sehr grosse Kopplung => Instabilitaet => FTLE > 0."""
    model = ContinuousRNN(task_dim=0, hidden_dim=8, tau=2.0)
    with torch.no_grad():
        model.W.copy_(model.W * 8.0)
    x, _ = make_continuous_batch("sinus_memory", 2, 24)
    assert benettin_ftle_max(model, x) > 0.0


def test_stability_audit_flags():
    model = ContinuousRNN(task_dim=0, hidden_dim=8, tau=1.0)
    a = stability_audit(model)
    assert 0.0 < a["svJ0_times_dt"]
    assert isinstance(a["rk2_ok"], bool)
    # extrem grosses W muss failen
    with torch.no_grad():
        model.W.copy_(model.W * 30.0)
    b = stability_audit(model)
    assert not b["rk2_ok"]


def test_benettin_designed_normal_net():
    """kappa=1 designed: Benettin muss endliche Werte liefern."""
    model = make_designed_model(8, 1.0, 1.15, 0.0, seed=0, task_dim=0)
    x, _ = make_continuous_batch("sinus_memory", 2, 16)
    f = benettin_ftle_max(model, x)
    assert -50.0 < f < 50.0
