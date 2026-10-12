"""Smoke-Tests für F4/F5/F6-Module (continuous, cavity) — kein langes Training."""
import pytest
import torch

from cgfn.cavity import (
    chaos_criterion,
    compare_spectrum,
    effective_gain,
    linearized_spectrum,
    q_factor,
)
from cgfn.continuous import CONTINUOUS_TASKS, make_continuous_batch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


@pytest.mark.parametrize("task", CONTINUOUS_TASKS)
def test_continuous_batch_shapes_and_finiteness(task):
    torch.manual_seed(0)
    x, y = make_continuous_batch(task, batch_size=4, seq_len=16)
    assert x.shape == y.shape == (4, 16, 1), task
    assert torch.isfinite(x).all() and torch.isfinite(y).all(), task
    assert x.abs().max() <= 2.0, task  # sinus/lorenz-scaled/OU sind sanft normiert


def test_continuous_training_reduces_loss():
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=0, hidden_dim=16, tau=1.5)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossfn = torch.nn.MSELoss()
    x, y = make_continuous_batch("lorenz", batch_size=8, seq_len=16)
    first = None
    loss = None
    for _ in range(60):
        out, _ = model(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    assert loss is not None and loss.item() < first


def test_cavity_reference_values():
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=1.5)
    idx = TASKS.index("parity")
    x, _ = make_batch("parity", batch_size=4, seq_len=8)
    tv = torch.zeros(4, len(TASKS))
    tv[:, idx] = 1.0
    _, states = model(x, tv)

    spec = linearized_spectrum(model)
    assert spec.shape == (16,)
    assert torch.isfinite(spec).all()

    g = effective_gain(model)
    assert 0.0 < g < 100.0

    Q = q_factor(model, states)
    assert 0.0 <= Q <= 1.0  # tanh^2 in [0,1]

    crit = chaos_criterion(model, states)
    # F7 hat chaos_criterion um pred_mf/pred_eig/pred_sv, tau_eig/tau_sv,
    # eig_max/sv_max erweitert — Subset-Pruefung statt exakter Key-Menge,
    # damit neue Diagnose-Keys den Smoke-Test nicht mehr rot machen.
    required = {"g", "Q", "tau_g_sqrtQ", "chaos_predicted",
                "lin_sv_max", "lin_log_sv_max"}
    assert all(k in crit for k in required), set(crit)
    assert isinstance(crit["chaos_predicted"], bool)
    # F7-Erweiterungen: drei Regeln + Nicht-Normalitaets-Groessen
    for k in ("pred_mf", "pred_eig", "pred_sv", "tau_eig", "tau_sv",
              "eig_max", "sv_max"):
        assert k in crit, k

    comp = compare_spectrum(model, states)
    assert comp["lin_spectrum"].shape == comp["traj_spectrum"].shape
    assert -1.01 <= comp["spectrum_corr"] <= 1.01
