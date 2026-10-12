"""Tests fuer F2-Regler und F3-Plasticity-Metriken (Frontier-Module)."""
import copy

import torch

from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.regulator import FtleRegulator
from cgfn.plasticity import correlate, dormant_fraction, plasticity_snapshot, weight_churn


def _states(model, task="copy", seq_len=8):
    task_dim = len(TASKS)
    x, _ = make_batch(task, batch_size=8, seq_len=seq_len)
    tv = torch.zeros(8, task_dim)
    tv[:, TASKS.index(task)] = 1.0
    _, states = model(x, tv)
    return states


def test_regulator_updates_tau():
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=0.5)
    reg = FtleRegulator(target=0.0, gain=1.0, interval=1)
    tau0 = float(model.tau)
    for step in range(5):
        reg.step(step, model, _states(model))
    assert len(reg.history) == 5
    assert float(model.tau) != tau0  # Regler hat eingegriffen
    assert all(h["tau"] > 0 for h in reg.history)


def test_regulator_respects_bounds():
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=1.0)
    reg = FtleRegulator(target=0.0, gain=50.0, interval=1,
                        tau_min=0.05, tau_max=50.0)
    for step in range(10):
        reg.step(step, model, _states(model))
    assert 0.05 <= float(model.tau) <= 50.0


def test_regulator_ema_smoothing():
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=1.0)
    reg = FtleRegulator(target=0.0, gain=1.0, interval=1, smoothing=0.7)
    h = reg.update(0, model, _states(model))
    assert h["ftle_ema"] == h["ftle"]  # erster Schritt: EMA == Rohwert


def test_dormant_fraction_detects_constant_units():
    states = torch.zeros(4, 10, 8)
    states[..., 0] = torch.randn(4, 10)  # 1 von 8 Units aktiv
    frac = dormant_fraction(states)
    assert frac >= 0.7  # 7/8 Units schlafen


def test_weight_churn_zero_for_identical():
    torch.manual_seed(0)
    a = ContinuousRNN(hidden_dim=8)
    b = copy.deepcopy(a)
    assert weight_churn(a, b) == 0.0
    with torch.no_grad():
        b.W += 1.0
    assert weight_churn(a, b) > 0.0


def test_plasticity_snapshot_finite():
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=1.5)
    snap = plasticity_snapshot(model, _states(model))
    for key in ("ftle_mean", "dormant_fraction", "w_norm"):
        assert snap[key] == snap[key]  # not NaN
        assert abs(snap[key]) < 1e6


def test_correlate_perfect():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert abs(correlate(xs, [x * 2 for x in xs]) - 1.0) < 1e-9
    assert abs(correlate(xs, [-x for x in xs]) + 1.0) < 1e-9
