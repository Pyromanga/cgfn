"""
F14b-Tests: BenettinRegulator — Richtung, Determinismus, Grenzen.
"""
import torch

from cgfn.model import ContinuousRNN
from cgfn.regulator import BenettinRegulator
from cgfn.tasks import make_batch


def _model(tau=1.0, hidden=16):
    torch.manual_seed(0)
    return ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=hidden, tau=tau)


def test_ben_regulator_measurement_deterministic():
    m = _model()
    r1 = BenettinRegulator(target=-0.2)
    r2 = BenettinRegulator(target=-0.2)
    a = r1.measure_ftle(m)
    b = r2.measure_ftle(m)
    assert a == b  # feste Probe-Seed => identisch


def test_ben_regulator_direction():
    """Ziel weit UEBER dem Messwert => tau steigt; weit UNTER => tau faellt."""
    m_lo = _model(tau=1.0)
    r_lo = BenettinRegulator(target=+10.0, gain=1.0)  # Ziel unerreichbar hoch
    r_lo.update(0, m_lo)
    assert m_lo.tau > 1.0  # Regler will Expansion -> tau hoch

    m_hi = _model(tau=1.0)
    r_hi = BenettinRegulator(target=-10.0, gain=1.0)  # Ziel unerreichbar tief
    r_hi.update(0, m_hi)
    assert m_hi.tau < 1.0  # Regler will Kontraktion -> tau runter


def test_ben_regulator_respects_bounds():
    m = _model(tau=1.0)
    r = BenettinRegulator(target=+1e6, gain=100.0, tau_min=0.05, tau_max=50.0)
    for i in range(5):
        r.update(i, m)
    assert m.tau <= 50.0
    r2 = BenettinRegulator(target=-1e6, gain=100.0, tau_min=0.05, tau_max=50.0)
    m2 = _model(tau=1.0)
    for i in range(5):
        r2.update(i, m2)
    assert m2.tau >= 0.05


def test_ben_regulator_interval_gate():
    """step() regelt nur am Intervall — sonst no-op."""
    m = _model(tau=1.0)
    r = BenettinRegulator(target=+10.0, interval=3)
    assert r.step(0, m) is None
    assert r.step(1, m) is None
    assert r.step(2, m) is not None
    assert m.tau != 1.0


def test_f14b_mini_smoke():
    """Benettin-Regler + kappa-Cap im Mini-Trainingsloop."""
    m = _model(tau=1.0, hidden=8)
    from cgfn.kcap import kappa_cap_penalty
    reg = BenettinRegulator(target=-0.2, gain=0.8, interval=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for step in range(4):
        x, y = make_batch("copy", batch_size=4, seq_len=8, device="cpu")
        out, states = m(x, None)
        loss = lossfn(out, y) + 0.02 * kappa_cap_penalty(m, 4.0)
        opt.zero_grad(); loss.backward(); opt.step()
        reg.step(step, m, states)
    assert loss.item() == loss.item()  # finite
