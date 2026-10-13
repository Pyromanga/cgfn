"""
F15-Tests: Doom-Simulation — Richtung, Monotonie, Benettin-Stabilitaet.
"""
import torch

from cgfn.design import make_designed_model
from cgfn.regulator import FtleRegulator, BenettinRegulator


def test_svd_regulator_diverges_on_nonnormal_net():
    """kappa-hohes Netz + SVD-Regler => tau laeuft Richtung Clamp."""
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 2.0, 0, task_dim=0)
    m.tau = 1.0
    reg = FtleRegulator(target=-0.2, gain=0.8, interval=1)
    tau0 = float(m.tau)
    for i in range(30):
        reg.update(i, m, None)
    assert float(m.tau) < tau0 / 2.0  # starker Drift Richtung unten


def test_svd_regulator_mild_on_normal_net():
    """kappa~1 (s=0): Artefakt klein => kein Doom-Drift."""
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 0.0, 0, task_dim=0)
    m.tau = 1.0
    reg = FtleRegulator(target=-0.2, gain=0.8, interval=1)
    for i in range(30):
        reg.update(i, m, None)
    assert 0.3 < float(m.tau) < 3.0  # im Bereich geblieben


def test_benettin_regulator_bounded_everywhere():
    """Benettin-Regler auf kappa-hohem Netz: kein Doom."""
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 2.0, 0, task_dim=0)
    m.tau = 1.0
    reg = BenettinRegulator(target=-0.2, gain=0.8, interval=1)
    for i in range(30):
        reg.update(i, m, None)
    assert 0.1 < float(m.tau) < 10.0
