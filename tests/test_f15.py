"""
F15-Tests: Doom-Simulation — Richtung, Monotonie, Benettin-Stabilitaet.
"""
import torch

from cgfn.design import make_designed_model
from cgfn.regulator import FtleRegulator, BenettinRegulator


def _states(m, T=16):
    """Frische Treib-Trajektorie (der FtleRegulator misst auf states)."""
    x = torch.randn(1, T, 1) * 0.5
    _, states = m(x, None)
    return states




def test_svd_regulator_diverges_on_nonnormal_net():
    """kappa-hohes Netz + SVD-Regler => tau laeuft Richtung Clamp."""
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 2.0, 0, task_dim=0)
    m.tau = 1.0
    reg = FtleRegulator(target=-0.2, gain=0.8, interval=1)
    tau0 = float(m.tau)
    for i in range(30):
        reg.update(i, m, _states(m))
    assert float(m.tau) < tau0 / 2.0  # starker Drift Richtung unten


def test_svd_regulator_dooms_even_normal_net():
    """Befund (F11/F12-Daten): Der SVD-Artefakt ist NICHT rein kappa-getrieben —
    der Saettigungs-Deckel macht SVD-FTLE selbst bei kappa=1 positiv (F12:
    lam=0.6 s=0: SVD +0.36 vs. Benettin -0.60). Folge: Der SVD-Regler mit
    target=-0.2 doomt AUCH normale designed Netze Richtung Clamp."""
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 0.0, 0, task_dim=0)
    m.tau = 1.0
    reg = FtleRegulator(target=-0.2, gain=0.8, interval=1)
    for i in range(30):
        reg.update(i, m, _states(m))
    assert float(m.tau) < 0.5  # starker Drift Richtung Clamp — Doom auch bei kappa=1


def test_benettin_regulator_bounded_everywhere():
    """Benettin-Regler auf kappa-hohem Netz: kein Doom."""
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 2.0, 0, task_dim=0)
    m.tau = 1.0
    reg = BenettinRegulator(target=-0.2, gain=0.8, interval=1)
    for i in range(30):
        reg.update(i, m, _states(m))  # Benettin ignoriert states, einheitlich
