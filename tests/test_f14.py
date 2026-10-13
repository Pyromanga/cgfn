"""
F14-Tests: kappa-Cap-Penalty (differentierbar, korrekt) + Mini-Smoke.
"""
import torch

from cgfn.model import ContinuousRNN
from cgfn.design import make_designed_model
from cgfn.kcap import kappa_of, kappa_cap_penalty
from cgfn.regulator import FtleRegulator
from cgfn.tasks import make_batch


def test_kappa_normal_design_is_one():
    """s=0 (Schur ohne obere Stoerung) => W symmetrisch => kappa exakt 1."""
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 0.0, 0, task_dim=0)
    assert abs(kappa_of(m) - 1.0) < 1e-4


def test_kappa_cap_penalty_zero_when_target_huge():
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 2.0, 0, task_dim=0)
    pen = kappa_cap_penalty(m, kappa_target=100.0)
    assert float(pen) == 0.0


def test_kappa_cap_penalty_positive_with_grad_for_nonnormal():
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 2.0, 0, task_dim=0)
    m.W.requires_grad_(True)  # designed friert W — fuer den Penalty-Test auftauen
    pen = kappa_cap_penalty(m, kappa_target=1.5)
    assert float(pen) > 0.0
    pen.backward()
    assert m.W.grad is not None and m.W.grad.abs().sum().item() > 0.0


def test_penalty_pulls_kappa_down():
    """Ein paar Gradienten-Schritte NUR auf die Penalty senken kappa."""
    torch.manual_seed(0)
    m = make_designed_model(16, 1.0, 0.6, 2.0, 0, task_dim=0)
    m.W.requires_grad_(True)
    k0 = kappa_of(m)
    assert k0 > 2.5  # Start klar ueber dem Cap (Vorbedingung, s.o.)
    opt = torch.optim.Adam([m.W], lr=0.1)
    loss = None
    for _ in range(400):
        opt.zero_grad()
        loss = kappa_cap_penalty(m, kappa_target=2.0)
        if float(loss.detach()) < 1e-4:
            break  # Cap erreicht — Penalty praktisch null
        loss.backward()
        opt.step()
    k1 = kappa_of(m)
    assert k1 < k0, f"kappa nicht gesunken: {k0} -> {k1}"
    assert k1 <= 2.0 + 0.5, f"Cap nicht erreicht: kappa={k1}, penalty={float(loss.detach())}"


def test_f14_mini_smoke():
    """Kleiner Trainings-Loop mit Penalty + Regler wirft keine Exception."""
    torch.manual_seed(0)
    m = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=8, tau=1.0)
    reg = FtleRegulator(target=-0.2, gain=0.8, interval=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for step in range(4):
        x, y = make_batch("copy", batch_size=4, seq_len=8, device="cpu")
        out, states = m(x, None)
        loss = lossfn(out, y) + 0.02 * kappa_cap_penalty(m, 4.0)
        opt.zero_grad(); loss.backward(); opt.step()
        reg.step(step, m, states)
    assert kappa_of(m) > 0 and loss.item() == loss.item()  # finite
