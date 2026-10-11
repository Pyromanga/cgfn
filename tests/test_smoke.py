import torch

from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.ftle import ftle_summary
from cgfn.effdim import effective_dimension


def test_model_forward_shapes():
    """CT-RNN Forward-Pass liefert korrekte Output-/State-Shapes."""
    torch.manual_seed(0)
    model = ContinuousRNN(input_dim=1, task_dim=len(TASKS), hidden_dim=16, tau=1.5)
    x = torch.randint(0, 2, (4, 8, 1)).float()
    tv = torch.zeros(4, len(TASKS))
    tv[:, 0] = 1.0
    out, states = model(x, tv)
    assert out.shape == (4, 8, 1)
    assert states.shape == (4, 8, 16)


def test_tasks_shapes():
    """Alle Toy-Tasks liefern konsistente (B, T, 1)-Paare."""
    for task in TASKS:
        x, y = make_batch(task, batch_size=2, seq_len=8)
        assert x.shape == y.shape == (2, 8, 1), task


def test_training_reduces_loss():
    """Ein paar Adam-Schritte senken den Copy-Task-Loss (Smoke-Level)."""
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=1.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossfn = torch.nn.BCEWithLogitsLoss()
    x, y = make_batch("copy", batch_size=16, seq_len=8)
    tv = torch.zeros(16, len(TASKS))
    tv[:, TASKS.index("copy")] = 1.0
    first = None
    loss = None
    for _ in range(100):
        out, _ = model(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss is not None and loss.item() < first


def test_ftle_summary_runs():
    """FTLE-Zusammenfassung laesst sich aus einer Trajektorie berechnen."""
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=8, tau=1.5)
    x, _ = make_batch("parity", batch_size=4, seq_len=8)
    tv = torch.zeros(4, len(TASKS))
    tv[:, TASKS.index("parity")] = 1.0
    _, states = model(x, tv)
    summary = ftle_summary(model, states)
    assert "ftle_mean" in summary
    assert -100.0 < summary["ftle_mean"] < 100.0  # endlich & numerisch stabil


def test_effective_dimension_runs():
    """Effective-Dimension-Schaetzung liefert sinnvolle Werte."""
    torch.manual_seed(0)
    H = 8
    z = torch.randn(256, H)  # volles Rauschen -> d_eff nahe H
    d = effective_dimension(z, threshold=0.99)
    assert d >= 1
    z_collapse = torch.randn(256, 1).repeat(1, H)  # Rank-1 -> d_eff klein
    d_c = effective_dimension(z_collapse, threshold=0.99)
    assert d_c <= 2
