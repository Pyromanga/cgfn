"""F12-Smoke: Memory-Capacity-Messgeraet (Jaeger, Ridge-Readout)."""
import torch

from cgfn.design import make_designed_model
from cgfn.memory_capacity import memory_capacity
from cgfn.model import ContinuousRNN


def test_mc_finite_and_lag0_strong():
    """MC ist endlich; lag 0 (Input-Injektion) muss stark dekodierbar sein."""
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=0, hidden_dim=16, tau=1.0)
    mc, r2 = memory_capacity(model, max_lag=6, seq_len=48)
    assert mc >= 0.0 and torch.isfinite(torch.tensor(mc))
    assert r2[0] > 0.5, f"lag-0 corr^2 sollte >0.5 sein (Input-Injektion), ist {r2[0]}"


def test_mc_edge_net_beats_tiny_net():
    """Ressourcen-Ordnung: Netz nahe der Kante speichert mehr als Tiny-W."""
    torch.manual_seed(0)
    tiny = ContinuousRNN(task_dim=0, hidden_dim=16, tau=1.0)
    with torch.no_grad():
        tiny.W.copy_(tiny.W * 0.01)
    edge = make_designed_model(16, 1.0, 0.95, 0.0, seed=0, task_dim=0)
    mc_tiny, _ = memory_capacity(tiny, max_lag=8, seq_len=48)
    mc_edge, _ = memory_capacity(edge, max_lag=8, seq_len=48)
    assert mc_edge > mc_tiny, f"Edge-Netz ({mc_edge:.2f}) sollte Tiny-W ({mc_tiny:.2f}) schlagen"


def test_mc_deterministic_for_same_seed():
    """MC ist deterministisch (eigener Generator): zwei Laeufe identisch."""
    torch.manual_seed(1)
    model = ContinuousRNN(task_dim=0, hidden_dim=12, tau=1.5)
    mc1, _ = memory_capacity(model, max_lag=4, seq_len=32)
    mc2, _ = memory_capacity(model, max_lag=4, seq_len=32)
    assert abs(mc1 - mc2) < 1e-9
