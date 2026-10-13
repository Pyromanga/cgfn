"""F13-Smoke: Zoo-Adapter (GRU/coRNN/CT) + getriebene Lyapunov-Messung."""
import torch

from cgfn.driven_lyap import _jacobian_step, driven_measures
from cgfn.memory_capacity import memory_capacity
from cgfn.zoo import make_zoo_model


def test_gru_step_matches_torch_cell():
    """Handgeschriebene GRU == torch.nn.GRUCell (Gleichungs-Validitaet)."""
    torch.manual_seed(0)
    m = make_zoo_model("gru", hidden_dim=8, seed=0)
    ref = torch.nn.GRUCell(1, 8)
    with torch.no_grad():
        ref.weight_ih.copy_(m.core.weight_ih)
        ref.weight_hh.copy_(m.core.weight_hh)
        ref.bias_ih.copy_(m.core.bias)   # gleiche Bias-Behandlung:
        ref.bias_hh.zero_()              # alles in bias_ih, hh-Bias null
        # ACHTUNG Reihenfolge: torch nutzt (r, z, n) — wie wir.
    x = torch.randn(1)
    h = torch.randn(8)
    ours = m.core.step(h, x)
    theirs = ref(x.unsqueeze(0), h.unsqueeze(0))[0]
    assert torch.allclose(ours, theirs, atol=1e-6), (ours - theirs).abs().max()


def test_step_jacobian_matches_finite_difference():
    """jacrev-Jacobi == Finite-Differenzen (Messgeraet-Kalibrierung)."""
    torch.manual_seed(0)
    m = make_zoo_model("ct", hidden_dim=6, seed=0)
    z = torch.randn(6) * 0.1
    x = torch.randn(1) * 0.5
    J = _jacobian_step(m.step, z, x)
    eps = 1e-4
    J_fd = torch.zeros(6, 6)
    for i in range(6):
        e = torch.zeros(6); e[i] = eps
        J_fd[:, i] = (m.step(z + e, x) - m.step(z - e, x)) / (2 * eps)
    assert torch.allclose(J, J_fd, atol=1e-3), (J - J_fd).abs().max()


def test_driven_lyap_signs_ct():
    """CT-Anker unter dem ALLGEMEINEN Messgeraet: Tiny-W subkritisch."""
    torch.manual_seed(0)
    m = make_zoo_model("ct", hidden_dim=8, seed=0)
    with torch.no_grad():
        m.core.W.mul_(0.01)
    x = torch.randn(2, 16, 1) * 0.5
    res = driven_measures(m, x)
    assert res["lyap_ben"] < 0.0
    assert res["lyap_svd"] < 0.0


def test_driven_lyap_runs_on_all_zoo_models():
    for name in ("gru", "cornn", "ct"):
        torch.manual_seed(0)
        m = make_zoo_model(name, hidden_dim=8, seed=0)
        x = torch.randn(2, 8, 1) * 0.5
        res = driven_measures(m, x)
        for k in ("lyap_ben", "lyap_svd", "kappa_traj"):
            assert torch.isfinite(torch.tensor(res[k])), (name, k, res)


def test_design_measures_and_mc_finite():
    for name in ("gru", "cornn", "ct"):
        torch.manual_seed(0)
        m = make_zoo_model(name, hidden_dim=8, seed=0)
        d = m.design()
        assert d["kappa"] >= 1.0 - 1e-6
        assert isinstance(d["design_ok"], bool)
        mc, r2 = memory_capacity(m, max_lag=4, seq_len=32)
        assert mc >= 0.0 and r2[0] >= 0.0
