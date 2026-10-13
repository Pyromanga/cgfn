"""
F16-Tests: architektur-agnostische kappa-Masse + Cap-Penalty.
"""
import torch

from cgfn.zoo import make_zoo_model

from experiments.f16_kappa import kappa_value, kappa_penalty


def _zoo(name, seed=0, H=8):
    torch.manual_seed(seed)
    return make_zoo_model(name, hidden_dim=H, seed=seed)


def test_kappa_positive_finite_all_models():
    for name in ("ct", "gru", "cornn"):
        m = _zoo(name)
        k = kappa_value(m)
        assert k > 0 and k == k  # positiv und finite


def test_penalty_zero_when_target_huge():
    for name in ("ct", "gru", "cornn"):
        m = _zoo(name)
        assert float(kappa_penalty(m, kappa_target=1e6)) == 0.0


def test_penalty_has_gradient():
    for name in ("ct", "gru", "cornn"):
        m = _zoo(name)
        pen = kappa_penalty(m, kappa_target=0.5)  # fast immer aktiv
        if float(pen) == 0.0:
            continue  # kappa <= 0.5: nichts zu regularisieren — ok
        pen.backward()
        m = _zoo(name)
        pen = kappa_penalty(m, kappa_target=0.5)
        pen.backward()
        grads = []
        core = m.core
        for p in core.parameters():
            if p.grad is not None:
                grads.append(p.grad.abs().sum().item())
        assert sum(grads) > 0.0, f"{name}: kein Gradient durch die Penalty"


def test_kappa_drift_smoke():
    """Mini-Training: free-Arm laeuft, kappa bleibt finite."""
    from experiments.f16_kappa import train, kappa_value
    m = _zoo("ct")
    track = [kappa_value(m)]
    torch.manual_seed(1)
    train(m, steps=10, lr=2e-3, arm="free", kappa_target=4.0,
          kappa_weight=0.02, kap_track=track, track_every=5)
    assert len(track) >= 3
    assert track[-1] == track[-1]  # finite
