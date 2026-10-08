import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import make_batch
from cgfn.criticality import jacobian_spectral_radius


def test_forward_shape():
    m = ContinuousRNN()
    x, _ = make_batch("copy", batch_size=4, seq_len=8)
    out, states = m(x)
    assert out.shape == (4, 8, 1)
    assert states.shape == (4, 8, 32)


def test_spectral_radius_finite():
    m = ContinuousRNN()
    x, _ = make_batch("copy", batch_size=4, seq_len=8)
    with torch.no_grad():
        _, s = m(x)
    sr = jacobian_spectral_radius(m, s[:, -1])
    assert torch.isfinite(sr).all()