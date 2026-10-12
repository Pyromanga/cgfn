"""F7-Smoke: Spectral-Ruler-Maße (spectral_edge, eig_max, nonnormality, 3 Regeln)."""
import torch

from cgfn.cavity import chaos_criterion, eig_max, nonnormality, spectral_edge
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def test_spectral_measures():
    torch.manual_seed(0)
    model = ContinuousRNN(task_dim=len(TASKS), hidden_dim=16, tau=1.5)
    idx = TASKS.index("parity")
    x, _ = make_batch("parity", batch_size=4, seq_len=8)
    tv = torch.zeros(4, len(TASKS))
    tv[:, idx] = 1.0
    _, states = model(x, tv)

    sv = spectral_edge(model)
    lam = eig_max(model)
    kappa, sv2, lam2 = nonnormality(model)
    assert sv == sv2
    # nonnormality() liefert den SPEKTRALRADIUS max|lambda| (damit
    # kappa = sv_max/max|lambda| >= 1 garantiert ist), eig_max() den
    # groessten REALTEIL. Beides ist nur identisch, wenn der dominante
    # Eigenwert reell-positiv ist — hier nur Ordnungsrelation pruefen:
    assert lam2 >= lam - 1e-6  # Spektralradius >= max Re lambda, immer
    assert sv > 0 and lam > 0
    assert kappa >= 1.0 - 1e-6  # sv_max >= Spektralradius immer

    crit = chaos_criterion(model, states)
    for key in ("pred_mf", "pred_eig", "pred_sv", "tau_g_sqrtQ",
                "eig_max", "sv_max", "chaos_predicted"):
        assert key in crit
    assert isinstance(crit["pred_mf"], bool)
    assert isinstance(crit["pred_eig"], bool)
    assert isinstance(crit["pred_sv"], bool)


def test_eig_rule_detects_scaled_chaos():
    """g_scale hoch => Eigenwert-Regel muss Chaos vorhersagen (grosses W)."""
    torch.manual_seed(0)
    base = ContinuousRNN(task_dim=len(TASKS), hidden_dim=32, tau=1.0)
    with torch.no_grad():
        base.W.copy_(base.W * 5.0)  # sehr grosse Kopplung
    assert eig_max(base) * base.tau > 1.0  # Regel sagt Chaos
    with torch.no_grad():
        base.W.copy_(base.W * 0.001)  # sehr kleine Kopplung
    assert eig_max(base) * base.tau < 1.0  # Regel sagt kein Chaos
