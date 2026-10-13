"""Smoke-Tests für F11 (Benettin-QR-Messgeraet + Stabilitaets-Audit).

WICHTIG (2026-10-13, Fix nach erstem F11-Run): Der urspruengliche
test_benettin_unstable_net_positive skalierte ein Random-W mit x8 und
erwartete FTLE > 0. Das ist die SVD-Logik ("grosses sv_max => Chaos")
auf ein Messgeraet uebertragen, das ehrlicher ist: Riesen-W saettigt
ALLE Units (tanh -> +-1), damit wird sech^2(z) -> 0, die Jacobi-Diagonale
verschwindet und die Tangenten-Dynamik kontrahiert mit -1/tau —
FTLE korrekt negativ (gemessen: -0.27 bei tau=2). Benettin zeichnet
hier genau den Edge-of-Chaos-Befund von F10 nach: jenseits der Kante
kehrtdie Dynamik in die Subkritikalitaet zurueck. Der Test pruefte also
das falsche Regime.

Statt dessen: designedes NORMALES Netz (kappa=1, s=0) mit
lam_max = 0.75, tau = 2.0 => Eigenwert-Regel tau*lam = 1.5 > 1
(supercritical), Input skaliert auf 10%, damit z im linearen Bereich
bleibt (sech^2 ~ 1). Erwartung: FTLE ~ lam_max - 1/tau = +0.25.
"""
import torch

from cgfn.benettin import benettin_ftle_max, benettin_spectrum, stability_audit
from cgfn.continuous import make_continuous_batch
from cgfn.design import make_designed_model
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch


def test_benettin_spectrum_finite_and_sorted():
    model = ContinuousRNN(task_dim=0, hidden_dim=8, tau=1.5)
    x, _ = make_continuous_batch("sinus_memory", 2, 12)
    spec, t_total = benettin_spectrum(model, x)
    assert spec.shape == (8,)
    assert torch.isfinite(spec).all()
    assert spec[0] >= spec[-1] - 1e-6  # absteigend sortiert
    assert t_total > 0


def test_benettin_decaying_net_negative():
    """Kleine Kopplung, tau gross => alles kontrahiert => FTLE < 0."""
    model = ContinuousRNN(task_dim=0, hidden_dim=8, tau=2.0)
    with torch.no_grad():
        model.W.copy_(model.W * 0.01)
    x, _ = make_continuous_batch("sinus_memory", 2, 24)
    assert benettin_ftle_max(model, x) < 0.0


def test_benettin_saturated_net_returns_to_subcritical():
    """Riesen-W (x8) saettigt alle Units: sech^2 -> 0, Tangenten kontrahieren.

    Benettin-FTLE muss NEGATIV sein (Edge-of-Chaos-Rueckkehr, s. Modul-
    Kommentar) — aber WEIT UEBER dem strikt kontrahierenden Tiny-W-Netz
    (dessen FTLE ~ -1/tau): das transiente Wachstum vor der Saettigung
    hebt den Fenstermittelwert deutlich an.
    """
    model = ContinuousRNN(task_dim=0, hidden_dim=8, tau=2.0)
    with torch.no_grad():
        model.W.copy_(model.W * 8.0)
    x, _ = make_continuous_batch("sinus_memory", 2, 24)
    f_saturated = benettin_ftle_max(model, x)
    assert f_saturated < 0.0

    tiny = ContinuousRNN(task_dim=0, hidden_dim=8, tau=2.0)
    with torch.no_grad():
        tiny.W.copy_(tiny.W * 0.01)
    f_tiny = benettin_ftle_max(tiny, x)
    # transientes Wachstum vor der Saettigung sichtbar:
    assert f_saturated > f_tiny


def test_benettin_supercritical_linear_regime_positive():
    """Designed normal (kappa=1), tau*lam_max = 1.5 > 1, kleiner Input:

    z bleibt klein (sech^2 ~ 1) => FTLE ~ lam_max - 1/tau = +0.25 > 0.
    Deterministische Konstruktion statt Random-W-Skalierung.
    """
    model = make_designed_model(8, tau=2.0, lam_max=0.75, s_nonnormal=0.0,
                                seed=0, task_dim=0)
    x, _ = make_continuous_batch("sinus_memory", 2, 24)
    x = x * 0.1  # z im linearen Bereich halten (keine Saettigung)
    f = benettin_ftle_max(model, x)
    assert f > 0.1, f"erwartet ~ +0.25, gemessen {f}"
    assert f < 0.5, f"erwartet ~ +0.25, gemessen {f}"


def test_stability_audit_flags():
    model = ContinuousRNN(task_dim=0, hidden_dim=8, tau=1.0)
    a = stability_audit(model)
    assert 0.0 < a["svJ0_times_dt"]
    assert isinstance(a["rk2_ok"], bool)
    # extrem grosses W muss failen
    with torch.no_grad():
        model.W.copy_(model.W * 30.0)
    b = stability_audit(model)
    assert not b["rk2_ok"]


def test_benettin_designed_normal_net():
    """kappa=1 designed: Benettin muss endliche Werte liefern."""
    model = make_designed_model(8, 1.0, 1.15, 0.0, seed=0, task_dim=0)
    x, _ = make_continuous_batch("sinus_memory", 2, 16)
    f = benettin_ftle_max(model, x)
    assert -50.0 < f < 50.0
