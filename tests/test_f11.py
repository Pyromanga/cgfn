"""Smoke-Tests fuer F11 (Benettin-QR-Messgeraet + Stabilitaets-Audit).

WICHTIG (Fix-Nachlese, 2026-10-13, Runden 1-3): Die urspruenglichen
Test-Praemissen waren physikalisch zu naiv und haben echte F11-Befunde
als "Fehler" getarnt:

1. W*8 => FTLE>0 war SVD-Denkweise. Benettin misst korrekt SUBKRITISCH:
   Riesen-W saettigt alle Units, sech^2(z)->0, Tangenten kontrahieren.
   Das ist die Edge-of-Chaos-Rueckkehr (F10) direkt am Messgeraet.

2. "Designed supercritical => FTLE ~ lam-1/tau" stimmt nur AUF der
   Trajektorie z(t)=0: Dort ist z exakt im (instabilen) Gleichgewicht,
   sech^2=1, Benettin misst exakt +0.25. JEDER Input treibt z in den
   gesaettigten Fixpunkt (z* ~ 1.3 bei lam=0.75, tau=2), dort ist die
   lokale Rate 0.75*sech^2(1.3) - 0.5 ~ -0.30 — die "Superkritikalitaet"
   verschwindet auf der getriebenen Trajektorie. Debug-Messungen:
   Langfenster je nach Input zwischen +0.006 und -0.24 — die Zeichen
   flippen mit dem Input (marginales Regime um den Fixpunkt).

Robust getestet wird jetzt:
  - UNGETRIEBEN (x=0): Trajektorie bleibt exakt bei z=0 => Benettin
    misst exakt lam - 1/tau = +0.25 (deterministisch, jede Fenstergroesse).
  - GETRIEBEN (langes Fenster): Sattigung deckelt den FTLE weit unter
    den linearen Wert — FTLE nahe der Fixpunkt-Rate ~ -0.3.
"""
import torch

from cgfn.benettin import benettin_ftle_max, benettin_spectrum, stability_audit
from cgfn.continuous import make_continuous_batch
from cgfn.design import make_designed_model
from cgfn.model import ContinuousRNN


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


def test_benettin_unforced_unstable_equilibrium_exact():
    """Designed normal (kappa=1), tau*lam_max = 1.5 > 1, x = 0:

    z(t)=0 ist exakter (instabiler) Fixpunkt — RK2 erhaelt ihn exakt
    (f(0)=0), die Tangenten sehen sech^2(0)=1 fuer immer. Benettin muss
    den LINEAREN Wert lam_max - 1/tau = +0.25 exakt messen, deterministisch
    und unabhaengig von Fenstergroesse/Input-Zufall. Das ist die
    Messgeraet-Kalibrierung am analytischen Referenzwert.
    """
    model = make_designed_model(8, tau=2.0, lam_max=0.75, s_nonnormal=0.0,
                                seed=0, task_dim=0)
    x = torch.zeros(2, 48, 1)
    f = benettin_ftle_max(model, x)
    assert abs(f - 0.25) < 0.05, f"erwartet +0.25, gemessen {f}"


def test_benettin_saturation_caps_driven_ftle():
    """DOKUMENTATIONS-TEST (Edge-of-Chaos-Kernbefund am Messgeraet):

    Dasselbe superkritische Netz (tau*lam=1.5>1), ABER mit Input getrieben:
    Der Input treibt z vom instabilen Gleichgewicht in den gesaettigten
    Fixpunkt (z* ~ 1.3, sech^2 ~ 0.27) — dort ist die lokale Rate
    0.75*sech^2(z*) - 1/tau ~ -0.30. Lange Fenster mitteln auf diese
    Fixpunkt-Rate: der FTLE liegt WEIT unter dem linearen Wert +0.25
    (den der ungetriebene Fall exakt misst, siehe Test darueber).
    Saettigung deckelt die Superkritikalitaet — Input entscheidet,
    ob die "kritische" Dynamik tatsaechlich chaotisch wird.
    """
    torch.manual_seed(0)
    model = make_designed_model(8, tau=2.0, lam_max=0.75, s_nonnormal=0.0,
                                seed=0, task_dim=0)
    x_unforced = torch.zeros(2, 96, 1)
    f_unforced = benettin_ftle_max(model, x_unforced)
    assert abs(f_unforced - 0.25) < 0.05

    x_driven, _ = make_continuous_batch("sinus_memory", 2, 96)
    f_driven = benettin_ftle_max(model, x_driven)
    # Saettigungs-Deckel: getrieben weit unter dem linearen Wert
    assert f_driven < f_unforced - 0.1, (
        f"getrieben {f_driven} sollte deutlich unter ungetrieben {f_unforced} liegen"
    )
    # und in der Naehe der negativen Fixpunkt-Rate (~ -0.3) statt +0.25:
    assert -0.6 < f_driven < 0.0, f"Fixpunkt-Rate ~-0.3 erwartet, gemessen {f_driven}"


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
