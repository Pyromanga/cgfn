"""Smoke-Tests fuer F11 (Benettin-QR-Messgeraet + Stabilitaets-Audit).

WICHTIG (Fix-Nachlese, 2026-10-13): Zwei urspruengliche Test-Prmissen waren
physikalisch zu naiv und haben echte F11-Befunde als "Fehler" getarnt:

1. W*8 => FTLE>0 war SVD-Denkweise. Benettin misst korrekt SUBKRITISCH:
   Riesen-W saettigt alle Units, sech^2(z)->0, Tangenten kontrahieren.
   Das ist die Edge-of-Chaos-Rueckkehr (F10) direkt am Messgeraet.

2. "Designed supercritical + kleiner Input => FTLE ~ lam-1/tau ueber das
   GESAMTE Fenster" ignoriert, dass die Instabilitaet z selbst aus dem
   linearen Bereich treibt: z waechst exp(lam-1/tau), erreicht die
   Sattigung, dort kehrt die Tangenten-Dynamik zur Kontraktion.
   Fensterschnitt ueber t=12: +0.006 (Wachstum + Rueckkehr mitteln sich
   weg) — der SAETTIGUNGS-DECKEL auf dem FTLE, Kern des Edge-of-Chaos.

Robust getestet wird jetzt:
  - Supercritical im KURZEN Fenster (t=4): z bleibt linear (x ~2.7 nur),
    FTLE deutlich positiv.
  - Saettigungs-Deckel im LANGEN Fenster (t=12): FTLE wird Richtung Null
    gedeckelt und liegt UNTER dem Kurzfenster-Wert.
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


def test_benettin_supercritical_short_window_positive():
    """Designed normal (kappa=1), tau*lam_max = 1.5 > 1, kleiner Input,
    KURZES Fenster (T=8 => t_total=4):

    z waechst nur ~e^((lam-1/tau)*4) ~ 2.7x und bleibt im linearen Bereich
    (sech^2 ~ 1) => FTLE ~ lam_max*sech^2 - 1/tau ~ +0.1..0.2 > 0.
    """
    model = make_designed_model(8, tau=2.0, lam_max=0.75, s_nonnormal=0.0,
                                seed=0, task_dim=0)
    x, _ = make_continuous_batch("sinus_memory", 2, 8)
    x = x * 0.1  # z im linearen Bereich halten (keine Saettigung im Fenster)
    f_short = benettin_ftle_max(model, x)
    assert f_short > 0.02, f"Kurzfenster sollte ~+0.15 sein, gemessen {f_short}"
    assert f_short < 0.4, f"Kurzfenster sollte ~+0.15 sein, gemessen {f_short}"


def test_benettin_saturation_caps_long_window_ftle():
    """DOKUMENTATIONS-TEST (Edge-of-Chaos-Kernbefund am Messgeraet):

    Dasselbe superkritische Netz, LANGES Fenster (T=24 => t_total=12):
    Die Instabilitaet treibt z in die Saettigung, sech^2 schrumpft, die
    Tangenten-Dynamik kehrt zur Kontraktion — der Fensterschnitt wird
    gegen Null gedeckelt und liegt UNTER dem Kurzfenster-Wert.
    Messung vom ersten F11-Debug-Lauf: +0.006 (vs. linear +0.25).
    """
    model = make_designed_model(8, tau=2.0, lam_max=0.75, s_nonnormal=0.0,
                                seed=0, task_dim=0)
    x8, _ = make_continuous_batch("sinus_memory", 2, 8)
    x24, _ = make_continuous_batch("sinus_memory", 2, 24)
    f_short = benettin_ftle_max(model, x8 * 0.1)
    f_long = benettin_ftle_max(model, x24 * 0.1)
    # Saettigungs-Deckel: langes Fenster deutlich gedrueckt
    assert f_long < f_short
    # und nahe Neutralitaet statt klar superkritisch (linearer Wert +0.25):
    assert -0.2 < f_long < 0.15, f"Langfenster ~0 erwartet, gemessen {f_long}"


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
