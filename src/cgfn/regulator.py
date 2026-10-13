"""
F2 — FTLE-Feedback-Regler: kausaler Kritikalitäts-Zwang während des Trainings.

Idee (ROADMAP F2): FTLE ~ 0 (Edge of Chaos) korreliert mit gutem Lernen
(corr(FTLE, loss) ~ -0.41, s. frontier3_results.txt). Der Regler macht daraus
einen KAUSALEN Test: er misst FTLE während des Trainings online und regelt
model.tau nach, sodass FTLE_mean gegen ein Ziel (default 0.0) laeuft.

Update-Regel (Vorzeichen-Logik):
  - groesseres tau => langsamerer Zerfall -z/tau => mehr Expansion => groesseres FTLE.
  - err = ftle_ema - target; tau <- tau * exp(-gain * err), geclampt auf [tau_min, tau_max].
  D.h. zu chaotisch (err > 0) => tau runter; zu kontraktiv (err < 0) => tau rauf.

Der Regler ist bewusst simpel (P-Regler in log-tau): robust, interpretierbar,
keine Hyperparameter-Kaskade. smoothing (EMA) verhindert Rausschwingen auf
Batch-Rauschen.
"""
import math

import torch

from .ftle import ftle_summary


class FtleRegulator:
    """Regelt model.tau online, sodass FTLE_mean gegen `target` laeuft.

    Nutzung im Trainingsloop:
        reg = FtleRegulator(target=0.0, interval=100)
        for step, batch in enumerate(loader):
            out, states = model(x, tv)
            reg.step(step, model, states)   # no-op ausser alle `interval` Steps
    """

    def __init__(self, target=0.0, gain=1.0, interval=100,
                 tau_min=0.05, tau_max=50.0, smoothing=0.7):
        self.target = float(target)
        self.gain = float(gain)
        self.interval = int(interval)
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        self.smoothing = float(smoothing)
        self._ema = None
        self.history = []  # Dicts: step, ftle, ftle_ema, tau_prev, tau

    @torch.no_grad()
    def measure_ftle(self, model, states):
        """FTLE-Mittelwert einer frischen Trajektorie (B, T, H)."""
        return float(ftle_summary(model, states)["ftle_mean"])

    @torch.no_grad()
    def update(self, step, model, states):
        """Ein Regel-Update: misst FTLE und stellt tau nach. Rueckgab: Dict|None."""
        raw = self.measure_ftle(model, states)
        if self._ema is None:
            self._ema = raw
        else:
            s = self.smoothing
            self._ema = s * self._ema + (1.0 - s) * raw
        err = self._ema - self.target
        tau_prev = float(model.tau)
        tau_new = tau_prev * math.exp(-self.gain * err)
        tau_new = min(self.tau_max, max(self.tau_min, tau_new))
        model.tau = tau_new
        entry = {
            "step": int(step),
            "ftle": raw,
            "ftle_ema": self._ema,
            "tau_prev": tau_prev,
            "tau": tau_new,
        }
        self.history.append(entry)
        return entry

    @torch.no_grad()
    def step(self, step, model, states):
        """Aufrufen nach JEDEM Trainingsstep; regelt nur alle `interval` Steps."""
        if self.interval <= 0 or (step + 1) % self.interval != 0:
            return None
        return self.update(step, model, states)

    def reset(self):
        """EMA- und Verlaufs-Zuruecksetzung (z.B. zwischen Tasks)."""
        self._ema = None
        self.history = []


# ─────────────────────────────────────────────
# F14b — Benettin-basierter Regler (artefaktfrei)
# ─────────────────────────────────────────────

class BenettinRegulator:
    """Wie FtleRegulator, aber das Messgeraet im Regelkreis ist Benettin-QR.

    F14-Befund: Der SVD-basierte Regler hetzte tau auf den Clamp, weil
    das SVD-Kurzfenster bei kappa>>1 Expansion vortaeuscht (F11/F12/F13:
    corr(kappa, Messdifferenz) > 0) — ein artefaktbehaftetes Messgeraet
    im Reglerkreis sabotiert das Netz, das es pinnen soll. Diese Klasse
    ersetzt die Messung durch das Standard-Messgeraet (Benettin-QR auf
    einer kurzen deterministischen Treib-Probe, dieselben Input-
    Statistiken wie im F12/F14-Messvertrag: weisses Rauschen, scale 0.5).

    Update-Regel identisch zu FtleRegulator (P-Regler in log-tau):
      err = ftle_ben_ema - target; tau <- tau * exp(-gain * err).
    """
    def __init__(self, target=0.0, gain=1.0, interval=100,
                 tau_min=0.05, tau_max=50.0, smoothing=0.7,
                 probe_len=32, probe_batch=2, input_scale=0.5):
        self.target = float(target)
        self.gain = float(gain)
        self.interval = int(interval)
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        self.smoothing = float(smoothing)
        self.probe_len, self.probe_batch = int(probe_len), int(probe_batch)
        self.input_scale = float(input_scale)
        self._ema = None
        self.history = []

    def measure_ftle(self, model, states=None):
        """Benettin-FTLE auf deterministischer Treib-Probe (states unbenutzt,
        Signatur-kompatibel zu FtleRegulator.step-Aufrufen)."""
        from .benettin import benettin_ftle_max
        gen = torch.Generator().manual_seed(20261013)
        x = torch.randn(self.probe_batch, self.probe_len, 1,
                        generator=gen) * self.input_scale
        return benettin_ftle_max(model, x)

    @torch.no_grad()
    def update(self, step, model, states=None):
        import math
        raw = self.measure_ftle(model, states)
        if self._ema is None:
            self._ema = raw
        else:
            s = self.smoothing
            self._ema = s * self._ema + (1.0 - s) * raw
        err = self._ema - self.target
        tau_prev = float(model.tau)
        tau_new = tau_prev * math.exp(-self.gain * err)
        tau_new = min(self.tau_max, max(self.tau_min, tau_new))
        model.tau = tau_new
        entry = {"step": int(step), "ftle": raw, "ftle_ema": self._ema,
                 "tau_prev": tau_prev, "tau": tau_new}
        self.history.append(entry)
        return entry

    @torch.no_grad()
    def step(self, step, model, states=None):
        if self.interval <= 0 or (step + 1) % self.interval != 0:
            return None
        return self.update(step, model, states)

    def reset(self):
        self._ema = None
        self.history = []
