"""
F22 — DIFFERENZIERBARES BENETTIN: Der Homeostat-Regelkreis ohne
Block-Penalty — lambda als GLATTES, dauerhaftes Trainingsziel.

F21 nutzt eine Block-Penalty (alle 50 Steps messen, dann ein Update).
F22 geht weiter: der Benettin-FTLE wird mit einem KURZEN, fixen
QR-Fenster (K=8 Renorm-Schritte) als UNTERSCHEIDBARER Term in den
Loss genommen — jeder SGD-Schritt spuert den Kanten-Abstand, der
Regler wird kontinuierlich statt stoßweise.

Vorsicht vor der F14-Falle: der durchgehende Gradient des Benettin
durch das QR (differenzierbar via Householder-Implementierung oder
Umantel mit pytorch.autograd auf kleinen K) ist rechenintensiv —
deshalb K=8 klein, alle Schritte, und gegen die F21-Block-Version
als WETTKAMPF gemessen (nicht nur gegen baseline).

ARME (sinus_memory + lorenz, 4 Seeds, 900 Steps, wie F21):
  A baseline (frei)
  C1 Homeostat-Block (F21, alle 50 Steps)
  C2 Homeostat-Kontinuierlich (K=8 Benettin im Loss, jeder Schritt)
  D  reines Benettin-regularisiert OHNE Task (isoliert der Regler)

MESSUNGEN: loss-Kurven, MC, lambda-Trajektorie, KOSTEN (sec/step),
Transfer (sinus->lorenz->MC(sinus)).
"""
import argparse
import os
from pathlib import Path
import time

import torch

from cgfn.model import ContinuousRNN
from cgfn.benettin import benettin_ftle_max
from cgfn.memory_capacity import memory_capacity
from cgfn.continuous import make_continuous_batch
from cgfn.field import drive_input

BEN_TARGET = -0.3
PENALTY_W = 0.05
K_QR = 8


def benettin_short_diff(model, x_probe, K=K_QR):
    """Kurzes, differenzierbares Benettin-Fenster: QR auf der
    Jacobi des letzten Zustands, K Renorm-Schritte, mit autograd.
    Die QR selbst ist undifferenzierbar — Trick: wir berechnen Q,R
    im no_grad-Kontext und nutzen R-Diagonale als (konstante)
    FTLE-Schaetzung; der Gradient fliesst ueber die Zustaende."""
    raise NotImplementedError("F22-Platzhalter — wird implementiert, wenn F21 entscheidet")


def main():
    raise NotImplementedError("F22 wird implementiert, wenn F21-Entscheidung vorliegt")


if __name__ == "__main__":
    main()
