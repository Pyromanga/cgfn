

---

# Nachtrag 22 (2026-11-08): T16-REGISTRIERUNG — 2D-KARTE (H x s), ERSTER KERNEL-GENERIERTER STEIN (genagelt VOR dem Lauf)

## Warum T16 (und nicht P3/P4 zuerst)
Der W1-Kernel ist abgenommen (Nachtrag 21) und im Echtbetrieb UNGE-
TESTET. Der erste scharfe Schuss muss ein Physik-Stein mit bekanntem
Instrument sein (T15-Wortlaut), nicht eine neue Werkzeug-Ebene auf
ungetestetem Werkzeug. T16 ist der registrierte Folge-Stein aus
Nachtrag 17 ("2D-Karte H x s, NEU zu registrieren").

## T16-THESE (P-RAMPE-SHIFT-SUBH, die eine These)
Die Lernbarkeits-Rampe P(s|H) verschiebt sich mit H — die Rampen-Mitte
s50 waechst mit H —, ABER sub-proportional zu H:
ratio50 = s50(H=96)/s50(H=24) liegt strikt in (1, 4) (4 = H-Verhaeltnis).

## Architektur (Kernel-Anwendung, genagelt)
Contract: contracts/t16.py (reine Daten — Arme, Seeds, s-Raster,
Task-Konstanten, Invarianten, Sanity, Verdikt-Logik). Instrument:
experiments/t16_2d_karte.py (Konstruktion, Training, Selbst-Messung —
liest Arme/Seeds/Raster/Konstanten AUSSCHLIESSLICH aus dem Contract).
VERDICT generiert tools/lab_kernel.py (W1-P2). K-Pre-Flight vor dem
ersten Trainingsschritt (Runner-Minuten bleiben bei Verweigerung
unberuehrt). Neu nach A-Standard: Der Lauf MISST selbst Spaltennorm
(T11-Klasse), dmdev (DM-Konstanz innerhalb H, Gruppen-Relativ-
abweichung), ndiv, s50, ratio50.

ARME (14): REF24, REF96 (unnnormierte Konvention, SANITY je H) +
R24_s / R96_s fuer s in {1.5, 2.0, 2.8, 3.5, 4.4, 5.5} (REFGEOM,
exakt normiert). Seeds 160..169 (frisch). H in {24, 96}. Paritaet
wortgetreu wie T15 (Delayed-XOR, T=24, RK2, tau=1.0, dt=0.1,
substeps=5, Adam 600, Batch 32, lr 5e-3, 1024 Test-Samples, Test-Seed
777, FIXED_P0=4242, WB=0.6, LAM_MAX=0.6, b=0.1*randn fix).
P_lernt = Fraktion(acc >= 0.99).

s50-DEFINITION (genagelt): Erste 0.5-Durchquerung der nach s
sortierten P_lernt-Kurve der NORMIERTEN Arme eines H, linear
interpoliert zwischen Rasterpunkten. Divergenz-Arme (pf NaN) zaehlen
als nicht gelernt (0 in der Kurve). Nicht einklammerbar (pf am
unteren Rand >= 0.5 oder nirgends >= 0.5) -> s50 NICHT GEMESSEN ->
Kernel-Kollaps-Alarm -> UNENTSCHEIDEN (Aufloesung), EINE Reparatur
erlaubt, dann T16-Linie tot.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG — der Contract IST
diese Registrierung in ausfuehrbarer Form):
  SANITY: P_lernt(REF24) >= 0.5 UND P_lernt(REF96) >= 0.5.
  PASS (P-RAMPE-SHIFT-SUBH): 1.0 < ratio50 < 4.0 — Rampe verschiebt
    sich mit H, sub-proportional zu H.
  KILL-A (KEIN/INVERSER SHIFT): ratio50 <= 1.0 — die T15-Richtung
    (groessere Netze brauchen mehr Norm) ist auf Rampen-Ebene falsch.
  KILL-B (H-PROPORTIONAL): ratio50 >= 4.0 — schlimmster Fall: Rezepte
    muessen mit H skalieren; sqrt-artige Regeln reichen nicht aus.
  UNENTSCHEIDEN (Crash-Klasse): s50 nicht einklammerbar, Sanity
    gerissen, ndiv > 3 in einem Arm, dmdev-Verletzung (DM-Konstanz
    innerhalb H) oder colnorm-Verletzung (Normierung fehlgeschlagen).
    Rerun nur bei Crash; EINE Reparatur.
  SEKUNDAER: median acc, DM je H, s50/sqrt(H) je H (sqrt-Kollaps-
    Frage nur BERICHT, kein Verdikt — dafuer ist das Raster zu grob),
    Laufzeit, D-Diff-Snapshot (results/t16_measurements.json, ma-
    schinenlesbar) fuer kuenftige Reruns (erster Lauf ohne Vorgaenger).
  Bewahrung: Kein Arm wird nachjustiert; Rerun nur bei Crash.
  KOSTEN-SCHAETZUNG: ~45-55 Runner-Minuten (14 Arme x 10 Seeds).

## Ehrliche Risiken (vor dem Lauf genannt)
1. pf(1.5|H=24) koennte >= 0.5 sein (T15: pf(2.0|24)=0.70; T13:
   pf(1.0|48)=0.00) -> s50 nicht einklammerbar -> UNENTSCHEIDEN.
   Risiko akzeptiert; die EINE Reparatur waere ein versetztes/
   feineres Raster.
2. n=10 Seeds je Arm: pf-Rauschen +-0.15. Genau deshalb liegen die
   registrierten Schwellen weit auseinander (1.0 und 4.0).
3. 6-Punkt-Interpolation ist grob: ratio50 ist eine GROBE Schaetzung;
   registriert ist der weite Bereich (1, 4), keine genaue Zahl.
