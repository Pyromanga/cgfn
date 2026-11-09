

---

# Nachtrag 26 (2026-11-09): EXPONENTEN-ANALYSE (ehrlich: KEIN exaktes Pinning aus 3 Punkten) + T18-REGISTRIERUNG — Theory-Core-Vorhersage-Stein (genagelt VOR dem Lauf)

## Exponenten-Analyse (Antwort auf die Kapitaens-Frage)
Der 1%-Fit s50 ~ 0.546 * H^0.4083 ist real, ABER: Die Raster-
unsicherheit (n=10, pf=k/10, s50 +- ~0.25) blaest den Exponenten auf
das Band [0.382, 0.433]. Darin leben ALLE Kandidaten:
  2/5 = 0.400 | ln(3/2) = 0.405 | 1/sqrt(6) = 0.408 (0.02% Abweichung
  — VERDAECHTIG exakt, aber Numerologie-Gefahr: 3 Punkte pinnen keine
  Konstante) | 5/12 = 0.417. KEIN Kandidat ist ausschliessbar; <1%-
  Abweichung beweist hier KEINE genaue Formel, nur eine gute
  Naeherung. Schritt-Exponenten (0.404 / 0.418) zeigen keinen Drift.
ROBUST ueber alle Kandidaten: die Vorhersage s50(12) in [1.47, 1.52]
(e=0.40: 1.523 | e=0.408: 1.498 | e=5/12: 1.471).
PINNING-PLAN (registriert): Wenn T18 PASS -> T19 als Pinning-Stein
(H in {12, 48} mit feinem Raster und 20 Seeds an den Klammer-
punkten; Ziel: Exponenten-Band +-0.01) ODER analytische Herleitung
(Warum H^alpha mit alpha ~ 0.41? Kandidat-Mechanismus: DM/Spektrum-
Kopplung der T6-B60-Familie). VERALLGEMEINERUNG (offen, eigene
Steine): Das Gesetz ist bisher an EINER Familie gemessen (Paritaet,
T6-B60-W, REFGEOM); allgemein gueltig heisst: Familie variieren.

## T18 — THE-CORE-VORHERSAGE-TEST (P-GESCHLOSSENE-VORHERSAGE)
ERSTER VORHERSAGE-STEIN der Serie: Das gefittete Weltmodell (Nachtrag
25) sagt eine Zahl voraus; der Kernel verdiktet sie. Das ist der
geschlossene Kreislauf in Miniatur und der erste messbare
Research-Taste-Datenpunkt (Nutzer-These, Nachtrag 24).

VORHERSAGE (aus s50 ~ 0.546 * H^0.408, Nachtrag 25):
  s50(H=12) ~ 1.51 (Kandidaten-Band [1.47, 1.52]).

INSTRUMENT (kernel-generiert wie T16/T17): Contract contracts/t18.py,
Instrument experiments/t18_core_vorhersage.py, Verdikt generiert.
H=12 (neu, billigste Breite), Seeds 180..189, Paritaet wie T15/T16/
T17. ARME (7): REF12 (unnnormierte Konvention, role ref — Bericht,
KEIN Sanity: 0.5*sqrt(12)=1.73 liegt VORHERGESAGT auf der Rampe —
T3-Lehre: kein Sanity-Arm auf einem marginalen Punkt), HIGH12
(REFGEOM exakt Norm 3.5, role anchor — SANITY: weit oberhalb der
Vorhersage-Rampe, dort muss das Instrument lernen), R12_s fuer s in
{0.8, 1.1, 1.5, 1.9, 2.4} (role sgrid, REFGEOM exakt normiert).

s50-Definition wie T16/T17 genagelt (erste 0.5-Durchquerung der
sgrid-Kurve, linear interpoliert; nicht einklammerbar -> NICHT
GEMESSEN -> Kernel-Alarm -> UNENTSCHEIDEN, EINE Reparatur).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(HIGH12) >= 0.5.
  PASS (P-GESCHLOSSENE-VORHERSAGE): s50(12) in [1.1, 1.9]
    (Vorhersage 1.51, Band +-27 Prozent — grosszuegig gegen n=10-
    Rauschen, scharf genug, um einen kaputten Fit zu toeten).
  KILL-A (VORHERSAGE ZU HOCH): s50(12) < 1.1 — das Netz lernt frueher
    als das Gesetz sagt; Fit war Zufall oder H^alpha falsch geformt.
  KILL-B (VORHERSAGE ZU NIEDRIG): s50(12) >= 1.9 — das Netz lernt
    spaeter; dito.
  UNENTSCHEIDEN (Crash): s50 nicht einklammerbar (Raster verfehlt),
    Sanity gerissen, ndiv > 3, dmdev-/colnorm-Verletzung. Rerun nur
    bei Crash; EINE Reparatur, dann T18-Linie tot.
  SEKUNDAER: REF12-pf (Konventions-Bericht), DM, Laufzeit,
    D-Diff-Snapshot, implizierter Exponent aus 4 Punkten (Bericht).
  Bewahrung: kein Arm wird nachjustiert.
  KOSTEN-SCHAETZUNG: ~10-15 Runner-Minuten (H=12 ist schnell).
  RISIKEN: pf(0.8) >= 0.5 -> s50 < 0.8 -> UNENTSCHEIDEN (haette
  KILL-A-Richtung); pf(2.4) < 0.5 -> UNENTSCHEIDEN (KILL-B-Richtung).
