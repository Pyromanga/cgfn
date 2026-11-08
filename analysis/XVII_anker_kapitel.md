

---

# Nachtrag 23 (2026-11-08): T16 — PASS (P-RAMPE-SHIFT-SUBH), erster kernel-generierter Stein. Die Rampe verschiebt sich mit H, sub-proportional zu H. Und: Der Kernel im Echtbetrieb — sauber.

## T16 (Lauf 22:50-23:23 UTC, ~33 min, Bot-Commit 731933f, Seeds 160..169, 14 Arme, 0 Divergenzen)
VERDICT (GENERIERT vom Kernel): PASS — P-RAMPE-SHIFT-SUBH.
  ratio50 = s50(96)/s50(24) = 3.5/2.0 = 1.75 — strikt in (1, 4).
  RAMPE H=24: pf = 0.20 / 0.50 / 0.80 / 1.00 / 1.00 / 0.90
       (s = 1.5 / 2.0 / 2.8 / 3.5 / 4.4 / 5.5)
  RAMPE H=96: pf = 0.00 / 0.10 / 0.30 / 0.50 / 0.80 / 0.90
  Beide Kurven erstmals der Serie: SAUBER MONOTON (bis auf n=10-
  Rauschen an der Saettigung), keine Kante, kein Kliff — die
  T15-Rampen-Lesart ist voll bestaetigt.

## Lesart (handgeschrieben, wie es das N-Modul verlangt)
1. DIE EINE THESE HAELT: Die Rampe verschiebt sich mit H
   (2.0 -> 3.5), sub-proportional zu H (1.75 << 4). KILL-B
   (H-proportional) ist weit weg; KILL-A (kein Shift) auch.
2. SPAERER BEFUND (nur Material, nicht registriert): ratio50 = 1.75
   liegt sogar UNTER sqrt(4) = 2 — s50/sqrt(H) FAELLT mit H
   (0.408 -> 0.357). Vorsicht: s50 sitzt exakt auf Rasterpunkten
   (pf ist k/10 diskret, 0.50-Rasterzier), das echte s50 liegt
   je einen halben Rasterpunkt unsicher. Moegliche Ordnung:
   Der Shift ist SUB-SQRT — groessere Netze brauchen weniger
   sqrt-skalierte Norm als vermutet. Wenn das haelt, ist das
   Design-Rezept GUENSTIGER als gedacht: 0.5*sqrt(H) liegt
   OBERHALB der Rampe bei beiden H (0.41/0.36 gemessen).
   Naechster Stein dafuer: H=192-Punkt (kostet ~8 min) als
   Extrapolations-Test der s50(H)-Kurve.
3. DESIGN-REZEPT (Stand T16, konservativ): Antriebsnorm >=
   1.0*sqrt(H) bleibt die sichere Wahl; die Rampe liegt bei
   ~0.4*sqrt(H) (Mitte), Ref-10%-Untergrenze unverletzt.
4. T15-KONSISTENZ: R96_3.5 pf 0.50 (T15: 0.60, andere Seeds/W-
   Ziehung) — im Rauschen. REF-Lotterei: REF24 0.90, REF96 1.00.

## Kernel im Echtbetrieb (der eigentliche Werkzeug-Befund)
- K-PRE-FLIGHT: OK vor dem ersten Trainingsschritt (0 verbrannte
  Minuten bei Verweigerung — dieses Mal nicht noetig).
- SELBST-MESSUNG: colnorm exakt (max Abweichung 0.0), dmdev ~1e-16,
  ndiv=0 ueberall. Die T11-Blindstelle ist geschlossen: haette das
  Instrument still normiert, haette der Audit-Alarm geschlagen.
- VERDICT generiert, LUECKENLISTE: KEINE, N-Skelett maschinell.
- BUG-PROTOKOLL: LEER. Erster Stein der Serie ohne jeglichen Bug-
  Eintrag, null Reparaturen, Laufzeit 33 min (unter der 45-55-
  Schaetzung). D-Diff-Snapshot liegt (results/t16_measurements.json)
  fuer den Crash-Fall.
- Ehrliche Einordnung: Ein sauberer Lauf ist EIN Datenpunkt, kein
  Beweis. Aber der Stein lief end-to-end (Contract -> Pre-Flight ->
  Selbst-Messung -> generiertes Verdikt -> Skelett) ohne jede
  menschliche Auswertungshand.

## Serie-Stand nach T16
Energie-Gesetz-Kette komplett: Energie -> NL-Eintritt -> Lernbarkeit
(T12/T13/T14), Uebergang = RAMPE (T15/T16), Rampen-Mitte s50(H)
steigt mit H, sub-proportional (T16), groessenordnungsmassig
s50 ~ 0.4*sqrt(H) im getesteten H-Bereich {24, 96}.
Offen als naechster Stein: s50(H)-Kurve um H=192 erweitern
(Extrapolationstest der Sub-Sqrt-Ordnung) — NEU zu registrieren.
Danach P3 (XVII als abfragbare Wissensstruktur), P4 (These-
Generator).
