

---

# Nachtrag 25 (2026-11-09): T17 — PASS (P-SUBSQRT-EXTRAPOLATION). Drei s50-Punkte, ein Kandidaten-Gesetz: s50 ~ 0.55 * H^0.41. Der Theory-Core hat sein erstes Material.

## T17 (Lauf 23:43-00:06 UTC, 23 min, Bot-Commit 7e6da59, Seeds 170..179, 6 Arme, 0 Divergenzen, Bug-Protokoll leer)
VERDICT (GENERIERT): PASS — P-SUBSQRT-EXTRAPOLATION.
  s50(192) = 4.675 | Anker s50(96) = 3.5 (T16) | ratio2 = 1.3357,
  strikt in (1.0, 1.4142) — knapp unter sqrt(2), Sub-Sqrt haelt.
  RAMPE H=192: pf = 0.00 / 0.30 / 0.40 / 0.80 / 0.80
       (s = 2.8 / 3.5 / 4.4 / 5.5 / 7.0)
  SANITY REF192: 0.90. colnorm exakt, dmdev ~1e-16, ndiv = 0.

## Die drei s50-Punkte (T16/T17, konsistent gemessen)
  H= 24: s50 = 2.000  (T16)
  H= 96: s50 = 3.500  (T16)
  H=192: s50 = 4.675  (T17)

## Kandidaten-Gesetz (MATERIAL, unregistriert — der Theory-Core-Fit)
Log-Log-Fit ueber alle drei Punkte:
  s50(H) ~ 0.547 * H^0.408
  Probe: H=24 -> 2.00 (exakt), H=96 -> 3.52 (vs 3.50), H=192 ->
  4.67 (vs 4.675). Alle drei Punkte innerhalb 1 Prozent.
Der EXPONENT ~0.41 ist stabil ueber beide H-Schritte (0.40 im
ersten, 0.42 im zweiten Doppel-Schritt) — kein Drift. s50/sqrt(H)
faellt weiter (0.408 / 0.357 / 0.337), konsistent mit Exponent <
0.5. EHRliche Grenzen: (1) drei Punkte, n=10-Seeds, s50-Raster-
unsicherheit +-1 halber Rasterpunkt; (2) nur EINE Aufgabenfamilie
(Paritaet), EINE W-Familie (T6-B60), EINE Antriebsgeometrie
(REFGEOM); (3) der Fit ist deskriptiv, KEIN registriertes Verdikt.

## Was das Gesetz bedeutet (wenn es haelt)
- Design-Rezept GUENSTIGER als jede fruehere Version: Antriebsnorm
  >= ~1.5 * s50(H) ~ 0.8 * H^0.41 waechst deutlich langsamer als
  sqrt(H) — grosse Netze sind ANTRIEBS-BILLIGER als die sqrt-Regel
  behauptete.
- Die NL-Eintritts-Kette (T14) bekommt eine quantitative Spitze:
  Der benoetigte Antrieb waechst sub-thermisch mit der Breite.

## DER NAECCHSTE SCHRITT: der erste THEORY-CORE-PRUEFSTEIN (T18)
Der Stack bekommt seinen ersten echten Vorhersage-Test: Das gefittete
Gesetz sagt EINE Zahl voraus, ein neuer Stein prueft sie am Kernel.
  T18 (registriert VOR dem Lauf, NEU): H=12 (billigste Breite, keine
  Messung existiert). VORHERSAGE aus dem Fit: s50(12) ~ 0.547 *
  12^0.408 ~ 1.51. Design: REF12 + R12_s-Raster um die Vorhersage
  (s in {0.8, 1.1, 1.5, 1.9, 2.4}), Seeds 180..189. Registriertes
  Verdikt: PASS wenn das interpolierte s50(12) in (1.1, 1.9) liegt
  (Vorhersage +-25 Prozent, grosszuegig gegen n=10-Rauschen und
  Rastergroebheit); KILL wenn ausserhalb — dann faellt das
  Potenz-Gesetz als Einzelgesetz und der Fit war Zufall.
  Kosten: H=12 ist der billigste Stein der Serie (~10-15 min).
Das ist der geschlossene Kreislauf in Miniatur: Weltmodell (Fit)
sagt voraus -> Contract -> Lauf -> generiertes Verdikt -> Graph.
GENAuso wird Research-Taste messbar (Nutzer-These, Nachtrag 24):
Jede Vorhersage des Modells ist ein Trainingsbeispiel.

## Kernel-Protokoll (Echtbetrieb, Stein 2 von 2)
Zweiter kernel-generierter Stein, wieder: K-Pre-Flight OK, Selbst-
Messung sauber, Verdikt generiert, LUECKENLISTE: KEINE, Bug-
Protokoll: LEER, Laufzeit 23 min (unter der 40-55-Schaetzung).
Echtbetriebs-Bilanz des Kernels: 2 Steine, 0 Bugs, 0 Reparaturen.
