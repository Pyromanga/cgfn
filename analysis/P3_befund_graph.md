# P3-Befund-Graph (v1, 2026-11-10) — L2-Instrument, 0 Runner-Minuten

Format: KNOTEN = Stein (id | Kern-Frage | Verdikt | Kern-Zahl | XVII-Anker).
Zweck: maschinenlesbares Fundament fuer L3/L5 (Vorschlags-Interface).
v1 hand-autoriert aus analysis/XVII_anker_kapitel.md; Update-Pflicht:
jeder neue Stein bekommt einen Knoten IM SELBEN COMMIT wie sein Nachtrag.

## Serie G — Anker-Kapitel (Reservoir-Geometrie, XVII Kopf, 2026-11-06)
G1  | Hat der spektrale Knob Kontroll-Autoritaet?              | KILL          | max 4/10 Treffer (registriert >=5/10)           | XVII Kopf
G2  | Seed-Lotto: Basis oder Spektrum?                          | KILL          | std(nur-Basis) ~ std(alles)                    | XVII Kopf
G3  | Antriebs-Ausrichtung der Knopf?                           | KILL          | std_A/std_B = 0.62..1.11                       | XVII Kopf
G4  | Elementweise tanh der Anker?                              | UNENTSCHEIDEN | NC kollaps in 3/4 Zellen (Bias-Confound)       | XVII Kopf
G5  | Anker-These in f64, Instrument repariert                  | PASS          | std_NC bis 1e-12 (bis 1e12x Kollaps)            | XVII Kopf
G6  | Traegt der Anker-Knopf Aufgabenleistung?                  | KILL          | MC -40% bis -79% bei std=0 (NC)                | XVII Kopf
G7  | Anker-Winkel ein kontinuierlicher Regler?                 | KILL          | Varianz-Kollaps nur als Kliff bei w=1           | XVII Kopf
G8  | Zufalls-Anker ein Kapazitaets-Suboptimum?                 | PASS          | +16%/+41% MC bei halbierter Streuung            | XVII Kopf
G9  | Verteiler-Mechanismus (UNI vs CONC vs RND)?               | KILL          | UNI haelt nur in s=0-Zellen; Defekte D1/D2     | Nachtrag 1
G10 | Budget-matched Verteiler?                                 | KILL          | MC = SKALAR-Funktion des Budgets                | Nachtrag 1

## Serie T — Tiefgang (minimale echte Netze)
T1    | Feldgesetz-Reparatur (gerahmter Jacobianer)?        | KILL          | P3 alt-Fehler 0.73 vs neu 0.085 (2x, unzureichend) | Nachtrag 2
T2    | Transfer-Stein                                       | UNENTSCHEIDEN | 2 Strikes; Explorations-Signal                 | Nachtrag 3
T3    | Primartoere                                          | UNENTSCHEIDEN | Sanity-Klausel selbst gemurkst (formal)      | Nachtrag 4
T4/T4b| Produkt-Familie                                     | UNENTSCHEIDEN | 2 Strikes; Linie TOT                          | Nachtrag 5
T6    | Lernbarkeits-Spektrum                                | KILL          | Geometrie = Entkopplung                        | Nachtrag 6
T7    | MC im T-Bau                                          | KILL          | MC FLAT; G8 unberuehrt (Verdikt-Text zurueckgenommen) | Nachtrag 7
T8/T8b| G8-Knopf im Trainingslabor                          | UNENTSCHEIDEN | Antriebs-Verhungerung diagnostiziert          | Nachtrag 8
T9    | Kohaerenter Moden-Antrieb                            | UNENTSCHEIDEN | Sanity gerissen                               | Nachtrag 9
T9b   | Richtung oder Zahl?                                  | KILL-C        | RICHTUNG, nicht ZAHL                          | Nachtrag 10
T10   | Zerlegung des Antriebs                               | KILL-B        | Nur iid-Koordinaten-Antriebe lernen           | Nachtrag 11
T11   | Amplituden-Kontrast                                  | UNENTSCHEIDEN | Instrument-Bug (mea culpa); Nebenbefund: Antriebs-ENERGIE eigener Knopf | Nachtrag 12
T12   | Energie-Kante                                        | PASS (P-ENERGIE) | Verhungerungs-Story war die ENERGIE (groesste Selbstkorrektur) | Nachtrag 13
T14   | Mechanismus-Kette                                    | PASS (P-MECH) | Energie -> NL-Eintritt -> Lernbarkeit geschlossen | Nachtrag 15
T15   | Skalierungsgesetz der Kante                          | KILL-A (Rerun; Run1 UNENTSCHEIDEN Readout-Bug) | Uebergang ist RAMPE, keine Kante | Nachtraege 16/17
T16   | 2D-Karte H x s (erster kernel-generierter Stein)     | PASS (P-RAMPE-SHIFT-SUBH) | Rampe verschiebt sich sub-proportional zu H | Nachtrag 23
T17   | s50-Extrapolation bei H=192                          | PASS (P-SUBSQRT-EXTRAPOLATION) | s50 ~ 0.55 * H^0.41 | Nachtrag 25
T18   | Theory-Core-Vorhersage                               | PASS formal  | reines Potenz-Gesetz KNICKT (14% daneben)      | Nachtrag 27
T19   | s50(48)-Vorhersage 2.55                              | UNENTSCHEIDEN | Raster verfehlt; Reparatur R2                 | Nachtrag 28
T19-R2| Raster-Reparatur                                     | KILL-A        | s50(48)=1.83 < 2.05; NICHT-MONOTONIE: 12:1.30, 24:2.00, 48:1.83, 96:3.50, 192:4.675 | Nachtrag 29
T20   | DM-Kovariat kausal? (Konjugation)                    | UNENTSCHEIDEN | Stufen-Raster verfehlt; Doppelkollaps DM 13.21->1.4 & P(2.2) 0.75->0.05 = Schalter-Signal | Nachtrag 31
T20-R2| DM-Treppe (WB-Regler)                                | KILL (DM-NEBENSPUR) | pfspread220=0.05 < 0.4; DM um Faktor 2.3 gedreht, nichts passiert; DM-Linie TOT | Nachtrag 32
T21   | Delle real oder Seed-Rauschen? (100 frische Seeds)  | KEHRUNG       | d183=-0.38; H24 lernt an ALLEN Rasterpunkten frueher; Delle existiert im Block 210..309 nicht; H-Trend BLOCK-EMPFINDLICH (P48(2.2) 0.75 alt vs 0.43 frisch ≈ 2.9 Sigma) | Nachtraege 33-36
T22   | War der alte Block ein Ausreisser? (5 neue disjunkte 20er-Bloecke; erster LLM-Vorschlag) | PASS (ALT-AUSREISSER) | zalt=2.72 >= 2.5; Block-P 0.35/0.40/0.50/0.40/0.65; mean5=0.46 std5=0.107; B4=0.65 reproduziert den Ausreisser-Mechanismus; L5-Band 3/3 | Nachtraege 38-39
T23   | Repliziert der a=1-Kollaps auf frischen Bloecken? (erster MASCHINEN-Vorschlag P4 v1.1; R5-Reparatur Pfad (b)) | PASS (SCHALTER-BESTAETIGT) | drop1=0.55 (~5.7 SE >= 0.30); P-Stufen 0.575/0.025/0.10/0.15; keine Dosis-Kurve (Plateau + Mini-Erholung); L5 Maschine 5/6 (drop124 verfehlt an duenner Quelle) | Nachtraege 40-42
T24   | Kippt JEDE 1-dim-Konjugation? (5 Achsen q_seed x 20 frische Seeds; Maschinen-Vorschlag P4 v1.2, R8-Band) | PASS (RICHTUNGS-AGNOSTISCH) | pqmax=0.10 <= 0.15; Achsen-P_Q 0.05/0.05/0.05/0.10/0.00 (im Block-Rauschen, kein Achsen-Lotto); P(0)=0.60; L5-Band 7/7 (R8-Erstfahrt); sauberster Stein: 0 Reparaturen | Nachtraege 43-44
T25   | Ist DM ein Ordnungsparameter des Schalters? (Stufen a {0,1,4,24}; Maschinen-Vorschlag P4 v1.3, Doppelklausel dmeff) | PASS (ORDNUNGSPARAMETER EBENE) | dmeff=11.65 (~55 SE); DM 13.06 -> 1.33-1.59 Plateau wie P-Kurve; BEYFANG: Per-Seed-Trennung NEGATIV (13.13 vs 12.97, Delta < SE); L5-Band 7/7; zweiter interventionsfreier Stein | Nachtraege 45-46
T26   | Trennt eine Groesse gelernt vs gescheitert INNERHALB eines Blocks? (Maschinen-Vorschlag P4 v1.4; R5 = Emissions-Reparatur) | KILL (INTRA-BLOCK-LOTTO) | sepmax=-1; max t=0.71 (gainmean) << 3.0 — Faktor 4 Luft; alle 5 Kandidaten tot; Per-Seed-Frage NEGATIV geschlossen; L5-Band 3/5 (Maschinen-Band zu eng) | Nachtraege 47+49
T27   | Kippt der Schalter im GRADIENTEN-trainierten Netz? (LINIE TRAIN, erster Gradienten-Stein; R5 600->2500) | UNENTSCHEIDEN (SANITY, 2x) | P_lernt=0.00 in ALLEN Armen inkl. Kontrolle; DM_final trennt 7.58 (~54 SE) — Struktur ueberlebt Training; R5 verbraucht | Nachtraege 48+50

## Serie W — Werkzeug
W1-P1 | Lab-Kernel Historien-Test                           | PASS          | T15 exakt wiedergegeben; T11-Bug angezeigt; T15-Run1-Bug verweigert | Nachtrag 19
W1-P2 | Vollausbau D/K/N                                    | PASS          | im ERSTEN Versuch; Architektur-Schuld eingeloest | Nachtrag 21

## Kanten (Folge-Logik, wichtigste)
G5 -> G8 (Anker bewiesen -> Suboptimum quantifiziert)
T8 -> T9 -> T9b -> T10 -> T12 (Verhungerungs-Diagnose-Kette: RICHTUNG -> iid -> ENERGIE)
T12 -> T14 -> T15 (Kante -> Mechanismus -> Skalierung)
T15 -> T16 -> T17 -> T18 -> T19 -> T19-R2 (Theory-Core-Kette; KILL-A oeffnet Delle-Frage)
T19-R2 -> T20 -> T20-R2 (Kausal-Test H-Alternative DM; beide Achsen tot)
T19-R2 -> T21 (Delle-Frage)
T21 -> T22 (Block-Robustheits-Frage; erster LLM-Vorschlag des L5-Interfaces, Band 3/3)
T22 -> OFFEN: BLOCK-MITTEL-GESETZ (H-Trend nur noch als Block-Mittel mit std(P_Block) testbar)
T20-R2 -> T23 (Schalter-Signal als Dosis-Kurve; erster Maschinen-Vorschlag, Kritiker-R7 fing v1.0-Raster)
T23 -> T24 (Richtungs-Frage; Maschinen-Vorschlag P4 v1.2, R8-Band-Haertung Erstfahrt)
T24 -> T25 (Ordnungsparameter-Frage; Maschinen-Vorschlag P4 v1.3, Doppelklausel dmeff)
T25 -> OFFEN: PER-SEED-ORDNUNG (DM trennt Ebenen, nicht Seeds; Kandidaten Gain-Verteilung / DM-Varianten, sonst Intra-Block-Lotto unveraenderlich)
T25 -> T26 (Per-Seed-Frage; Maschinen-Vorschlag P4 v1.4 aus T25-Beyfang)
T25 -> T27 (Bruecken-Frage LINIE TRAIN; erster Gradienten-Stein, Hand-Registrierung N48)
T27 -> OFFEN: TRAIN-INSTRUMENT (delayed recall fuer flaches MLP ungelernbar — P=0 auch in Kontrolle; neuer Stein nur mit neuem Task-Design, Kapitaen-Entscheidung)

## Offene Fragen (Stand 2026-11-12, nach T26/T27)
0. PER-SEED-ORDNUNG: GESCHLOSSEN (T26 KILL — Intra-Block-Lotto unveraenderlich mit fixierten Kandidaten; DM bleibt Ebenen-Marker). Backlog: Gain-VERTEILUNG als Ebenen-Kandidat.
1. PER-SEED-ORDNUNG: Trennt UEBERHAUPT eine Groesse gelernt vs gescheitert INNERHALB eines Blocks (Kandidaten: Gain-Verteilung, DM-Varianten), oder ist das Intra-Block-Lotto unveraenderlich? — T26-Kandidat nach Generator-Rotation.
2. RESEARCH TASTE: Serie 22/23 gesamt (LLM 3/3; Maschine 19/20; R8-Bilanz 14/14 nach zwei Einsaetzen); Kalibration verengt (Nutzer-These Nachtrag 24).
3. BLOCK-MITTEL-GESETZ: vertagt (Cap + konfirmatorisch; T22 liefert 10 Block-Konstanten als Startpunkt).

