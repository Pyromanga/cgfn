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
T21 -> OFFEN: BLOCK-ROBUSTHEIT (Seed-Lotto des G-Kapitels schlaegt im T-Bau zurueck)

## Offene Fragen (Stand 2026-11-11)
1. BLOCK-ROBUSTHEIT: Wie gross ist std(P_lernt) ueber disjunkte 20er-Seed-Bloecke? (T22-Kandidat; T21-Zahlen deuten auf Groessenordnung des T16-Trends)
2. RESEARCH TASTE: messbar machen (Nutzer-These Nachtrag 24; L3/L5: LLM schlaegt Stein vor, registriert VOR dem Lauf)
3. SCHALTER-PHAENOMEN: harte Konjugation brachte Lernen um (Strukturbruch, nicht Last) — Kandidat fuer eigene Linie, ungetestet.
