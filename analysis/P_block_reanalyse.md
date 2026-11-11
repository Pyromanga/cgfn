# P-Block-Reanalyse (2026-11-11, 0 Runner-Minuten) — D-Diff aus T21-Einzelseed-Daten

Quelle: results/t21_delle_statistik_table.csv (Bot-Commit be35ddf, 700 Einzelseed-Werte).
Methode: Die 100 frischen Seeds 210..309 sind 5 disjunkte 20er-Bloecke
(B0=210-229, B1=230-249, B2=250-269, B3=270-289, B4=290-309).
Block-P_lernt je Arm (Anteil acc>=0.99 je Block).

## Block-P je Zelle
Arm        | B0   | B1   | B2   | B3   | B4   | mean  | std
H24_s1.55  | 0.50 | 0.40 | 0.55 | 0.45 | 0.55 | 0.490 | 0.058
H24_s1.83  | 0.75 | 0.70 | 0.60 | 0.55 | 0.75 | 0.670 | 0.081
H24_s2.2   | 0.95 | 0.75 | 0.85 | 0.55 | 0.80 | 0.780 | 0.133
H48_s1.55  | 0.00 | 0.20 | 0.10 | 0.20 | 0.15 | 0.130 | 0.075
H48_s1.83  | 0.25 | 0.25 | 0.40 | 0.30 | 0.25 | 0.290 | 0.058
H48_s2.2   | 0.35 | 0.40 | 0.55 | 0.50 | 0.35 | 0.430 | 0.081
HIGH48     | 1.00 | 1.00 | 1.00 | 0.95 | 0.90 | 0.970 | 0.040

## Gepaarte H-Differenzen je Block (gleiche Seeds an H24 und H48)
d183 je Block: -0.50 / -0.45 / -0.20 / -0.25 / -0.50  (KEHRUNG in 5/5)
d220 je Block: -0.60 / -0.35 / -0.30 / -0.05 / -0.45  (negativ in 5/5)

## Lesart (zu Nachtrag 37)
1. std(P_Block) liegt zwischen 0.06 und 0.13 — dieselbe Groessen-
   ordnung wie die T19-Delle (8.5 Prozent) und die T16-Trend-Offsets.
   Ein einzelner 20er-Block kann Effekte dieser Groesse vortaeuschen
   und verstecken.
2. Die KEHRUNG ist davon UNBERUEHRT: gepaart (dieselben Seeds an
   beiden H) ist d183 in jedem der 5 Bloecke negativ. Der H-Effekt
   im frischen Block ist kein Block-Artefakt. Offen: die Versoehnung
   mit dem alten Block 190..209 (dort nur ungepaartes Material;
   P48(2.2)=0.75 vs frisch 0.43, ≈ 2.9 Sigma zwischen den Block-
   Mitteln).

## Referenz fuer T22
Die 5 frischen Bloecke dieser Reanalyse sind die Vergleichsbasis;
T22 misst 5 WEITERE disjunkte Bloecke (310..409) und bewertet den
alten Wert 0.75 als z-Mass gegen die neue Verteilung.
