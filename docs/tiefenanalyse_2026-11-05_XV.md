# Tiefenanalyse XV — Ende des Betriebszustands-Programms
**Datum:** 2026-11-05 | **Commits:** F34b d012b95/11264a8/99177dc, F35 74f41bb/b35fdde/1df7eaf, F36 02295cb/391b809/eda076b/4a6db56, F37 4563f7c/8cc4bcd
**Charakter:** Epilog eines Programms, das an Messhygiene wuchs und an Physik starb.

---

## 1. Der Tagesbogen (4 Steine, 1 Tag)

| Stein | Frage | Ergebnis | Lehre |
|---|---|---|---|
| F34 Phase 2 | delta_H-Bestaetigung (8/10) | **TEIL-PASS 4/10** | Range-Restriktion an BEIDEN Achsen (dH-Saettigung bei H=128, R2-Decke bei trend/prod) |
| F35 | Zwei-Knopf: dH-Varianz bei festem ben | **UNENTSCHEIDEN** | lambda*->ben-Karte bei band-strukturierter Nicht-Normalitaet NICHT MONOTON (H=32); Manipulations-Check verhinderte falsches Verdikt |
| F36 | MSDM: Supply ohne Aufgabe, Demand ohne Reservoir | **UNENTSCHEIDEN** (+ Wurf gefallen: 0/5 gueltige Zellen >= +0.6, best +0.414) | Aufgaben-bewusster Beobachter schlaegt Skalare NICHT; Anker-Bruch repliziert |
| F37 | delta_H unter monotonie-robustem Matching, finaler Stein | **FINALER KILL: 2/5 gueltige Zellen** (sinus +0.795, prod +0.495; ben1-std 0.048 = Anker sauber) | Unter sauberem Matching traegt der Skalar nicht |

## 2. Die drei robusten Befunde, die den KILL ueberleben

### 2.1 Der Anker ist struktur-abhaengig (dreifach beobachtet)
ben=-0.30 ist bei band-strukturierter Nicht-Normalitaet NICHT sauber pin-bar:
F35/F36 (Bisektion): ben1 streute +0.05..-0.51 bei H=32. F37 (robustes
Bracket-Verfahren + Akzeptanz-Gate): 11/30 Modelle akzeptiert bei H=32
vs. 21/30 bei H=128. Interpretation: Bei H=32 dominieren transiente
Verstaerkungen den Benettin-Exponenten; die Karte lambda*->ben wird
nicht-monoton/flaechig verrauscht. Der groesste Lyapunov-Exponent ist
unter starker Nicht-Normalitaet KEIN stabiler Regler-Ankerpunkt.
Konsequenz fuer jeden kuenftigen Homeostaten: Ankerwahl ist selbst ein
Physik-Problem, kein Konventions-Problem.

### 2.2 Korrelationen sind Gitter-Eigenschaften (F35 vs F36 Forensik)
delta_H bei H=128 (jeweils gueltiges Matching): F35 wdelay8/prod/trend
-0.35/-0.35/-0.34; F36 +0.90/+0.58/+0.76. Dasselbe Observable, derselbe
Messpfad, verschobenes Gitter (b in {1,2,4,8}x s<=1.2 vs {1,8}x s<=1.6),
Vorzeichen-Kipp. Mit F34 Phase 1 (10/10) vs. Phase 2 (4/10): Das Muster
ist nowendungsfrei — Skalar-Korrelationen in diesem Designraum sind
Kovarianz-Artefakte der Gitterwahl, keine Systemeigenschaften.
Das ist die eigentliche, generalisierbare Lehre des Programms.

### 2.3 delta_H <-> sinus-Memory: das eine, replizierte Signal
Ueber DREI unabhaengige Designs (F35: +0.891, F36: +0.902, F37: +0.795
bei sauberem Anker): delta_H korreliert robust mit sinus_memory.
Dokumentiert als repliziertes NEBENresultat, aufgaben-spezifisch,
KEIN Betriebszustand. Freeze-Regel: kein neuer Stein darauf, kein
Post-hoc-Promotion. Es ist ein Hinweis auf Eingangsrichtung-transiente
Kopplung — abgelegt, nicht angefasst.

## 3. Was NICHT ueberlebt hat
- MC: Zombie des Tages (+0.93/+0.95/+0.96 bei H=32 in F36 — immer nur
  im ben1-Konfund-Territorium; F37: 1/5 gueltige Zellen).
- MSDM (Supply-Demand-Trennung): gefallen, ehrlich, registriert.
- top25/FTLE (F33), E3b-Task-Bezug, F30.3-Ressourcen-These: blieben tot.
- Die Skalar-Hypothese als solche: F37 FINALER KILL, kein vierter
  Versuch, keine Nachverhandlung. Rueckstufung auf F27-Phaenomenologie.

## 4. Methoden-Kapital (bleibt im Werkzeugkasten)
1. Manipulations-Check (corr(ben1, TASK_R2) < 0.4): rettete F35 UND F36
   vor falschen Verdikten. Billigste Versicherung des Programms.
2. Akzeptanz-Gate (|ben - Ziel| <= 0.05, Ungueltige LAUT): staerker als
   jeder post-hoc-Check — Konfund wird konstruktiv getoetet (F37:
   ben1-std 0.048, Manipulations-Check damit ueberfluessig-streng).
3. Monotonie-robustes Matching (Sweep + Bracket + lokale Verfeinerung):
   ersetzt Bisektion, wo Nicht-Monotonie verdaechtig ist.
4. Zwei-Phasen-Suche (Breite ohne Schwelle, dann registrierte
   Befaetigung auf frischen Seeds): F34 bewaehrt, F37 als its cleaner
   Endpunkt.
5. Messungen strikt trennen (Demand ohne Reservoir, Supply ohne
   Aufgabe): F6-Muster — der Ansatz fiel, aber die Trenn-Disziplin
   bleibt.

## 5. Literatur-Anschluss (Recherche 2026-11-04/05)
- arXiv 2607.17858 (Organization of computation, Jul 2026): mode-
  aufgeloeste IPC — aggregate Skalare zeigen nicht, wo Information
  sitzt. Unser KILL ist deren These in Aktion: delta_H als Aggregat
  verliert gegen Moden-Geometrie.
- PNAS 2424991122 (Optomechanical RC, 2025): spektrales Alignment
  (aggregiert, Frequenzraum) — MSDM war der modale, getrennte Wurf
  darueber hinaus; er fiel, das Feld bleibt offen fuer eine runde 2
  mit anderem Design — NICHT in diesem Programm.
- Huelser et al. 2023 (Nanophotonics): totale IPC korreliert schlecht
  mit Aufgabenleistung — deckt sich mit MC-Zombie und F30.4.
- Kleyko et al. 2511.14484 (Nov 2025): Perzeptron-Theorie vorhersagt
  ESN-Leistung analytisch — Theorie-Baseline-Kandidat fuer jeden
  kuenftigen korrelativen Claim.
- Carroll 2012.01409 (lokal vs. global Lyapunov): Anker-Befund 2.1 ist
  dessen Reservoir-Version — der globale Exponent verliert seine
  Ordnungsrolle, wenn transiente Verstaerkung dominiert.

## 6. Konsequenzen
- **Betriebszustands-Programm: BEENDET.** Kein neuer Skalar-Stein,
  keine Design-Fixes, keine dritte Phase. Dokumentiert hier.
- **Aktive Front: F27** (kappa_T-Erosion durch Adam, zweiphasig,
  mechanistisch, repliziert) — dort liegt das ehrige, harte Resultat.
- Optional offen (Ressourcen-Wette): F27-Entzensierung (SGD 600 Steps)
  als kleines Detail; Feuerplan (F28, F26, F25...) bleibt eingefroren
  bis entschieden.
- Chess-manifold (C1-C4): unberuehrt, laeuft parallel weiter.

## 7. Wort an den Kapitaen
Das Programm starb nicht an einem Messfehler — es starb an drei
sauber registrierten Steinen, von denen zwei die eigene Messung
verweigerten und der dritte sauber negativ war. Das ist der Unterschied
zwischen einem Grab und einer Grube: Wir wissen, wo es liegt, warum es
liegt, und was es kostet, nochmal zu graben. 40 Euro Labore finden keine
Durchbrueche, indem sie laenger suchen — sondern dadurch, dass sie auf
die naechste richtige Frage stossen, und die heisst heute F27.
