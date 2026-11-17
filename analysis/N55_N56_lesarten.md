# Nachtraege 55/56 — Lesarten (handgeschrieben, ehrlich), 2026-11-17

## Nachtrag 55 — T31 VOLL-AEQUIVARIANTE ROTATION: VERDICT GEMISCHT
Gemessen: P_PART(0)=0.80 | P_PART(1)=0.10 | P_FULL(0)=0.60 | P_FULL(1)=0.20.
Sanity intakt: |P_FULL(0) - P_PART(0)| = 0.20 < 0.32 (Block-Rauschen-Grenze).
Band: P_PART(1) = 0.10 in [0.00, 0.22] — TREFFER (T23/T24/T25-Band haelt).
Keine registrierte Konstellation sitzt: P_FULL(1) = 0.20 liegt in der
vorab registrierten Grauzone [0.10, 0.40].

### Lesart
1. Der Readout-Einwand traegt NICHT als vollstaendige Erklaerung: der
   Kollaps ueberlebt die volle Rotation (P_FULL(1) = 0.20, klar unter 0.50).
   Waere der Kollaps ein reines Teil-Rotations-Artefakt, muesste FULL bei
   a=1 auf Kontroll-Niveau zurueckkommen. Tut er nicht.
2. ABER: Teil-Rotation kollabiert STAERKER als volle Rotation (0.10 vs
   0.20). Ein Teil des T23-Kollaps kommt also doch aus der Inkonsistenz
   der Teil-Rotation (W gedreht, b/Uw in der alten Basis). Ehrliche
   Konsequenz: die T23-T25-Lesart stuetzt sich kuenftig auf die FULL-Zahl
   (0.20), nicht auf die PART-Zahl (0.10). Der Schalter ist real, aber
   die fruehere Effektstaerke war teilweise ein Rotations-Implementierungs-
   Effekt. Das ist eine Verengung, kein Widerruf.
3. 0.10 vs 0.20 liegt innerhalb des Block-Rauschens (SE je Arm ~ 0.09-0.11,
   n=20): die Differenz PART vs FULL bei a=1 ist HINWEIS, kein Befund.
   Getrennt ist nur FULL(1)=0.20 vs FULL(0)=0.60 und die Replikation des
   PART-Kollaps auf frischem Block.
4. Washout-Caveat: beruehrt GEMISCHT nicht (registriert: er schwaecht nur
   den PASS, nie den KILL). T31b ist damit NICHT noetig — der Vorbehalt
   galt nur fuer einen PASS-IMMUN-Anspruch, der nicht eingetreten ist.

## Nachtrag 56 — T32 RUECKMELDUNGS-DICHTE BEI FIXEM HORIZONT: VERDICT PASS (DICHTE-TRAEGER)
Gemessen: P_SPARSE(0)=0.75 | P_SPARSE(1)=0.05 | P_DENSE(0)=1.00 | P_DENSE(1)=1.00.
Sanity intakt (0.75 >= 0.25, 1.00 >= 0.50). Falsifikation nicht getreten.
Band: P_SPARSE(1) = 0.05 in [0.00, 0.22] — TREFFER; P_SPARSE(0) = 0.75
ueber dem registrierten Band [0.25, 0.64] — NEIN (nachgetragen, nicht
nachverhandelt: Band verfehlt nach OBEN, Richtung entlastet).
Die registrierte PASS-Konstellation sitzt exakt.

### Lesart
1. Kern-Befund: Bei FIXEM Gedaechnis-Horizont D=20 kollabiert die SPARSE-
   Aufgabe unter Konjugation (0.05) und die DENSE-Aufgabe lernt in ALLEN
   Konjugationen (1.00, 40/40). Die Rueckmeldungs-Dichte, nicht die
   Gedaechnis-Forderung, ist der Traeger des Schalters. Konsistent mit
   arXiv 2602.13934 (dichte, lokale, verifizierbare Rueckmeldung als
   Skalierungstraeger) — jetzt an einem kontrollierbaren Mini-System
   kausal-variiert, eine Variable, ein Befund.
2. Ehrliche Konfundierung, VOR der Feier registriert: Dichte variiert
   ZWEI Dinge — die Rueckmeldungs-Struktur UND die Gradienten-Signal-
   staerke pro Step (DENSE hat ~20x mehr Loss-Terme je Sequenz). Die
   alternative Lesart lautet: nicht die Dichte als Aufgabenstruktur,
   sondern das staerkere Gradientensignal rettet das Training (oder
   wascht die Rotation schneller aus). Der Befund Differenz SPARSE vs
   DENSE haelt auch dann, aber die DEUTUNG (Aufgabenstruktur vs
   Optimierungs-Dynamik) ist damit noch nicht entschieden.
3. Registrierte Konsequenz: T33-Kandidat SIGNALSTAERKE-MATCHED-KONTROLLE
   (DENSE mit lr/20 ODER SPARSE mit 20x Steps) — erst wenn der Schalter
   unter gematchter Signalstaerke erhalten bleibt, ist die Dichte-These
   als geschlossen zu behandeln. Bis dahin: PASS als registriert,
   Deutung vorlaeufig.
4. Washout-Caveat (aus N55): trifft hier NICHT die Kern-Logik — Washout
   wuerde GLEICHMAESSIG helfen und kann die SPARSE-Kollaps-Differenz
   nicht erzeugen. Er verschaerft aber Punkt 2: schnelleres Washout im
   DENSE-Arm ist genau die Optimierungs-Lesart.

## Bilanz (2026-11-17)
Zwei Steine gefeuert (T31, T32), 160 Runs, 0 verbrannte Minuten durch
Gates. Ein GEMISCHT (T31: Verengung der T23-Lesart, Schalter bleibt),
ein PASS (T32: Dichte-Traeger, Deutung mit registrierter Konfundierung
offen). Der externe Kritiker hat zweimal richtig gelegen: die Teil-
Rotation war ein ungetesteter Freiheitsgrad (T31), und die Dichte-
Frage war ohne seine Korrektur eine Tautologie-Zelle gewesen (T32).
