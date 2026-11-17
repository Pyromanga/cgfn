# Nachtraege 55/56 — Lesarten (handgeschrieben, ehrlich), 2026-11-17
(Korrektur 2026-11-17, siehe Punkt 2 der N56-Lesart und Nachtrag 57.)

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
   (0.20), nicht auf die PART-Zahl. Der Schalter ist real, aber die
   fruehere Effektstaerke war teilweise ein Rotations-Implementierungs-
   Effekt. Das ist eine ZERLEGUNG in zwei Mechanismen, kein Widerruf:
   tanh-Basis-Fehlausrichtung (ueberlebt volle Rotation, traegt ~0.20)
   und Input-W-Inkonsistenz (nur bei Teil-Rotation, traegt den Rest).
   Zitier-Pflicht: wer T23 zitiert, zitiert kuenftig 0.20 (FULL), nicht
   die alte PART-Zahl (Nachtrag 57, T23-Reankerzung).
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
2. KORREKTUR (2026-11-17, nach externem Einwand): Diese Lesart sprach
   zunaechst von "~20x mehr Loss-Terme" — FALSCH. Korrekt: T_SEQ=24,
   DELAY=20 → 4 Label-Frames; DENSE hat 4x mehr Terme je Step (128 vs 32
   bei batch=32). Und bei Mean-Reduction ist die Gradienten-MAGNITUDE je
   Step vergleichbar — der Konfund ist die VARIANZ (Mittelung ueber 4
   Lehrer-Terme), nicht die Magnitude. Die urspruenglich geplante
   Falsifikation "DENSE lr/20 oder SPARSE 20x Steps" matchte daher die
   falsche Groesse und wurde VOR dem Lauf von T33 ersetzt (Nachtrag 57):
   PRIMARY ist der maskierte Backprop (MB), plus SPARSE-4x-Steps (S4X).
   Der Konfund selbst bleibt real und wird jetzt geschlossen, nicht nur
   benannt.
3. Registrierte Konsequenz: T33 (Signalstaerke-Match, N57). Erst wenn
   der Schalter unter gematchtem Signal erhalten bleibt (MB haelt,
   S4X faellt), ist die Dichte-These als geschlossen zu behandeln.
   Bis dahin: PASS als registriert, Deutung vorlaeufig.
4. Washout-Caveat (aus N55): trifft hier NICHT die Kern-Logik — Washout
   wuerde GLEICHMAESSIG helfen und kann die SPARSE-Kollaps-Differenz
   nicht erzeugen.

## Bilanz (2026-11-17)
Zwei Steine gefeuert (T31, T32), 160 Runs, 0 verbrannte Minuten durch
Gates. Ein GEMISCHT (T31: Zerlegung in zwei Mechanismen, Schalter
bleibt), ein PASS (T32: Dichte-Traeger, Konfund registriert und an T33
uebergeben). Der externe Kritiker hat zweimal richtig gelegen: die Teil-
Rotation war ein ungetesteter Freiheitsgrad (T31), und die Dichte-Frage
waere ohne seine Korrektur eine Tautologie-Zelle gewesen (T32). Ein
Fehler geht auf mein Konto: die ~20x-Zahl in der ersten Fassung dieser
Lesart (korrigiert, Punkt 2 oben) — der Kritiker hat sie uebernommen,
bevor ich sie selbst pruefte.
