# Tiefenanalyse XII (2026-11-03): Der Kill von F30 v2 — und die fuenfte Ratte

**Ergebnis (Run 26, Instrument v2):**
    corr(MC, NLM_R2)     = +0.817
    corr(MC, TASK-speed) = -0.499   <<< KILL (vorab: >= +0.4)
    corr(ben, speed)     = +0.729
    Schwellen-Reichweite: 48/48 ohne Threshold (400 Steps, sinus_memory)

**URTEIL: F30 v2 = KILL.** Das Urteil STEHT — der Konstrukt 'MC sagt
die Adaptationsgeschwindigkeit unter Adam voraus' ist tot. Kein
Nachverhandeln, kein Re-Roll. Aber die Zerlegung zeigt, WAS genau
gestorben ist — und das aendert die Lage vollstaendig.

## 1. Die Zerlegung (nachtraeglich, diagnostisch, nicht entscheidend)

    corr(speed, log loss0)        = +0.991
    corr(MC, log loss0)           = -0.540
    corr(MC, -log lossE)          = +0.429
    partial corr(MC, speed|loss0) = +0.316
    lossE je s: 0.381 / 0.329 / 0.424 (Baseline 'immer 0 raten': ~0.5)

(a) 'speed' ist zu 99% der Anfangsverlust-Kanal: lossE ist flach,
    also misst speed = log(loss0/lossE) fast nur die anfaengliche
    Zustaendsexplosion (chaotische Konfigs starten bei loss bis 29.5,
    Adam drueckt ALLES auf 0.35-0.42 — 'Fortschritt' war Warm-up).
(b) KEINE Konfiguration lernte den Task (s2thr 48/48, lossE hauchduenn
    unter Baseline) — ein 'Lerngeschwindigkeits'-Mass, wo nichts lernt,
    misst keine Lerngeschwindigkeit.
(c) Der Konstrukt-Kanal, der uebrig bleibt, ist positiv:
    corr(MC, -log lossE) = +0.429 — ueber dem Nagel, schwach, aber
    im Vorzeichen der These. Und corr(MC, NLM) = +0.817 ist stark.

## 2. Ratte Nr. 5: Messung durch den Antagonisten

F30 v2 mass die Ressource DURCH den Optimierer (Adam-Adaption).
Unsere eigene E3b-These (Tiefenanalyse VI/XI) sagt VORHER, dass der
Optimierer ein nicht-normaler Antagonist ist, der die Ressource
erodiert und verzerrt. Ein Null-Test, der die Ressource durch den
Antagonisten misst, ist zirkulaer — wir haben gegen unsere eigene
Theorie ein Messgerät gebaut. (Run 24: gesaettigt. Run 26: vom
Antagonisten dominiert. Zwei Instrumente, zwei Ratten, eine Lehre.)

## 3. F30.3 — der korrekte Null-Test (VORAB REGISTRIERT)

Konstruktwechsel, deklariert und begruendet: statt Optimierer-
Geschwindigkeit jetzt Task-LEISTUNG, optimizer-frei:
    - sinus_memory (identisch zu F30 v2: freq 0.25-2.0 rad/step,
      delay 3), States sammeln,
    - Ridge-Readout in geschlossener Form (deterministisch, kein
      Gradient, kein Adam, keine Lernrate) auf disjunktem Testset,
    - TASK_R2 = Anteil erklaerter Zielvarianz.

VORAB FESTGENAGELT (2026-11-03, vor jedem Lauf):
    KILL:  corr(MC, TASK_R2) < +0.4  => Ressource = Reservoir-Folklore,
           Feuerplan F24-F28 verliert Deutungs-Kraft, Pivot auf
           deskriptive Dynamik.
    PASS:  corr(MC, TASK_R2) >= +0.4 => Ressource ist task-validiert
           im optimizer-freien Sinn; der F30-v2-Kill wird dem
           Optimierer-Kanal zugeschrieben (E3b-Staerkung: der
           Antagonist VERDECKT die Ressource, wenn man durch ihn misst).
    Sekundaer (registriert): corr(NLM, TASK_R2) erwartet positiv;
    corr(ben, TASK_R2) erwartet negativ (Chaos schadet).

Beide Ausgaenge sind Erkenntnis. Das ist der letzte Null-Test auf
diesem Konstrukt — ein drittes Messgeraet gibt es nicht, und braucht
es nicht: Ridge ist die untere Schranke dessen, was der Zustand
linear traegt.

## 4. Konsequenzen (sofort)

- F30 v2 Urteil bleibt aktenkundig: Optimierer-Adaption ist als
  Validierungs-Kanal tot (das ist selbst ein Befund!).
- Interpretationskraft von F24/F26/F25/F27/F28: EINGEFROREN bis
  F30.3 entschieden hat.
- Feuerreihenfolge: F30.3 SOFORT (billig, ~5 Min), danach
  Weiterfeuerung nur bei PASS.
