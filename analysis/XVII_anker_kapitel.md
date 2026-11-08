

---

# Nachtrag 24 (2026-11-08): T17-REGISTRIERUNG — s50-EXTRAPOLATION BEI H=192 (genagelt VOR dem Lauf). Plus: Nutzer-These zur Research-Taste als offene Design-Frage registriert.

## Nutzer-These (registriert als offene Design-Frage, nicht als Beschuss-These)
Der Kapitaen haelt "Research Taste bleibt beim Menschen" fuer den
wahrscheinlichen FLASCHENHALS, nicht fuer ein Feature (Widerlegung
meiner Einordnung aus der Recon-Diskussion). Registriert fuer P4/L5:
TASTE AUTOMATISIEREN ist ein spaeter eigener Stein (Generator-Regeln
fuer "was ist eine wertvolle Frage"), nicht ein Tabu. Die Recon
unterstuetzt die Sorge: Die Literatur nennt research taste als eine
der drei offenen Kernluecken autonomer Wissenschaft. Vorbehalt
meinerseits, ehrlich: Taste ohne Verlust-Verdikt-Signal ist nicht
trainierbar — genau das liefert aber unser Stack ab P3. Offen.

## T17-THESE (P-SUBSQRT-EXTRAPOLATION, die eine These)
  Die s50(H)-Kurve wachst SUB-SQRT WEITER: ratio2 = s50(192)/s50(96)
  liegt strikt in (1.0, 1.4142) (1.4142 = sqrt(2), das H-Verhaeltnis
  2 waere sqrt-Skalierung). Basis: T16 s50(24)=2.0, s50(96)=3.5
  (ratio50=1.75 < sqrt(4)=2, Commit 731933f).

## INSTRUMENT (wortgetreu T16, Kernel-generiert wie T16)
  Contract contracts/t17.py (reine Daten), Instrument experiments/
  t17_s50_extrapolation.py, Verdikt generiert. H=192 (neu), Seeds
  170..179 (frisch), Paritaet wie T15/T16. ARME (6): REF192
  (unnnormiert, SANITY) + R192_s fuer s in {2.8, 3.5, 4.4, 5.5, 7.0}
  (REFGEOM exakt normiert). s50-Definition wie T16 genagelt (erste
  0.5-Durchquerung, linear interpoliert; nicht einklammerbar ->
  NICHT GEMESSEN -> Kernel-Alarm -> UNENTSCHEIDEN, EINE Reparatur).
  ratio2 = s50(192)/3.5 mit T16-Anker s50(96)=3.5 als registrierte
  Konstante im Contract.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  SANITY: P_lernt(REF192) >= 0.5.
  PASS (P-SUBSQRT-EXTRAPOLATION): 1.0 < ratio2 < 1.4142.
  KILL-A (SQRT-ARTIG ODER STAERKER): ratio2 >= 1.4142 — die Sub-Sqrt-
    Ordnung von T16 extrapoliert nicht; H-Skalierung naeher an sqrt(H).
  KILL-B (KEIN WEITERER SHIFT): ratio2 <= 1.0 — die Rampe stoppt.
  UNENTSCHEIDEN (Crash): s50 nicht einklammerbar (Raster verfehlt die
    Rampe: alles < 0.5 oder unterster Rand >= 0.5), Sanity gerissen,
    ndiv > 3, dmdev- oder colnorm-Verletzung. Rerun nur bei Crash;
    EINE Reparatur, dann T17-Linie tot.
  SEKUNDAER: median acc, DM, s50/sqrt(192), Laufzeit, D-Diff-Snapshot.
  Bewahrung: kein Arm wird nachjustiert.
  KOSTEN-SCHAETZUNG: ~40-55 Runner-Minuten (H=192 ist teuer: 6 Arme x
    10 Seeds; Ehrlichkeit gegenueber der frueheren "~10 min"-Schaetzung
    des Kapitaens-Chat: das war zu optimistisch, H skaliert quadratisch
    im Rollout).
  RISIKEN (vor dem Lauf): (1) s50(192) > 7.0 -> Rampe verlaesst das
  Raster nach oben -> UNENTSCHEIDEN (Richtung KILL-A, unmesbar).
  (2) s50(192) < 2.8 -> ebenso nach unten (Richtung KILL-B). (3)
  n=10-Diskretisierung: s50 unsicher +-1 Rasterpunkt, die Schwellen
  1.0/1.4142 liegen dicht — dieses Mal bewusst scharf, denn es ist
  ein EXTRAPOLATIONSTEST, kein breites Karten-Gate.
