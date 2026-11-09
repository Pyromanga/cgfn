

---

# Nachtrag 27 (2026-11-09): T18 — PASS formal, aber das reine Potenz-Gesetz KNICKT. Der Theory-Core lernt aus seinem ersten Vorhersage-Test: 14 Prozent daneben heisst nicht "bestaetigt", es heisst "fast".

## T18 (Lauf 08:03-08:20 UTC, 17 min, Bot-Commit b6cd8f4, Seeds 180..189, 7 Arme, 0 Divergenzen, Bug-Protokoll leer)
VERDICT (GENERIERT): PASS — P-GESCHLOSSENE-VORHERSAGE (s50(12) = 1.30
in [1.1, 1.9]). ABER die scharfe Vorhersage war 1.51 — das Messergebnis
liegt 14 Prozent DARUNTER, nah an der Bandunterkante. Formal PASS,
inhaltlich: Das 3-Punkt-Gesetz s50 ~ 0.546*H^0.408 ist als GLOBALES
Ein-Parameter-Gesetz GEFAELLEN.

## Die vier Punkte jetzt (T16/T17/T18)
  H= 12: s50 = 1.300 (T18)   H= 24: s50 = 2.000 (T16)
  H= 96: s50 = 3.500 (T16)   H=192: s50 = 4.675 (T17)
SCHRITT-EXPONENTEN: 12->24: 0.6215 | 24->96: 0.4037 | 96->192: 0.4176.
Der Exponent ist NICHT konstant: Die Kurve ist am kleinen H STEILER
und flacht auf ~0.41 ein. Ein Ein-Exponent war von Anfang an zu
einfach geglaubt (drei Punkte), jetzt ist es widerlegt (vier Punkte).

## Refit ueber alle vier Punkte (Material)
  (a) Reiner Potenz-Fit: s50 ~ 0.447 * H^0.450 — Residuen bis
      +-6.5 Prozent (H=24: -6.5, H=12: +5.3). Kein <1%-Gesetz.
  (b) AFFINE Form: s50 ~ 0.564 * H^0.41 - 0.176 — Residuen bis
      +-6.7 Prozent. Praktisch aequivalent zu (a) im Bereich
      [12, 192] (beide sagen z.B. H=48 bei ~2.55 voraus).
  EHRLEICHES FAZIT: Beide Formen treffen die vier Punkte auf ~5-6
  Prozent. Als VORHERSAGE-Werkzeug ist der Theory-Core brauchbar
  (Vorhersageband +-15 Prozent waere ehrlich); als EXAKTES Gesetz
  ist nichts gepinnt. Die <1%-Frage des Kapitaens (Nachtrag 26)
  ist damit negativ beantwortet: Es gibt (noch) keine genaue
  Formel, nur eine gute Naeherung mit Biegung am unteren H-Rand.

## Was der Theory-Core aus T18 gelernt hat (der eigentliche Ertrag)
1. DAS MODELL HAT EINEN VERLUST-SIGNAL-DATENPUNKT: Vorhersage 1.51,
   gemessen 1.30, Fehler 14 Prozent. Erste Zeile des kuenftigen
   Research-Taste-Trainings (Nutzer-These, Nachtrag 24): Ein Fit aus
   3 Punkten ist zu optimistisch, wenn er keine
   Unsicherheits-Schaetzung traegt. Lehre: kuenftige Vorhersagen mit
   explizitem Band UND mit Modellvergleich gegen Alternativ-Form.
2. KANDIDAT-KOVARIATE ENTDECKT (Material): DM wachst mit H (4.84 bei
   H=12 bis 29.43 bei H=192). Moeglicherweise ist s50 eine Funktion
   von DM (oder beides folgen dem Spektrum), nicht von H direkt —
   das wuerde die Biegung erklaeren und ist der analytische Angriffs-
   punkt fuer die WARUM-Frage.
3. Die Sanity-Architektur (HIGH12-Anker statt REF12-auf-der-Rampe,
   T3-Lehre) hat funktioniert: Sanity 0.60, sauber getrennt vom
   Rampen-Bericht.

## T19-REGISTRIERUNG (VORHERSAGE, genagelt VOR dem Lauf) — DER ZWEITE VORHERSAGE-TEST, diesals MIT BAND UND MODELLVERGLEICH
Design: H=48 (die Luecke in der Kurve, billig). s-Raster um die
beiden Form-Vorhersagen (beide ~2.55): R48_s fuer s in {1.9, 2.2,
2.55, 2.9, 3.3} (role sgrid), HIGH48 (s=5.5, Sanity-Anker), REF48
(Konventions-Bericht, role ref). Seeds 190..209 (20 Seeds — doppelt,
denn T18s Lehre ist Raster/Rauschen-empfindlichkeit). f64, Paritaet
wie immer, kernel-generiert (contracts/t19.py).
VORHERSAGEN (registriert, beide Formen aus Nachtrag 27):
  s50(48) ~ 2.55 (reiner Potenz-Fit: 2.55 | affin: 2.58 — die Formen
  unterscheiden sich hier praktisch nicht; der Test ist gegen die
  MESSUNG, nicht gegen eine Form gegen die andere).
REGISTRIERUNG (GENAGELT):
  SANITY: P_lernt(HIGH48) >= 0.5.
  PASS (P-CORE-VORHERSAGE-2): s50(48) in [2.05, 3.05] (+-20 Prozent
    um 2.55, breiter als T18 gelernt hat: 14%-Fehler-Plus-Rauschen).
  KILL-A (ZU KLEIN): s50(48) < 2.05 — beide Formen zu optimistisch,
    Kurve biegt staerker.
  KILL-B (ZU GROSS): s50(48) >= 3.05 — Kurve steiler als beide.
  UNENTSCHEIDEN (Crash): Raster verfehlt (alles < 0.5 oder Rand >=
    0.5), Sanity gerissen, ndiv > 6 von 20, dmdev-/colnorm-Verletzung.
    Rerun nur bei Crash; EINE Reparatur.
  SEKUNDAER: 4-Punkt-Refit mit H=48 (Bericht), REF48-pf, DM, Laufzeit.
  Bewahrung: kein Arm nachjustiert. KOSTEN: ~15-25 min (H=48,
    8 Arme x 20 Seeds).
  RISIKO (ehrlich): REF48 (0.5*sqrt(48)~3.46) sitzt wieder auf der
  Rampe — deshalb ist Sanity auf HIGH48, REF48 nur Bericht.
