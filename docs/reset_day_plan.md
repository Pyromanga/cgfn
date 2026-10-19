# Kontingent-Reset-Day — Feuerplan (2026-11-01)

**Lage (2026-10-19):** Actions-Kontingent bei 100%. Alle Crons aus,
smoke auf Marker gepinnt, Gates in allen Frontier-Workflows.
Ab 1.11. wieder ~2000 Minuten — das ist der Feuerplan.

**Update 2026-10-19:** Batterie F24-F30 ist IMPLEMENTIERT und
feuerbereit (F24/F25/F26/F27/F28/F30 + Homeostat F21/F22 + Marathon
F23). Runde IX hat die Reihenfolge umgestellt: F30 (Null-Test) MUSS
erster Lauf sein — wenn der faellt, faellt die ganze Ressourcen-These.
F25 hat jetzt einen q-Robustheits-Sweep (P4, Wilting/Priesemann-
Subsampling-Bias) eingebaut. Tiefenanalyse X ergaenzt den
Selbstzeugungspfad (statistische Goedel-Maschine, F31-Skizze NUR
nach positivem F30).

## Prioritaeten-Liste (Batterie-Reihenfolge, NEU)

1. **F30 Null-Test** (~25-35 Min) — ERSTER LAUF, KEINE AUSNAHME.
   Prueft vorab festgenagelt: corr(MC, Task-Speed) >= +0.4.
   Faellt das, sind die Ressourcen-Koordinaten nur Reservoir-Folklore
   und F24/F26/F25 verlieren ihre Deutungs-Kraft. Marker: '.trigger/f30'
2. **F24 Memory-Ressource** (~30-40 Min) — MC vs kappa bei festem
   getriebenem Benettin. Nur feuern, wenn F30 positiv.
   Marker: '.trigger/f24'
3. **F27 Optimierer-Interferenz** (~40 Min) — kappa(W)-Drift vs
   kappa(V_orth): ist der Optimierer der E3b-Antagonist?
   Marker: '.trigger/f27'
4. **F28 Kausalitaets-Duell** (~40 Min) — targeted vs random Stoerung
   bei identischer Frobenius-Norm (vierfacher Tod, kanal-selektiv?).
   Marker: '.trigger/f28'
5. **F26 Strukturierte Nicht-Normalitaet** (~35 Min) — Band-Kette vs
   Rauschen im Schur-Delta bei festem kappa/lambda (Mamba-Pfadthese).
   Marker: '.trigger/f26'
6. **F25 Reverberation** (~50 Min, mit q-Sweep) — m-Fenster-Batterie
   inkl. P4 q-Robustheit. Marker: '.trigger/f25'
7. **F21 Homeostat** (~80 Min) — Arbeitstier, erster Bau-Test.
   Marker: '.trigger/f21'
8. **F22 Homeostat v2** (~60 Min) — Gram-Proxy-Regler. Nur feuern,
   wenn F21 positiv.
9. **F23 Marathon** (~90 Min) — 10 Task-Bloecke, Retentions-Kurve.
   Nur feuern, wenn F21/F22 mind. einen positiven Regler-Arm zeigen.
10. **F11.2 Statistik-Aufstockung** (~40 Min) — Artefakt-Claim
    braucht die n-Schaerfe.
11. Reserven: ~1400 Min fuer Iterationen, 8-Seed-Haertungen,
    Statistik-Disziplin (F20b-Lehre), F31-Bau (statistische
    Goedel-Maschine) nur bei positivem F30.

## Regeln (aus 8 Tagen Lehrgeld)
- **Gate-Step** faengt Syntax-Bugs in 20s (in ALLEN Workflows).
- **Kein paralleles Feuern**: Free-Plan-Concurrency — serial lesen
  und schliessen, dann naechster Lauf.
- **Timeouts realistisch** (Gate macht grosse Timeouts wieder sicher).
- **Task-Namen pruefen**: sinus_memory / lorenz / ou_noise — niemals
  erfinden ('copy'-Crash-Lehre).
- **F21-Auswertung**: C-vs-B prueft indirekt F24; Homeostat-Versagen
  bei starker Rekurrenz-Erregung ist HSE-erwartet, kein Bug.
- **F25-Auswertung**: m-Beiprodukt-Falle (eLife): nur monotone
  m(lam)-Wanderung + MC-Optimum im/rechts vom Fenster zaehlt als
  Regime-Koordinate. g_fit ~ g_pred = Quasi-Kritikalitaet.
  NEU: P4 q-Spread <= 0.05 noetig, sonst schaetzer-verdaechtig.
- **F30-Auswertung**: vorab festgenagelt, corr >= +0.4. Kein
  Post-hoc-Verschieben der Schwelle (F20b-Lehre: p=0.732-Kill war
  nur wertvoll, weil die Schwelle vorher stand).
- **Push-Log sparsam**: ein Commit pro Vorhaben; .trigger-Aenderung
  NUR beim gezielten Feuern.
