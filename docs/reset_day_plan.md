# Kontingent-Reset-Day — Feuerplan (2026-11-01)

**Lage (2026-10-20):** Actions-Kontingent bei 100%. Alle Crons aus,
smoke auf Marker gepinnt, Gates in allen Frontier-Workflows.
Ab 1.11. wieder ~2000 Minuten — das ist der Feuerplan.

**Update 2026-10-20 (Runde XI):** F31 (Ressourcen-Kern: DMFT-Selbst-
schluss, H2/MC-Theorie, Kappa-Zoo, Optimierer-Kokkyklus) implementiert.
Runde XI hat vier Zirkularitaeten im eigenen Messgeraet-Stapel gefunden
und geschnitten (docs/tiefenanalyse_2026-10-20_XI.md): die offene
Linearisierungs-Schleife (Homeostat regelt das falsche Signal), den
falsch etikettierten 'Kreiss'-Mass, die (lambda,kappa)-Vermischung
(Ressource = Tr(Sigma), ein observables Skalar), und EoS-Sharpness als
geschlossenen Optimierer-Kreis. F31 feuert DIREKT nach F30: erst der
Null-Test, dann das Messgeraet, mit dem F24-F28 zu lesen sind.

## Prioritaeten-Liste (Batterie-Reihenfolge, NEU)

1. **F30 Null-Test** (~25-35 Min) — ERSTER LAUF, KEINE AUSNAHME.
   Prueft vorab festgenagelt: corr(MC, Task-Speed) >= +0.4.
   Faellt das, sind die Ressourcen-Koordinaten nur Reservoir-Folklore
   und F24/F26/F25 verlieren ihre Deutungs-Kraft. Marker: '.trigger/f30'
2. **F31 Ressourcen-Kern** (~30-40 Min) — DMFT-Selbstschluss (P1/P2),
   MC/H2-Konsistenz (P3), Exzess E vs kappa-Zoo (P4), Heavy-Ball-
   Kompagnon g_opt(beta) (P5). Fuehrt das Messgeraet ein, mit dem
   F24/F25/F26/F27/F28 gelesen werden. Nur feuern, wenn F30 positiv.
   Marker: '.trigger/f31'
3. **F24 Memory-Ressource** (~30-40 Min) — MC vs kappa bei festem
   getriebenem Benettin. Marker: '.trigger/f24'
4. **F27 Optimierer-Interferenz** (~40 Min) — kappa(W)-Drift vs
   kappa(V_orth). Mit F31-P5 im Ruecken: g_opt waere die schaerfere
   Groesse — F27 laeuft trotzdem, Auswertung gegen beide lesen.
   Marker: '.trigger/f27'
5. **F28 Kausalitaets-Duell** (~40 Min) — targeted vs random Stoerung
   bei identischer Frobenius-Norm. Marker: '.trigger/f28'
6. **F26 Strukturierte Nicht-Normalitaet** (~35 Min) — Band-Kette vs
   Rauschen im Schur-Delta. Mit F31-P4: SSM-Exzess E=1 strukturell.
   Marker: '.trigger/f26'
7. **F25 Reverberation** (~50 Min, mit q-Sweep) — m-Fenster-Batterie
   inkl. P4 q-Robustheit. Marker: '.trigger/f25'
8. **F21 Homeostat** (~80 Min) — Arbeitstier. Auswertung NEU (Runde
   XI): Homeostat-v3-Zielgroesse waere Tr(Sigma_emp) — F21 liest, was
   der Regler WIRKLICH haelt gegen das, was er halten will.
   Marker: '.trigger/f21'
9. **F22 Homeostat v2** (~60 Min) — nur bei positivem F21.
10. **F23 Marathon** (~90 Min) — nur wenn F21/F22 mind. einen
    positiven Regler-Arm zeigen.
11. **F11.2 Statistik-Aufstockung** (~40 Min).
12. Reserven: ~1350 Min fuer Iterationen, 8-Seed-Haertungen,
    Statistik-Disziplin (F20b-Lehre), F32-Bau (Homeostat v3 auf
    Tr(Sigma) + DMFT-Modellpruefung) nur bei positivem F30 UND F31-P2.

## Regeln (aus 8 Tagen Lehrgeld)
- **Gate-Step** faengt Syntax-Bugs in 20s (in ALLEN Workflows; F31
  kompiliert zusätzlich src/cgfn/resource.py).
- **Kein paralleles Feuern**: serial lesen und schliessen, dann
  naechster Lauf.
- **Timeouts realistisch** (Gate macht grosse Timeouts wieder sicher).
- **Task-Namen pruefen**: sinus_memory / lorenz / ou_noise — niemals
  erfinden ('copy'-Crash-Lehre).
- **F21-Auswertung**: C-vs-B prueft indirekt F24; Homeostat-Versagen
  bei starker Rekurrenz-Erregung ist HSE-erwartet, kein Bug.
- **F25-Auswertung**: m-Beiprodukt-Falle (eLife): nur monotone
  m(lam)-Wanderung + MC-Optimum im/rechts vom Fenster zaehlt.
  P4 q-Spread <= 0.05 noetig, sonst schaetzer-verdaechtig.
- **F30-Auswertung**: vorab festgenagelt, corr >= +0.4. Kein
  Post-hoc-Verschieben der Schwelle (F20b-Lehre).
- **F31-Auswertung (NEU, Runde XI)**: P2 entscheidet, ob der Homeostat
  ein geschlossenes Modell (lambda_dmft) oder das offene Feld
  behaelt — beide Ausgaenge sind Erkenntnis. P3 ist Konsistenz-Test
  des linearen Regimes, KEINE These-Evidenz. P4: wenn kappa_T/Kreiss
  den Exzess besser vorhersagen als kappa_cond, ist der alte
  'Kreiss'-Mass-Koerper abgeloest.
- **Push-Log sparsam**: ein Commit pro Vorhaben; .trigger-Aenderung
  NUR beim gezielten Feuern.
