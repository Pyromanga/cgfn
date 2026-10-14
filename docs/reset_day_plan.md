# Kontingent-Reset-Day — Feuerplan (2026-11-01)

**Lage (2026-10-14, spaet):** Actions-Kontingent bei 100%. Alle Crons
aus, smoke auf Marker gepinnt, Gates in allen Frontier-Workflows.
Ab 1.11. wieder 2000 Minuten — das ist der Feuerplan.

**Update 14.10. ( Nacht):** F24 UND F25 sind jetzt IMPLEMENTIERT und
feuerbereit (F24: MC vs kappa bei festem getriebenem Benettin; F25:
Reverberating-Mass-Batterie mit MLE/KS-Exponenten, Quasi-Kritikalitaets-
Check, lam-Sweep gegen die Latent-Variablen-Beiprodukt-Falle, hidden-
Sweep). Beide brauchen NUR noch einen Marker-Push. Kein Trigger
gesetzt. F26 (strukturierte Nicht-Normalitaet: Ketten vs Rauschen im
Schur-Delta) ist entworfen (Tiefenanalyse V) und wartet auf F24.

## Prioritaeten-Liste (Batterie-Reihenfolge)

1. **F24 Memory-Ressource** (~30-40 Min) — ERSTER LAUF. Reines
   Reservoir, kein Training, unabhaengig vom Homeostat-Strang.
   Kollisionstest gegen Ganguli/Orhan/Schuecker. Marker: `.trigger/f24`
2. **F21 Homeostat** (~80 Min) — ARBEITSPFERD, erster Bau-Test.
   Marker: `.trigger/f21`
3. **F22 Homeostat v2** (~60 Min) — Gram-Proxy-Regler. Nur feuern,
   wenn F21 positiv.
4. **F23 Marathon** (~90 Min) — 10 Task-Bloecke, Retentions-Kurve.
   Nur feuern, wenn F21 oder F22 mind. 1 positiven Regler-Arm zeigt.
5. **F25 Reverberation** (~40 Min) — IMPLEMENTIERT & FEUERBEREIT.
   Batterie: m (2 Schaetzer), tauS/tauT (MLE+KS), gamma (2-fach),
   lam-Sweep, hidden-Sweep, MC am Gitter. Marker: `.trigger/f25`
   (Unabhaengig von F21-F23; gut als zweiter Lauf nach F24, weil
   gleiche designed-Model-Infrastruktur — Befunde kreuzen sich.)
6. **F11.2 Statistik-Aufstockung** (~40 Min) — Artefakt-Claim
   braucht die n-Schaerfe.
7. **F26 (NEU, nach F24)** — strukturierte Nicht-Normalitaet:
   Band-Kette vs unstrukturiert im Schur-Delta bei festem kappa
   und lambda. Design steht in Tiefenanalyse V; implementieren erst
   nach F24-Ergebnis (Freigabe-Logik: nur wenn kappa zahlt).
8. Reserven: ~1550 Min fuer Iterationen, 8-Seed-Haertungen,
   Statistik-Disziplin (F20b-Lehre).

## Regeln (aus 6 Tagen Lehrgeld)
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
- **Push-Log sparsam**: ein Commit pro Vorhaben; .trigger-Aenderung
  NUR beim gezielten Feuern.
