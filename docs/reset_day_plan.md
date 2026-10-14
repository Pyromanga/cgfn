# Kontingent-Reset-Day — Feuerplan (2026-11-01)

**Lage (2026-10-14):** Actions-Kontingent bei 100%. Alle Crons aus,
smoke auf Marker gepinnt, Gate-Steps eingebaut. Ab 1.11. stehen
wieder 2000 Minuten bereit — das ist der Feuerplan.

## Prioritaeten-Liste (Batterie-Reihenfolge)

1. **F21 Homeostat** (~80 Min) — ARBEITSPFERD: Der erste Bau-Test.
   3 Arme x 2 Tasks x 4 Seeds + Transfer. Marker: `.trigger/f21`
2. **F22 Homeostat v2** (~60 Min) — kontinuierlicher Gram-Proxy-Regler
   vs F21-Block vs baseline + Arm D (isolierter Regler) + Kostenmessung.
   Nur FEUERN, wenn F21 positiv — sonst erst Design-Anpassung.
3. **F23 Marathon** (~90 Min) — 10 Task-Bloecke, Retentions-Kurve.
   Der AGI-relevante Showdown. Nur feuern, wenn F21 oder F22 mind. 1
   positiven Regler-Arm zeigt.
4. **F11.2 Statistik-Aufstockung** (~40 Min) — der Artefakt-Claim
   traegt drei Paper-Saetze und braucht die n-Schaerfe.
5. Reserven: ~1650 Min fuer Iterationen, Statistik-Haertungen wie
   gelernt (F20b-Disziplin: 4 Seeds->8 Seeds, Permutations-Test).

## Regeln (aus 6 Tagen Lehrgeld)
- **Gate-Step** faengt Syntax-Bugs in 20s (steht jetzt in ALLEN Workflows).
- **Kein paralleles Feuern**: Free-Plan-Concurrency — Staffelung
 serial. Erst lesen, dann naechster Lauf.
- **Timeouts realistisch**: der Gate macht laengere Timeouts wieder
  vertretbar.
- **Task-Namen pruefen**: kontinuierlich = sinus_memory/lorenz/ou_noise.
  (Der 'copy'-Crash war ein erfundener Task-Name — nie wieder.)
- **Push-Log sparsam**: ein Commit pro Vorhaben, Marker nur beim
  gezielten Trigger aendern (jede .trigger-Aenderung startet den Lauf!).
