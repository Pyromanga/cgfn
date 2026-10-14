# Kontingent-Reset-Day — Feuerplan (2026-11-01)

**Lage (2026-10-14):** Actions-Kontingent bei 100%. Alle Crons aus,
smoke auf Marker gepinnt, Gate-Steps eingebaut. Ab 1.11. stehen
wieder 2000 Minuten bereit — das ist der Feuerplan.

**Update 14.10. (Abend):** F24 ist JETZT IMPLEMENTIERT und
feuerbereit (experiments/f24_memory_kappa.py + frontier_f24.yml,
Gate drin, KEIN Trigger gesetzt — Kontingent!). F25-Methodik wurde
per Tiefenanalyse IV auf die Crackling-Noise-Batterie erweitert
(nicht nur Branching-Ratio m). Feuerreihenfolge unten angepasst.

## Prioritaeten-Liste (Batterie-Reihenfolge)

1. **F21 Homeostat** (~80 Min) — ARBEITSPFERD: Der erste Bau-Test.
   3 Arme x 2 Tasks x 4 Seeds + Transfer. Marker: `.trigger/f21`
2. **F22 Homeostat v2** (~60 Min) — kontinuierlicher Gram-Proxy-Regler
   vs F21-Block vs baseline + Arm D (isolierter Regler) + Kostenmessung.
   Nur FEUERN, wenn F21 positiv — sonst erst Design-Anpassung.
3. **F23 Marathon** (~90 Min) — 10 Task-Bloecke, Retentions-Kurve.
   Der AGI-relevante Showdown. Nur feuern, wenn F21 oder F22 mind. 1
   positiven Regler-Arm zeigt.
4. **F24 Memory-Ressource** (~30-40 Min) — IMPLEMENTIERT & FEUERBEREIT.
   MC gegen kappa bei festem getriebenem Benettin (-0.30 und -0.05,
   lam-Bisektion). Kollisionstest gegen Ganguli/Orhan/Schuecker:
   Vorhersage MC STEIGT mit kappa. Marker: `.trigger/f24`
   (Kann unabhaengig von F21 gefeuert werden — ist rein Reservoir,
   kein Training. Guter Kandidat fuer den ERSTEN Lauf, weil er am
   billigsten Evidenz mit dem groessten Deutungshebel liefert.)
5. **F25 Reverberating-Mass m** (~30 Min) — Lawinen/Branching auf
   getriebenen CT-RNN-Kaskaden. NEU (Tiefenanalyse IV): volle
   Crackling-Batterie (Power-Law-Exponenten, Shape-Collapse,
   Skalierungsrelation) statt nur m; Erwartung QUASI-kritisch.
   Zielbild: m-Fenster 0.94-0.998, m -> 1 mit hidden_dim.
6. **F11.2 Statistik-Aufstockung** (~40 Min) — der Artefakt-Claim
   traegt drei Paper-Saetze und braucht die n-Schaerfe.
7. Reserven: ~1600 Min fuer Iterationen, Statistik-Haertungen wie
   gelernt (F20b-Disziplin: 4 Seeds->8 Seeds, Permutations-Test),
   plus F26+ (strukturierte Nicht-Normalitaet, kappa-Fenster-Design)
   je nach F24-Ausgang.

## Regeln (aus 6 Tagen Lehrgeld)
- **Gate-Step** faengt Syntax-Bugs in 20s (steht jetzt in ALLEN Workflows).
- **Kein paralleles Feuern**: Free-Plan-Concurrency — Staffelung
  serial. Erst lesen, dann naechster Lauf.
- **Timeouts realistisch**: der Gate macht laengere Timeouts wieder
  vertretbar.
- **Task-Namen pruefen**: kontinuierlich = sinus_memory/lorenz/ou_noise.
  (Der 'copy'-Crash war ein erfundener Task-Name — nie wieder.)
- **F21-Auswertung**: C-vs-B prueft indirekt F24; Homeostat-Versagen bei stark rekurrenten Zellen (lam 1.2/1.4) ist HSE-erwartet, kein Bug.
- **Push-Log sparsam**: ein Commit pro Vorhaben, Marker nur beim
  gezielten Trigger aendern (jede .trigger-Aenderung startet den Lauf!).
- **F24 zuerst wenn F21 spaet startet**: F24 ist unabhaengig vom
  Homeostat-Strang und liefert die Literatur-Kollision (Runde IV).
