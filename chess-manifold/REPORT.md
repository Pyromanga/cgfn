undefined
## 7.6 Längen-Statistik & längste quiet draws (Run #14, Profil lichess, 2026-11-02)

**Setup:** Run #14 (Commit 52e2a4c1), 418 Partien, 112 Remis, davon **20 quiet draws**.
Neu in summary.json: `game_plies` und `draw_reason` je quiet-draw-Partie
(`openings.quiet_draw_games`) sowie `openings.length_stats`.

### Längen-Statistik (game_plies)

- **quiet draws: mean = 68.05, median = 62.5** (n = 20)
- **squandered draws: mean = 104.12** (n = 92)
- alle Remis: mean = 97.68 (n = 112); alle Partien: mean = 88.94 (n = 418)

**Quiet draws sind SYSTEMATISCH KÜRZER als squandered draws** — im Mittel um ~36
Plies (68 vs. 104, d.h. quiet rund 34 % kürzer) und auch kürzer als der Schnitt
aller Partien (89). Struktur passt: Squandered draws brauchen erst den Ausstieg
aus dem Remis-Becken (Vorsprung >= Schwelle), dann die Rückkehr plus Endspiel-
abwicklung — ein langer Umweg. Quiet draws bleiben von Anfang an im Gleichgewicht
und enden oft früh per Absprache oder Wiederholung.

### Die 5 längsten quiet draws

| # | game_id | Eröffnung | Plies | Remis-Grund | Elo_min |
|---|---------|-----------|-------|-------------|---------|
| 1 | 338 | C47 Four Knights Game: Scotch Variation Accepted | **167** | insufficient_material | 2369 |
| 2 | 376 | B20 Sicilian Defense: Mengarini Variation | 143 | threefold_repetition | 2241 |
| 3 | 102 | A46 Indian Defense: Knights Variation | 127 | stalemate | 2405 |
| 4 | 258 | B54 Sicilian Defense: Modern Variations, Main Line | 107 | threefold_repetition | 2274 |
| 5 | 269 | D00 Queen's Pawn: Accelerated London System, Steinitz Countergambit | 98 | threefold_repetition | 2430 |

### Remis-Gründe: alle Remis vs. nur quiet draws

- **Alle 112 Remis:** agreement 55, threefold_repetition 37, insufficient_material 13, stalemate 4, fifty_moves 3
- **Nur die 20 quiet draws:** agreement 11, threefold_repetition 7, stalemate 1, insufficient_material 1

Bei quiet draws dominieren Absprache (55 %) und Wiederholung (35 %) anteilig noch
stärker als bei allen Remis (49 % / 33 %). Die "harten" Regel-Remis (insufficient_
material, stalemate, fifty_moves) sind fast ausschließlich ein Phänomen der squandered
draws — einzige quiet-Ausnahmen: game 338 (167 Plies, materialloses Endspiel) und
game 102 (Patt nach 127 Plies).

### Lesart

Die längste Gleichgewichts-Remispartie (C47 Vier-Springer-Schotte, 167 Plies) zeigt:
Auch ein perfektes Gleichgewicht kann extrem lange dauern — Weiss erreichte in
keinem der 12 gesampelten Plies je die Becken-Schwelle (max_wp = 0.528), das
Material schmilzt trotzdem bis zur Unentscheidbarkeit. Kurze quiet draws (< 40
Plies) sind dagegen fast alle agreement-Remis. Die Zugzahl allein verrät also
nichts über quiet vs. squandered — entscheidend ist die Trajektorie, nicht die Länge.
