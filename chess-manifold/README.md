# chess-manifold

**Falsifizierbarer Hypothesentest: Ist der Schach-Spielraum eine Drei-Becken-Attraktorstruktur,
deren topologische Geometrie den Spielausgang ohne Baumsuche vorhersagt?**

## Hypothese (vorregistriert)

H0 (Null): Die topologische Position der Startstellung im gelernten Manifold C
(Contrastive-Learning-Einbettung von Stellungsfeatures) hat **keine** praediktive Kraft
fuer die Beckenzuordnung ueber das hinaus, was eine flache Stockfish-Bewertung (depth=1) liefert.

H1: Die Startstellung liegt signifikant naeher am Remis-Becken als an den Sieg-Becken
(Geodaetendistanz, Bootstrap-CI 95 %, p < 0.05, einseitig), UND die Einbettungs-k-NN-
Klassifikation erreicht eine Kohaerenz in der Groessenordnung der Literaturwerte (~91 %).

**Entscheidungsregel:** H1 nur gestuetzt, wenn (a) Bootstrap-CI von d(start,draw) - d(start,sieg)
durchgehend negativ ist UND (b) topologische Genauigkeit > Baseline (depth=1).
Andernfalls: widerlegt (bei klarer Diskrepanz) oder unentschieden (bei breiten CIs / fehlender Engine).

## Pipeline

`LT`
data (geseedetes Random-Selfplay, python-chess)
  -> Stockfish-Evaluation (depth 12 Supervision, depth 1 Baseline)  [apt-install stockfish]
  -> Features: klassisch (Material, Bauernstruktur, Koenig, Raum, Mobilitaet)
              + topologisch (Vietoris-Rips-PH der Figuren-Interaktionspunkte, ripser)
  -> SupCon-Encoder (PyTorch Lightning, Embedding dim 16)
  -> PH je Becken (Betti-Kurven), Isomap-Geodaeten, Kohaerenz (LOO-k-NN),
     Bootstrap-Test, Gegenbeispielsuche
  -> results/report.json + summary.json + plots/
`LT`

## Nutzung

`LT`bash
make install   # Abhaengigkeiten
make test      # Smoke-Tests
make smoke     # ~300 Stellungen, ~5 min
make full      # ~10.000 Stellungen (Aufgabenspezifikation), ~30-60 min
`LT`

CI: ``.github/workflows/chess-manifold.yml`` (Repo-Root, cgfn) — Push mit Aenderungen
unter `chess-manifold/**` oder Marker `.trigger/chess-manifold` = Smoke-Run,
`workflow_dispatch` mit Profil `smoke|full`. Ergebnisse werden als Artefakt
hochgeladen und nach `chess-manifold/results/` zurueckcommittiert.

## Verzeichnisse

- `src/chess_manifold/` — Pipeline-Code
- `tests/` — Smoke-Tests (rot = keine Pipeline)
- `data/`, `results/` — Laufzeit-Artefakte (CSV/JSON/PNG)
- `notebooks/ANALYSIS.md` — Analyse-Notizen
- `REPORT.md` — Technischer Bericht (Ergebnisse siehe results/summary.json)

## Dokumentierte Annahmen (siehe REPORT.md)

1. Random-Selfplay statt Lichess-Datenbank: voll reproduzierbar, aber
   Remis-Partien selten -> Beckenlabels aus Engine-Gewinnwahrscheinlichkeit, nicht Spielausgang.
2. cp->Gewinnwahrscheinlichkeit via Sigmoid (Lambda=0.00368208), +-2000 cp geclippt.
3. PH auf Einbettungen pro Becken auf <=1200 Punkte geseedet subgesampelt (Laufzeit).
4. Geodaeten approximiert als diskrete kuerzeste Pfade im Isomap-k-NN-Graphen.
5. Lc0-Einbettungen optional, nicht implementiert (Engine-only-Verantwortung von Stockfish).
6. Fehlt Stockfish, laeuft die Pipeline im Material-Fallback-Modus; alle solchen
   Ergebnisse sind nur sanitiver Natur (engine_available: false im Report).
