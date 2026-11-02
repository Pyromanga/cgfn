#!/usr/bin/env python
"""C1 — Sekunden-Test der Strukturtheorie (S2/S3):

Praediktion P2: Tilt-Positionen (groesste wp-Spruenge aus Run #11/#13) haben ein
erhoehtes Forcing-Potenzial (beste Drohung gegen unterverteidigte Ziele),
quiet-draw-Positionen nicht.

Metric (reines python-chess, keine Engine, keine Wahrscheinlichkeit):
  F(x) = max ueber legalen Zuege von:
           + Capture-Gewinn (SEE-artig: gepackter Stein)
           + 3.0 pro Schachgebot, +100 bei Matt
           + fuer jedes gegnerische Ziel nach dem Zug:
               unverteidigt -> voller Wert
               unterverteidigt (Wert > billigster Angreifer) -> Wert-Differenz

Urteil: median(F_tilt) >= 2 * max(1.0, median(F_quiet)) => P2 bestanden.
"""
import argparse
import json
import statistics

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def forcing_potential(board: chess.Board) -> float:
    """Bestes erzwungenes Drohungsgewinn-Potenzial der Seite am Zug (S2-Zaehler)."""
    color = board.turn
    opp = not color
    best = 0.0
    for move in board.legal_moves:
        gain = 0.0
        if board.gives_check(move):
            gain += 3.0
        if board.is_capture(move):
            cap = board.piece_at(move.to_square)
            if cap is not None:
                gain += VAL.get(cap.piece_type, 0)
        board.push(move)
        if board.is_checkmate():
            gain += 100.0
        for sq in board.piece_map():
            if board.color_at(sq) != opp:
                continue
            if board.attackers(color, sq):
                attackers = board.attackers(color, sq)
                cheapest = min(
                    (VAL[board.piece_at(a).piece_type]
                     for a in attackers if board.piece_at(a) is not None),
                    default=0)
                target = board.piece_at(sq)
                if target is None:
                    continue
                tv = VAL[target.piece_type]
                defenders = len(board.attackers(opp, sq))
                if defenders == 0:
                    gain += tv
                elif tv > cheapest:
                    gain += tv - cheapest
        board.pop()
        if gain > best:
            best = gain
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str,
                   default="chess-manifold/results/summary.json")
    p.add_argument("--out", type=str, default="results/c1_deficit_results.txt")
    args = p.parse_args()

    with open(args.summary, encoding="utf-8") as f:
        s = json.load(f)

    tilt = s.get("dynamics", {}).get("tilt_examples", [])
    quiet = s.get("openings", {}).get("quiet_draw_games", [])
    if not tilt or not quiet:
        raise SystemExit("[e] tilt_examples oder quiet_draw_games fehlen in summary.json")

    tilt_scores = []
    tilt_rows = []
    for t in tilt:
        b = chess.Board(t["fen"])
        f_ = forcing_potential(b)
        tilt_scores.append(f_)
        tilt_rows.append((t["fen"], t.get("d_wp"), f_))

    quiet_scores = []
    quiet_rows = []
    for g in quiet:
        for tag, fen in (("first", g.get("fen_first")), ("last", g.get("fen_last"))):
            if not fen:
                continue
            b = chess.Board(fen)
            f_ = forcing_potential(b)
            quiet_scores.append(f_)
            quiet_rows.append((g.get("eco"), tag, f_))

    med_t = statistics.median(tilt_scores)
    med_q = statistics.median(quiet_scores) if quiet_scores else 0.0
    passed = med_t >= 2 * max(1.0, med_q)
    sep = (min(tilt_scores) - max(quiet_scores)) if quiet_scores else None

    lines = []
    lines.append("C1 — Sekunden-Test der Strukturtheorie (S2: Erzwingen = Zaehlsatz)")
    lines.append("=" * 72)
    lines.append(f"Tilt-Positionen (Kippkandidaten): n={len(tilt_scores)}, "
                 f"median F={med_t:.2f}, min={min(tilt_scores):.2f}, max={max(tilt_scores):.2f}")
    lines.append(f"Quiet-draw-Positionen:            n={len(quiet_scores)}, "

                 f"median F={med_q:.2f}, min={min(quiet_scores):.2f}, max={max(quiet_scores):.2f}")
    lines.append(f"Trennung (min_tilt - max_quiet):  {sep:.2f}" if sep is not None else "Trennung: n/a")
    lines.append("")
    lines.append("Tilt im Detail (FEN, d_wp, F):")
    for fen, dwp, f_ in tilt_rows:
        lines.append(f"  d_wp={dwp:+.3f}  F={f_:6.2f}  {fen}")
    lines.append("")
    lines.append("Quiet draws im Detail (ECO, first/last, F) — Top 10 nach F:")
    for eco, tag, f_ in sorted(quiet_rows, key=lambda r: -r[2])[:10]:
        lines.append(f"  {eco or '?':8s} {tag:6s} F={f_:6.2f}")
    lines.append("")
    if passed:
        lines.append("P2 BESTANDEN: Tilt-Forcing-Potenzial deutlich ueber quiet draws.")
        lines.append("=> S2 (Erzwingen = Bedrohungs-/Verteidigungs-Zaehlsatz) lebt. Naechster Schritt: P1 auf Tablebase-Daten.")
    else:
        lines.append("P2 NICHT BESTANDEN: Tilt-Positionen ohne erhoehtes Forcing-Potenzial.")
        lines.append("=> S2 in dieser Form FALSCH — Theorie begraben, Dokumentation in docs/schach_struktur_theorie.md.")
    text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
