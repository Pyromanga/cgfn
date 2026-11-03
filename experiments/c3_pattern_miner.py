#!/usr/bin/env python
"""C3 — Muster-Bergwerk: Ist die Wertefunktion endlich beschreibbar?

Jonnys These: "Es gibt nicht unendlich Muster — du muesstest es berechnen koennen."
Test auf exakter Wahrheit (Syzygy-Tablebase, 3-4 Mann): Sample zufaellige
Stellungen bekannter Materialklassen, hole den EXAKTEN Wert V(x) aus der
Tablebase, berechne strukturelle Features (Kantenbudget-Geometrie: Koenigs-
Distanzen, Opposition/Paritaet, Bauer-Rang) und schuerfe mit einem
Informationsgewinn-Baum die kuerzeste Regelmenge, die V reproduziert.

Killer-Demo, falls die These stimmt: Der Miner muss die bekannte Endspiel-
theorie UNBELEHRT wiederfinden (Key Squares, Opposition, Quadratregel) —
als Regeln aus Daten, nicht aus Buechern. Messbar: Genauigkeit vs.
Regelzahl vs. Mehrheits-Baseline.
"""
import argparse
import os
import random
import statistics

import chess
import chess.syzygy

MATERIAL = [("KQvK", chess.QUEEN), ("KRvK", chess.ROOK),
            ("KBvK", chess.BISHOP), ("KNvK", chess.KNIGHT),
            ("KPvK", chess.PAWN)]

PROMO = {"a": chess.A8, "b": chess.B8, "c": chess.C8, "d": chess.D8,
         "e": chess.E8, "f": chess.F8, "g": chess.G8, "h": chess.H8}


def cd(a, b):
    """Tschebyschow-Distanz."""
    return max(abs(chess.square_file(a) - chess.square_file(b)),
               abs(chess.square_rank(a) - chess.square_rank(b)))


def random_position(rng, piece_type):
    """Zufaellige legale Stellung der Materialklasse aus Sicht der starken Seite."""
    while True:
        b = chess.Board(None)
        wk = rng.randrange(64)
        bk = rng.randrange(64)
        if cd(wk, bk) < 2:
            continue
        sq = rng.randrange(64)
        if piece_type == chess.PAWN:
            sq = chess.square(rng.randrange(8), rng.randrange(1, 7))
        if sq in (wk, bk):
            continue
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(sq, chess.Piece(piece_type, chess.WHITE))
        b.turn = rng.choice((chess.WHITE, chess.BLACK))
        if not b.is_valid():
            continue
        if b.is_checkmate() or b.is_stalemate() or b.is_insufficient_material():
            continue
        return b


def features(b):
    """Struktur-Features (Kantenbudget-Geometrie, Paritaet) — stark-Seite = Weiss."""
    wk = b.king(chess.WHITE)
    bk = b.king(chess.BLACK)
    piece_sq = None
    piece_type = None
    for s in b.piece_map():
        p = b.piece_at(s)
        if p.piece_type not in (chess.KING,):
            piece_sq, piece_type = s, p.piece_type
    f = {}
    f["mover_is_strong"] = 1.0 if b.turn == chess.WHITE else 0.0
    f["dist_kings"] = float(cd(wk, bk))
    # Opposition/Paritaet: gleiche Linie/Reihe und Distanz gerade
    same_line = (chess.square_file(wk) == chess.square_file(bk)
                 or chess.square_rank(wk) == chess.square_rank(bk))
    f["opposition"] = 1.0 if (same_line and cd(wk, bk) % 2 == 0) else 0.0
    if piece_type == chess.PAWN:
        pf = chess.square_file(piece_sq)
        pr = chess.square_rank(piece_sq)
        promo = chess.square(pf, 7)
        f["pawn_rank"] = float(pr)
        f["pawn_file"] = float(abs(pf - 3.5))
        f["dist_strong_king_promo"] = float(cd(wk, promo))
        f["dist_weak_king_promo"] = float(cd(bk, promo))
        f["dist_weak_king_pawn"] = float(cd(bk, piece_sq))
        f["weak_king_before_pawn"] = 1.0 if chess.square_rank(bk) > pr else 0.0
        f["strong_king_before_pawn"] = 1.0 if chess.square_rank(wk) > pr else 0.0
    else:
        f["dist_piece_weak_king"] = float(cd(piece_sq, bk))
        f["dist_strong_king_weak_king_edge"] = float(min(
            chess.square_file(wk), 7 - chess.square_file(wk),
            chess.square_rank(wk), 7 - chess.square_rank(wk),
            chess.square_file(bk), 7 - chess.square_file(bk),
            chess.square_rank(bk), 7 - chess.square_rank(bk)))
    return f


def grow_tree(rows, depth, max_depth, min_leaf):
    """Informationsgewinn-Baum auf numerischen Features."""
    labels = [r["y"] for r in rows]
    if depth >= max_depth or len(labels) < min_leaf or len(set(labels)) == 1:
        w = sum(labels)
        return {"leaf": True, "n": len(labels), "p": w / len(labels)}
    best = None
    for k in rows[0]["x"]:
        vals = sorted(set(r["x"][k] for r in rows))
        for v in vals:
            left = [r for r in rows if r["x"][k] <= v]
            right = [r for r in rows if r["x"][k] > v]
            if not left or not right:
                continue
            gain = abs(sum(r["y"] for r in left) / len(left)
                       - sum(r["y"] for r in right) / len(right)) * len(left) * len(right)
            if best is None or gain > best[0]:
                best = (gain, k, v, left, right)
    if best is None:
        w = sum(labels)
        return {"leaf": True, "n": len(labels), "p": w / len(labels)}
    _, k, v, left, right = best
    return {"leaf": False, "feat": k, "thr": v,
            "left": grow_tree(left, depth + 1, max_depth, min_leaf),
            "right": grow_tree(right, depth + 1, max_depth, min_leaf)}


def rules_from_tree(node, path=None):
    """Baum als lesbare Regelmenge."""
    if path is None:
        path = []
    if node["leaf"]:
        return [(list(path), node["p"], node["n"])]
    out = []
    p2 = list(path) + [f"{node['feat']} <= {node['thr']:g}"]
    out += rules_from_tree(node["left"], p2)
    p3 = list(path) + [f"{node['feat']} > {node['thr']:g}"]
    out += rules_from_tree(node["right"], p3)
    return out


def predict(node, x):
    while not node["leaf"]:
        node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
    return node["p"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tablebase", type=str, default="tablebases")
    ap.add_argument("--n-per-class", type=int, default=4000)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--out", type=str, default="results/c3_pattern_mining_results.txt")
    args = ap.parse_args()
    rng = random.Random(42)

    with chess.syzygy.open_tablebase(args.tablebase) as tb:
        lines = []
        lines.append("C3 — Muster-Bergwerk: Regeln gegen EXAKTE Tablebase-Werte (3-4 Mann)")
        lines.append("=" * 74)
        for name, ptype in MATERIAL:
            rows = []
            for _ in range(args.n_per_class):
                b = random_position(rng, ptype)
                try:
                    v = tb.probe_wdl(b)
                except Exception:
                    continue
                # Weiss = starke Seite: 2 = Gewinn, -2 = Verlust, 0 = Remis, ±1 = cursed
                y = 1.0 if v == 2 else 0.0
                rows.append({"x": features(b), "y": y})
            if len(rows) < 500:
                lines.append(f"\n[{name}] nur {len(rows)} gueltige Stellungen — uebersprungen")
                continue
            base = max(sum(r["y"] for r in rows), len(rows) - sum(r["y"] for r in rows)) / len(rows)
            tree = grow_tree(rows, 0, args.max_depth, 200)
            n_rules = len([r for r in rules_from_tree(tree) if r[1] >= 0.5]) + \
                      len([r for r in rules_from_tree(tree) if r[1] < 0.5])
            acc = sum(1 for r in rows if (predict(tree, r["x"]) >= 0.5) == (r["y"] == 1)) / len(rows)
            lines.append(f"\n[{name}] n={len(rows)}, Gewinnrate={sum(r['y'] for r in rows)/len(rows):.3f}")
            lines.append(f"  Mehrheits-Baseline: {base:.4f}   REGEL-BAUM: {acc:.4f}  (Tiefe {args.max_depth}, ~{n_rules} Regeln)")
            for conds, p, n in rules_from_tree(tree):
                if n < 50:
                    continue
                win = "GEWINN" if p >= 0.5 else "REMIS/VERLUST"
                lines.append(f"    WENN {' UND '.join(conds):60s} DANN {win} (p={p:.3f}, n={n})")
        text = "\n".join(lines)
    text += "\n\nLesart: Reproduziert ein flacher Regelbaum die exakten Werte nahezu "
    text += "perfekt,\nist die Wertefunktion endlich beschreibbar (Jonnys These). "
    text += "Fuer KPvK\nerwartet man die Wiederentdeckung von Key Squares/Opposition — "
    text += "unbelehrte Regeln statt Buchwissen. Wenn nicht: Die Muster sind tiefer "
    text += "als Geometrie-Features — dann C4 mit Hypergraph-Defizit-Features."
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
