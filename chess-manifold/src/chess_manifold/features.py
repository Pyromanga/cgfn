"""Feature-Extraktion: klassische Merkmale + topologische Deskriptoren (Vietoris-Rips-PH).

Topologie: Figuren werden als Punkte (file, rank, wert*skala) in R^3 kodiert;
ripser berechnet H0/H1-Persistenz. Deskriptoren je Gruppe (Weiss, Schwarz, beide):
H0-Gesamtpersistenz, H1-Anzahl, H1-Gesamt/Max-Persistenz, H1-Persistenzentropie.
"""
import chess
import numpy as np
from ripser import ripser

PIECE_VALUES = {chess.PAWN: 1.0, chess.KNIGHT: 3.05, chess.BISHOP: 3.25,
                chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.5}
VALUE_SCALE = 0.25  # Figurenwert-Achse in Feldern


def _piece_points(board: chess.Board, color=None):
    pts = []
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None or (color is not None and pc.color != color):
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        pts.append((float(f), float(r), PIECE_VALUES[pc.piece_type] * VALUE_SCALE))
    return np.asarray(pts, dtype=float)


def _persistence_descriptors(pts) -> dict:
    out = {"h0_total": 0.0, "h1_count": 0, "h1_total": 0.0,
           "h1_max": 0.0, "h1_entropy": 0.0}
    if pts is None or len(pts) < 2:
        return out
    res = ripser(pts, maxdim=1)
    h0 = res["dgms"][0]
    finite0 = h0[np.isfinite(h0[:, 1])]
    out["h0_total"] = float(np.sum(finite0[:, 1] - finite0[:, 0])) if len(finite0) else 0.0
    h1 = res["dgms"][1]
    if len(h1):
        pers = h1[:, 1] - h1[:, 0]
        out["h1_count"] = int(len(h1))
        out["h1_total"] = float(np.sum(pers))
        out["h1_max"] = float(np.max(pers))
        p = pers / np.sum(pers)
        out["h1_entropy"] = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return out


def classical_features(board: chess.Board) -> dict:
    feats = {}
    for c, tag in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        opp = not c
        mat = sum(PIECE_VALUES[pt] * len(board.pieces(pt, c)) for pt in PIECE_VALUES)
        feats[f"material_{tag}"] = mat
        for pt, name in ((chess.PAWN, "p"), (chess.KNIGHT, "n"), (chess.BISHOP, "b"),
                         (chess.ROOK, "r"), (chess.QUEEN, "q")):
            feats[f"{name}_{tag}"] = len(board.pieces(pt, c))
        # Bauernstruktur
        files = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, c)]
        cnt = np.zeros(8, dtype=int)
        for f in files:
            cnt[f] += 1
        feats[f"doubled_{tag}"] = int(np.sum(np.maximum(cnt - 1, 0)))
        feats[f"isolated_{tag}"] = int(sum(
            1 for f in range(8) if cnt[f] > 0
            and (f == 0 or cnt[f - 1] == 0) and (f == 7 or cnt[f + 1] == 0)))
        # Freibauern (grob: kein gegnerischer Bauer blockiert die Bahn)
        passed = 0
        for sq in board.pieces(chess.PAWN, c):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            ahead = range(r + 1, 8) if c == chess.WHITE else range(0, r)
            blocked = False
            for rr in ahead:
                for df in (-1, 0, 1):
                    ff = f + df
                    if 0 <= ff < 8:
                        sq2 = chess.square(ff, rr)
                        pc2 = board.piece_at(sq2)
                        if pc2 and pc2.piece_type == chess.PAWN and pc2.color == opp:
                            blocked = True
            if not blocked:
                passed += 1
        feats[f"passed_{tag}"] = passed
        # Raum: kontrollierte Felder in gegnerischer Haelfte
        lo, hi = (4, 7) if c == chess.WHITE else (0, 3)
        feats[f"space_{tag}"] = sum(
            1 for sq in chess.SQUARES
            if board.is_attacked_by(c, sq) and lo <= chess.square_rank(sq) <= hi)
        # Koenigssicherheit: Gegnerangriffe auf Koenigsnahfeld
        ksq = board.king(c)
        if ksq is not None:
            near = [sq for sq in chess.SQUARES if chess.square_distance(sq, ksq) <= 1]
            feats[f"king_attackers_{tag}"] = sum(
                1 for sq in near if board.is_attacked_by(opp, sq) and sq != ksq)
        else:
            feats[f"king_attackers_{tag}"] = 0
    feats["material_diff"] = feats["material_w"] - feats["material_b"]
    # Mobilitaet
    feats["mobility_stm"] = float(board.legal_moves.count())
    b2 = board.copy(stack=False)
    b2.turn = not board.turn
    try:
        feats["mobility_opp"] = float(len(list(b2.legal_moves)))
    except Exception:
        feats["mobility_opp"] = 0.0
    feats["in_check"] = float(board.is_check())
    feats["castling_w"] = float(board.has_kingside_castling_rights(chess.WHITE) +
                                board.has_queenside_castling_rights(chess.WHITE))
    feats["castling_b"] = float(board.has_kingside_castling_rights(chess.BLACK) +
                                board.has_queenside_castling_rights(chess.BLACK))
    return feats


def extract_features(board: chess.Board) -> dict:
    feats = classical_features(board)
    for group, color in (("w", chess.WHITE), ("b", chess.BLACK), ("ab", None)):
        d = _persistence_descriptors(_piece_points(board, color))
        for k, v in d.items():
            feats[f"ph_{group}_{k}"] = v
    return feats


def features_matrix(boards):
    rows = [extract_features(b) for b in boards]
    keys = sorted(rows[0].keys())
    return np.asarray([[r[k] for k in keys] for r in rows], dtype=float), keys
