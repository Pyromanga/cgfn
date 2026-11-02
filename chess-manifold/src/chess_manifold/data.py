"""Datengenerierung: geseedete Random-Selfplay-Partien (reproduzierbar).

Annahme (dokumentiert im Report): Random-Selfplay erzeugt eine breite, unbalancierte
Stellungsverteilung. Remis-Partien sind selten; die Beckenlabels stammen daher primaer
aus der Engine-Gewinnwahrscheinlichkeit, nicht aus dem Spielausgang.
"""
import random

import chess

from .config import Config

START_FEN = chess.STARTING_FEN


def outcome_str(board: chess.Board, capped: bool) -> str:
    if capped and board.outcome() is None:
        return "cap"
    oc = board.outcome()
    if oc is None:
        return "cap"
    if oc.winner is chess.WHITE:
        return "1-0"
    if oc.winner is chess.BLACK:
        return "0-1"
    return "1/2"


def generate_positions(cfg: Config):
    """Liefert Liste von dicts: fen, ply, outcome, early_middlegame, is_start."""
    rng = random.Random(cfg.seed)
    rows = [{"fen": START_FEN, "ply": 0, "outcome": None,
             "early_middlegame": False, "is_start": True}]
    for _ in range(cfg.n_games):
        board = chess.Board()
        sampled = 0
        ply = 0
        while ply < cfg.max_game_plies:
            if board.is_game_over(claim_draw=True):
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
            ply += 1
            if ply >= cfg.min_ply and sampled < cfg.plies_per_game:
                if (ply % 4) == 0:
                    rows.append({
                        "fen": board.fen(),
                        "ply": ply,
                        "outcome": None,
                        "early_middlegame": cfg.early_middlegame[0] <= ply <= cfg.early_middlegame[1],
                        "is_start": False,
                    })
                    sampled += 1
        oc = outcome_str(board, capped=ply >= cfg.max_game_plies)
        for r in (rows[-sampled:] if sampled else []):
            r["outcome"] = oc
    return rows
