"""Lichess-Turnierdaten: Partien der Top-Klassik-Spieler via offizieller Lichess-API.

Reproduzierbarkeit: Der Download selbst ist ein Live-API-Zugriff (nicht geseedet),
aber alle geparsten Stellungen + Metadaten landen in data/positions.csv — ab dort
ist jede weitere Analyse exakt reproduzierbar (Seed 42 beim Sampling).
"""
import io
import json
import urllib.request

import chess
import chess.pgn

from .config import Config

API = "https://lichess.org/api"
HEADERS = {"User-Agent": "chess-manifold-research/0.1 (github.com/Pyromanga/cgfn)"}


def _get(url: str, accept: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={**HEADERS, "Accept": accept})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_elite_games(cfg: Config) -> str:
    """Top-N Classical-Spieler, je bis K rated Partien als PGN-Text."""
    top = json.loads(_get(f"{API}/player/top/{cfg.lichess_top_n}/classical",
                          "application/json"))
    users = [u["username"] for u in top.get("users", [])]
    print(f"    Top-Spieler: {len(users)}")
    chunks = []
    for u in users:
        url = (f"{API}/games/user/{u}"
               f"?max={cfg.lichess_games_per_user}"
               "&rated=true&perfType=classical"
               "&opening=true&clocks=false&evals=false&literate=false")
        try:
            chunks.append(_get(url, "application/x-chess-pgn").decode("utf-8", "replace"))
        except Exception as e:  # Spieler einzeln failen lassen
            print(f"    WARN: {u}: {e}")
    return "\n\n".join(chunks)


def _draw_reason(end_board: chess.Board, game) -> str:
    """Remis-Grund aus Endstellung + PGN-Header (Jonny: 'laengste' quiet draws)."""
    term = (game.headers.get("Termination") or "").lower()
    if end_board.is_stalemate():
        return "stalemate"
    if end_board.is_insufficient_material():
        return "insufficient_material"
    if end_board.can_claim_threefold_repetition():
        return "threefold_repetition"
    if end_board.can_claim_fifty_moves():
        return "fifty_moves"
    if "abandoned" in term or "timeout" in term:
        return "timeout_vs_material"
    return "agreement"


def generate_positions_lichess(cfg: Config):
    """Wie data.generate_positions, aber aus echten Elite-Turnierpartien.

    Rueckgabe: (rows, n_games). Pro Stellung: game_id, outcome (Spielausgang),
    white_elo/black_elo, eco, opening (Eröffnungsname), first_move (SAN des
    ersten Zuges) sowie game_plies (ECHTE Partielaenge in Halbzuegen) und
    draw_reason (Remis-Grund: agreement/stalemate/material/repetition/50-moves)
    — ermöglicht Laengen- und Remis-Grund-Analyse (quiet vs. squandered).
    """
    pgn_text = fetch_elite_games(cfg)
    rows = [{"fen": chess.STARTING_FEN, "ply": 0, "outcome": None,
             "early_middlegame": False, "is_start": True, "game_id": None,
             "white_elo": None, "black_elo": None,
             "eco": None, "opening": None, "first_move": None,
             "game_plies": None, "draw_reason": None}]
    n_games = 0
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        res = game.headers.get("Result")
        welo = game.headers.get("WhiteElo")
        belo = game.headers.get("BlackElo")
        try:
            welo = int(welo) if welo and welo.isdigit() else None
            belo = int(belo) if belo and belo.isdigit() else None
        except ValueError:
            welo = belo = None
        if welo is not None and belo is not None and min(welo, belo) < cfg.lichess_min_elo:
            continue
        eco = game.headers.get("ECO")
        opening = game.headers.get("Opening")
        moves = list(game.mainline_moves())
        game_plies = len(moves)
        # Endstellung fuer Remis-Grund (komplette Partie, unabhaengig vom Sampling)
        end_board = game.board()
        for mv in moves:
            end_board.push(mv)
        reason = _draw_reason(end_board, game) if res == "1/2-1/2" else None
        gid = n_games  # game_id zaehlt nur Partien mit gesampelten Stellungen
        board = game.board()
        sampled = 0
        ply = 0
        first_move = None
        for move in moves:
            if ply == 0:
                first_move = board.san(move)
            board.push(move)
            ply += 1
            if ply > cfg.max_game_plies:
                break
            if (ply >= cfg.min_ply and sampled < cfg.plies_per_game
                    and ply % 4 == 0):
                rows.append({
                    "fen": board.fen(), "ply": ply, "outcome": res,
                    "early_middlegame": cfg.early_middlegame[0] <= ply <= cfg.early_middlegame[1],
                    "is_start": False, "game_id": gid,
                    "white_elo": welo, "black_elo": belo,
                    "eco": eco, "opening": opening, "first_move": first_move,
                    "game_plies": game_plies, "draw_reason": reason,
                })
                sampled += 1
        if sampled:
            n_games += 1
    if len(rows) < 500:
        raise RuntimeError(
            f"Nur {len(rows)} Positionen aus Lichess-API erhalten — Abbruch "
            "(Netzwerk/API-Limit pruefen).")
    return rows, n_games
