"""Smoke-Tests: laufen bei jedem Pipeline-Run in CI."""
import chess
import numpy as np
import pytest

from chess_manifold.config import Config
from chess_manifold.data import generate_positions
from chess_manifold.engine import cp_to_winprob, find_stockfish, heuristic_eval_cp
from chess_manifold.features import extract_features, classical_features


def test_data_generation_valid_fens():
    cfg = Config()
    cfg.n_games = 3
    cfg.plies_per_game = 4
    rows = generate_positions(cfg)
    assert rows[0]["is_start"]
    for r in rows:
        chess.Board(r["fen"])
    assert len(rows) >= 10


def test_winprob_monotonic_and_range():
    assert cp_to_winprob(0) == pytest.approx(0.5)
    assert cp_to_winprob(1000) > cp_to_winprob(0) > cp_to_winprob(-1000)
    assert 0.0 <= cp_to_winprob(100000) <= 1.0
    assert cp_to_winprob(-5000) == pytest.approx(0.0, abs=1e-3)


def test_features_shape_and_finiteness():
    board = chess.Board()
    f = extract_features(board)
    assert len(f) >= 40
    assert all(np.isfinite(v) for v in f.values())
    assert f["material_diff"] == 0.0


def test_classical_features_symmetric_startpos():
    board = chess.Board()
    f = classical_features(board)
    assert f["material_w"] == f["material_b"]
    assert f["space_w"] == f["space_b"]


def test_topological_features_change_with_position():
    b1 = chess.Board()
    f1 = extract_features(b1)
    b1.push(chess.Move.from_uci("e2e4"))
    f2 = extract_features(b1)
    assert f1 != f2


def test_engine_availability_report():
    path = find_stockfish()
    assert path is None or isinstance(path, str)


def test_heuristic_eval_startpos():
    assert heuristic_eval_cp(chess.Board()) == 0.0
