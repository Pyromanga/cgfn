"""Tests fuer den Dynamik-Schritt (synthetisch, ohne Netzwerk/Engine)."""
import numpy as np
import pandas as pd
import chess

from chess_manifold.analysis_dynamics import (
    trajectory_pairs, local_wp_gradient, draw_cause_stats, tilt_candidates)
from chess_manifold.config import Config


def test_trajectory_pairs_ordering():
    gid = np.array([0.0, 0.0, 1.0, 1.0, np.nan])
    ply = np.array([4, 8, 4, 8, 0])
    pairs = trajectory_pairs(gid, ply)
    # Start-Zeile (nan) wird ignoriert, Paare nur innerhalb Partie
    assert pairs.shape[0] == 2
    assert list(map(tuple, pairs)) == [(0, 1), (2, 3)]


def test_gradient_downhill_alignment():
    # 1-d-Emb: wp faellt mit x -> -grad zeigt +x; Drift +x ist bergab
    emb = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    wp = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    grad = local_wp_gradient(emb, wp, k=2)
    # alle Gradienten sollten negativ sein (wp steigt nicht mit x)
    assert np.all(grad[:, 0] < 0)


def test_draw_cause_quiet_vs_squandered():
    cfg = Config()
    rows = []
    # Spiel 0: quiet draw (wp bleibt in [0.45, 0.55])
    for ply, wp in ((4, 0.50), (8, 0.52)):
        rows.append({"game_id": 0, "outcome": "1/2-1/2", "ply": ply,
                     "wp_engine": wp, "fen": chess.Board().fen(),
                     "white_elo": 2500, "black_elo": 2500})
    # Spiel 1: squandered draw (wp war 0.7)
    for ply, wp in ((4, 0.70), (8, 0.50)):
        rows.append({"game_id": 1, "outcome": "1/2-1/2", "ply": ply,
                     "wp_engine": wp, "fen": chess.Board().fen(),
                     "white_elo": 2500, "black_elo": 2500})
    # Spiel 2: Weiss gewinnt (wp 0.6)
    for ply, wp in ((4, 0.60), (8, 0.90)):
        rows.append({"game_id": 2, "outcome": "1-0", "ply": ply,
                     "wp_engine": wp, "fen": chess.Board().fen(),
                     "white_elo": 2500, "black_elo": 2500})
    df = pd.DataFrame(rows)
    res = draw_cause_stats(df, cfg)
    assert res["draw_classification"]["n_draws"] == 2
    assert res["draw_classification"]["quiet_draws_pct"] == 50.0
    assert res["draw_classification"]["white_squandered_pct"] == 50.0
    assert res["white_wins"]["n_games"] == 1
    assert res["white_edge_conversion"]["pct_won"] == 50.0


def test_tilt_candidates_finds_big_jump():
    rows = []
    for gid, (wps, oc) in enumerate((((0.5, 0.9), "1-0"), ((0.5, 0.51), "1/2-1/2"))):
        for ply, wp in zip((4, 8), wps):
            rows.append({"game_id": gid, "outcome": oc, "ply": ply,
                         "wp_engine": wp, "fen": chess.Board().fen(),
                         "white_elo": 2500, "black_elo": 2500})
    df = pd.DataFrame(rows)
    pairs = trajectory_pairs(df["game_id"].to_numpy(), df["ply"].to_numpy())
    tilts = tilt_candidates(df, pairs, top_k=2)
    assert tilts[0]["d_wp"] > 0.3
