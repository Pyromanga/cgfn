"""Synthetische Tests fuer die Opening-Statistik (quiet draws sichtbar machen)."""
import pandas as pd

from chess_manifold.analysis_openings import opening_stats


class _Cfg:
    wp_white = 0.55
    wp_black = 0.45


def _df(rows):
    return pd.DataFrame(rows)


def test_quiet_draw_listed_with_opening_and_length():
    rows = []
    # (eco, opening, first_move, maxwp, minwp, outcome, game_plies, draw_reason)
    for gid, (eco, op, mv, maxwp, minwp, outcome, plies, reason) in enumerate([
        ("C50", "Italian", "e4", 0.52, 0.48, "1/2-1/2", 40, "agreement"),
        ("D02", "London", "d4", 0.80, 0.44, "1/2-1/2", 80, "threefold_repetition"),
        ("B90", "Sicilian", "e4", 0.90, 0.30, "1-0", 60, None),
    ]):
        for ply in (4, 8):
            rows.append({"game_id": gid, "ply": ply, "outcome": outcome,
                         "wp_engine": (maxwp if ply == 8 else 0.50),
                         "eco": eco, "opening": op, "first_move": mv,
                         "white_elo": 2300, "black_elo": 2400,
                         "game_plies": plies, "draw_reason": reason,
                         "fen": f"fen{gid}_{ply}"})
    res = opening_stats(_df(rows), _Cfg())
    assert res["n_draws"] == 2 and res["n_quiet_draws"] == 1
    q = res["quiet_draw_games"][0]
    assert q["eco"] == "C50" and q["first_move"] == "e4"
    assert q["game_plies"] == 40 and q["draw_reason"] == "agreement"
    assert res["length_stats"]["mean_plies_quiet"] == 40
    assert res["longest_quiet_draws"][0]["game_plies"] == 40
    assert res["draw_reasons"]["agreement"] == 1


def test_selfplay_skipped():
    df = _df([{"game_id": None, "ply": 4, "outcome": None, "wp_engine": 0.5}])
    res = opening_stats(df, _Cfg())
    assert "skipped" in res
