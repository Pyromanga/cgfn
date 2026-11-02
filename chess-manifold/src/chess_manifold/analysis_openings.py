"""Öffnungsanalyse für Remis-Ursachen (Jonny): "Ich will die 16.2 % quiet draws
sehen — sind ALLE Eröffnungen dabei? Allein der erste Move kann alles entscheiden."

Leistet drei Dinge (auf Basis der gesampelten Plies, gleiche Aufloesung wie
analysis_dynamics):
1. Per-Partie-Liste aller quiet draws (game_id, ECO, Eröffnung, erster Move,
   Elo, erste/letzte gesampelte FEN) — die 16.2 % werden sichtbar.
2. Öffnungsabdeckung: wie viele Partien, ECO-Codes, erste Zuege insgesamt —
   und wie die Verteilung quiet/squandered je erstem Zug / ECO-Gruppe aussieht.
3. Robustheit: quiet-Rate je ECO-Hauptgruppe (A/B/C/D/E) und je erstem Zug —
   ist "quiet" ein Artefakt bestimmter Eröffnungen oder stabil ueber alle?
"""
import numpy as np
import pandas as pd


def _gid(v):
    """game_id als int wenn moeglich, sonst unveraendert (Lichess-IDs koennen Strings sein)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def _per_game(df: pd.DataFrame, cfg) -> pd.DataFrame:
    g = df.groupby("game_id")
    per = pd.DataFrame({
        "outcome": g["outcome"].first(),
        "max_wp": g["wp_engine"].max(),
        "min_wp": g["wp_engine"].min(),
        "n_sampled": g.size(),
        "first_move": g["first_move"].first() if "first_move" in df else None,
        "eco": g["eco"].first() if "eco" in df else None,
        "opening": g["opening"].first() if "opening" in df else None,
    })
    per["elo_min"] = g.apply(
        lambda x: np.nanmin([x["white_elo"].min(), x["black_elo"].min()])
        if "white_elo" in x else np.nan, include_groups=False)
    per["fen_first"] = g["fen"].first()
    per["fen_last"] = g["fen"].last()
    # game_id war nur der Groupby-Index (Run #12: KeyError 'game_id') — als Spalte materialisieren
    per = per.reset_index()
    quiet = ((per["outcome"] == "1/2-1/2") & (per["max_wp"] <= cfg.wp_white)
             & (per["min_wp"] >= cfg.wp_black))
    per["quiet"] = quiet
    return per


def _first_move_table(per: pd.DataFrame) -> list:
    out = []
    for mv, sub in per.groupby("first_move"):
        draws = sub[sub["outcome"] == "1/2-1/2"]
        out.append({
            "first_move": str(mv), "n_games": int(len(sub)),
            "pct_draws": float(100 * np.mean(sub["outcome"] == "1/2-1/2")),
            "n_draws": int(len(draws)),
            "n_quiet": int(draws["quiet"].sum()) if len(draws) else 0,
            "quiet_pct_of_draws": float(100 * draws["quiet"].mean()) if len(draws) else None,
        })
    return sorted(out, key=lambda r: -r["n_games"])


def _eco_table(per: pd.DataFrame) -> list:
    per = per.copy()
    per["eco_group"] = per["eco"].fillna("?").astype(str).str[0]
    out = []
    for grp, sub in per.groupby("eco_group"):
        draws = sub[sub["outcome"] == "1/2-1/2"]
        out.append({
            "eco_group": str(grp), "n_games": int(len(sub)),
            "n_draws": int(len(draws)),
            "quiet_pct_of_draws": float(100 * draws["quiet"].mean()) if len(draws) else None,
        })
    return sorted(out, key=lambda r: -r["n_games"])


def opening_stats(df: pd.DataFrame, cfg) -> dict:
    """Quiet-Draws sichtbar machen + Öffnungsabdeckung pruefen."""
    for col in ("outcome", "game_id"):
        if col not in df:
            return {"skipped": f"keine {col}-Spalte (Selfplay)"}
    d = df.dropna(subset=["game_id"])
    if d["outcome"].isna().all() or len(d) == 0:
        return {"skipped": "keine Spielausgaenge"}
    per = _per_game(d, cfg)
    draws = per[per["outcome"] == "1/2-1/2"]
    res = {"n_games": int(len(per)), "n_draws": int(len(draws)),
           "n_quiet_draws": int(draws["quiet"].sum())}
    has_opening = "eco" in per and per["eco"].notna().any()
    if not has_opening:
        res["skipped_openings"] = "keine ECO-Daten im Datensatz"
    # 1) Alle quiet draws einzeln (die will Jonny sehen)
    q = draws[draws["quiet"]].sort_values("game_id")
    quiet_games = []
    for _, r in q.iterrows():
        quiet_games.append({
            "game_id": _gid(r["game_id"]),
            "eco": r.get("eco"), "opening": r.get("opening"),
            "first_move": r.get("first_move"),
            "elo_min": None if pd.isna(r["elo_min"]) else int(r["elo_min"]),
            "n_sampled": int(r["n_sampled"]),
            "max_wp": float(r["max_wp"]), "min_wp": float(r["min_wp"]),
            "fen_first": r["fen_first"], "fen_last": r["fen_last"]})
    res["quiet_draw_games"] = quiet_games
    if has_opening:
        # 2) Abdeckung
        res["n_unique_eco"] = int(per["eco"].dropna().nunique())
        res["n_unique_openings"] = int(per["opening"].dropna().nunique())
        res["n_unique_first_moves"] = int(per["first_move"].dropna().nunique())
        # 3) Robustheit je erstem Zug / ECO-Gruppe
        res["by_first_move"] = _first_move_table(per)
        res["by_eco_group"] = _eco_table(per)
        # Quiet-Rate ohne die jeweiligen Top-Eroeffnungen (Stabilitaetscheck)
        for mv in ("e4", "d4", "Nf3", "c4"):
            sub = draws[draws["first_move"] != mv]
            if len(sub):
                res[f"quiet_pct_excl_{mv}"] = float(100 * sub["quiet"].mean())
    return res
