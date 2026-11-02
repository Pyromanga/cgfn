"""Öffnungsanalyse für Remis-Ursachen (Jonny): "Ich will die 16.2 % quiet draws
sehen — sind ALLE Eröffnungen dabei? Allein der erste Move kann alles entscheiden."

Leistet vier Dinge (auf Basis der gesampelten Plies, gleiche Aufloesung wie
analysis_dynamics):
1. Per-Partie-Liste aller quiet draws (game_id, ECO, Eröffnung, erster Move,
   Elo, ECHTE Partielaenge game_plies, Remis-Grund, erste/letzte gesampelte FEN).
2. Öffnungsabdeckung: wie viele Partien, ECO-Codes, erste Zuege insgesamt —
   und wie die Verteilung quiet/squandered je erstem Zug / ECO-Gruppe aussieht.
3. Robustheit: quiet-Rate je ECO-Hauptgruppe (A/B/C/D/E) und je erstem Zug —
   ist "quiet" ein Artefakt bestimmter Eröffnungen oder stabil ueber alle?
4. Laengen-Statistik (Jonny: "welches Opening geht am laengsten bis zum
   quiet draw?"): Partielänge quiet vs. squandered vs. alle + Top-5 der
   laengsten quiet draws mit Eröffnung.
"""
import numpy as np
import pandas as pd


def _gid(v):
    """game_id als int wenn moeglich, sonst unveraendert."""
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
        "game_plies": g["game_plies"].first() if "game_plies" in df else None,
        "draw_reason": g["draw_reason"].first() if "draw_reason" in df else None,
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


def _game_row(r) -> dict:
    return {
        "game_id": _gid(r["game_id"]),
        "eco": r.get("eco"), "opening": r.get("opening"),
        "first_move": r.get("first_move"),
        "elo_min": None if pd.isna(r["elo_min"]) else int(r["elo_min"]),
        "n_sampled": int(r["n_sampled"]),
        "game_plies": None if pd.isna(r.get("game_plies")) else int(r["game_plies"]),
        "draw_reason": r.get("draw_reason"),
        "max_wp": float(r["max_wp"]), "min_wp": float(r["min_wp"]),
        "fen_first": r["fen_first"], "fen_last": r["fen_last"]}


def opening_stats(df: pd.DataFrame, cfg) -> d
ict:
    """Quiet-Draws sichtbar machen + Öffnungsabdeckung + Laengen pruefen."""
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
    res["quiet_draw_games"] = [_game_row(r) for _, r in q.iterrows()]
    # 4) Laengen-Statistik: wie lange gehen quiet vs. squandered draws wirklich?
    if "game_plies" in per and per["game_plies"].notna().any():
        sq = draws[~draws["quiet"]]
        res["length_stats"] = {
            "mean_plies_quiet": float(q["game_plies"].mean()) if len(q) else None,
            "mean_plies_squandered": float(sq["game_plies"].mean()) if len(sq) else None,
            "mean_plies_all_draws": float(draws["game_plies"].mean()) if len(draws) else None,
            "mean_plies_all_games": float(per["game_plies"].mean()),
            "median_plies_quiet": float(q["game_plies"].median()) if len(q) else None,
        }
        longest = q.sort_values("game_plies", ascending=False).head(5)
        res["longest_quiet_draws"] = [
            {"game_id": _gid(r["game_id"]), "eco": r["eco"], "opening": r["opening"],
             "game_plies": int(r["game_plies"]), "draw_reason": r["draw_reason"],
             "elo_min": int(r["elo_min"]) if not pd.isna(r["elo_min"]) else None}
            for _, r in longest.iterrows()]
        if "draw_reason" in draws and draws["draw_reason"].notna().any():
   
         res["draw_reasons"] = draws["draw_reason"].value_counts().to_dict()
            res["quiet_draw_reasons"] = q["draw_reason"].value_counts().to_dict() if len(q) else {}
    if has_opening:
        # 2) Abdeckung
        res["n_unique_eco"] = int(per["eco"].dropna().nunique())
        res["n_unique_openings"] = int(per["opening"].dropna().nunique())
        res["n_unique_first_moves"] = int(per["first_move"].dropna().nunique())
        # 3) Robustheit je erstem Zug / ECO-Gruppe
        res["by_first_move"] = _first_move_table(per)
        res["by_eco_group"] = _eco_table(per)
        for mv in ("e4", "d4", "Nf3", "c4"):
            sub = draws[draws["first_move"] != mv]
            if len(sub):
                res[f"quiet_pct_excl_{mv}"] = float(100 * sub["quiet"].mean())
    # 5) Perfektions-Gradient (Jonny: "dann muesste man das perfekte Spiel
    #    berechnen koennen"): Wie nahe rueckt das Spiel dem perfekten Remis,
    #    je hoeher das Elo? Extrapolation: Bei welchem Elo geht der wp-Swing
    #    gegen null (= perfekte Partie bleibt im Remis-Becken)?
    if per["elo_min"].notna().any():
        swing = d.sort_values(["game_id", "ply"]).groupby("game_id")["wp_engine"]
        swing = (swing.max() - swing.min()).rename("wp_swing")
        pg = per.merge(swing, left_on="game_id", right_index=True, how="left")
        buckets = [(2200, 2400, "elo_2200_2400"), (2400, 2600, "elo_2400_2600"),
                   (2600, 3000, "elo_ge2600")]
        grad = []
        for lo, hi, tag in buckets:
            s = pg[(pg["elo_min"] >= lo) & (pg["elo_min"] < hi)]
            if len(s) >= 10:
                dr = s[s["outcome"] == "1/2-1/2"]
                grad.append({
                    tag: {
                        "n_games": int(len(s)),
                        "draw_rate": float(100 * np.mean(s["outcome"] == "1/2-1/2")),
                        "quiet_pct_of_draws": float(100 * dr["quiet"].mean()) if len(dr) else None,
                        "mean_wp_swing": float(s["wp_swing"].mean()),
                        "frac_games_with_white_edge": float(100 * np.mean(s["max_wp"] > cfg.wp_white)),
                        "mean_game_plies": float(s["game_plies"].mean()) if s["game_plies"].notna().any() else None,
                    }})
        res["perfection_gradient"] = grad
        # Lineare Extrapolation: wp_swing -> 0 (perfekte Partie verlaesst das
        # Remis-Becken nie). Eleganter Proxy fuer "perfektes Spiel berechenbar".
        centers, swings = [], []
        for lo, hi, tag in buckets:
            s = pg[(pg["elo_min"] >= lo) & (pg["elo_min"] < hi)]
            if len(s) >= 10:
                centers.append((lo + hi) / 2)
                swings.append(float(s["wp_swing"].mean()))
        if len(centers) >= 2:
            x = np.asarray(centers, dtype=float)
            y = np.asarray(swings, dtype=float)
            b1, b0 = np.polyfit(x, y, 1)
            elo_zero = float(-b0 / b1) if abs(b1) > 1e-12 else None
            res["perfection_extrapolation"] = {
                "slope_swing_per_elo": float(b1),
                "elo_at_zero_swing": elo_zero,
                "note": ("wp-Swing sinkt linear mit Elo; Extrapolation auf Swing=0 "
                         "gibt die 'perfekte Spielstaerke', bei der die Partie das "
                         "Remis-Becken nie verlaesst (Heuristik, kleine n)"),
            }
    return res
