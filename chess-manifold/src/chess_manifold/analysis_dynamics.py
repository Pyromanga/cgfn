"""Dynamikanalyse: Partien als Trajektorien + Remis-Ursachen-Test.

Zwei Thesen werden getestet:
(1) Gradientenfluss-These: Partien bewegen sich im Schnitt bergab im
    Evaluationspotential V(x) = wp_engine (Drift-Ausrichtung via
    k-NN-geschaetztem lokalen Gradient im Einbettungsraum).
(2) Remis-Ursachen-These (Jonny): "Remis = Folge unperfekter Moves".
    Test: Wurde in Remis-Partien ein Becken verlassen (wp > 0.55 oder < 0.45)
    und der Vorspruch anschliessend verspielt (squandered draw), oder blieb
    die Partie durchgehend im Remis-Becken (quiet draw)? Letzteres spricht
    gegen "Remis als Fehlerfolge" und fuer Remis als natuerliches Gleichgewicht.
    Zusaetzlich: Konversionsraten — wie oft fuehrte wp > 55 % tatsaechlich
    zum Sieg? ("Vermeidbar" ist nicht "kostenlos".)

Hinweis: wp nur an gesampelten Plies (alle 4 Halbzuege, max plies_per_game)
— die Statistik ist ein Proxy mit dokumentierter Aufloesung.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def trajectory_pairs(game_id: np.ndarray, ply: np.ndarray) -> np.ndarray:
    """Paare (i, j) aufeinanderfolgender Stellungen innerhalb einer Partie."""
    gid = np.asarray(game_id, dtype=float)
    pl = np.asarray(ply, dtype=int)
    valid = ~np.isnan(gid)
    pairs = []
    order = np.lexsort((pl[valid], gid[valid]))
    vs = np.where(valid)[0][order]
    for a, b in zip(vs[:-1], vs[1:]):
        if gid[a] == gid[b] and pl[b] > pl[a]:
            pairs.append((int(a), int(b)))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def local_wp_gradient(emb: np.ndarray, wp: np.ndarray, k: int = 10) -> np.ndarray:
    """k-NN-Schaetzer von grad V im Einbettungsraum (V = wp_engine)."""
    k = int(min(k, len(emb) - 1))
    if k < 2:
        return np.zeros_like(emb)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb)
    idx = idx[:, 1:]
    grad = np.zeros_like(emb)
    for i in range(len(emb)):
        diff = emb[idx[i]] - emb[i]
        dw = wp[idx[i]] - wp[i]
        m = np.linalg.norm(diff, axis=1) > 1e-9
        if int(m.sum()) >= 2:
            g, *_ = np.linalg.lstsq(diff[m], dw[m], rcond=None)
            grad[i] = g
    return grad


def drift_stats(emb, wp, pairs, n_bootstrap: int = 500, seed: int = 42):
    """cos(displacement, -grad V): bergab im Potential?"""
    if len(pairs) == 0:
        return {"skipped": "keine Trajektorien"}
    grad = local_wp_gradient(emb, wp)
    D = emb[pairs[:, 1]] - emb[pairs[:, 0]]
    G = grad[pairs[:, 0]]
    nd = np.linalg.norm(D, axis=1)
    ng = np.linalg.norm(G, axis=1)
    m = (nd > 1e-9) & (ng > 1e-9)
    if int(m.sum()) < 10:
        return {"skipped": "zu wenige gueltige Schritte", "n_steps": int(m.sum())}
    cos = np.sum(D[m] * -G[m], axis=1) / (nd[m] * ng[m])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        s = rng.integers(0, len(cos), len(cos))
        means.append(float(np.mean(cos[s])))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n_steps": int(m.sum()),
            "drift_alignment_cos_mean": float(np.mean(cos)),
            "drift_alignment_ci95": [float(lo), float(hi)],
            "frac_downhill": float(np.mean(cos > 0))}


def draw_cause_stats(df: pd.DataFrame, cfg) -> dict:
    """Remis-Ursachen-Test: squandered vs. quiet draws + Konversionsraten."""
    if "outcome" not in df or "game_id" not in df:
        return {"skipped": "keine outcome-/game_id-Spalten (Selfplay)"}
    d = df.dropna(subset=["game_id"])
    if d["outcome"].isna().all() or len(d) == 0:
        return {"skipped": "keine Spielausgaenge"}
    g = d.groupby("game_id")
    per_game = pd.DataFrame({
        "outcome": g["outcome"].first(),
        "max_wp": g["wp_engine"].max(),
        "min_wp": g["wp_engine"].min(),
        "elo_min": g.apply(lambda x: np.nanmin([x["white_elo"].min(),
                                                x["black_elo"].min()]), include_groups=False),
        "game_id": g.size().index,
    })
    res = {}
    for oc, tag in (("1-0", "white_wins"), ("0-1", "black_wins"),
                    ("1/2-1/2", "draws")):
        sub = per_game[per_game["outcome"] == oc]
        if len(sub) == 0:
            res[tag] = {"n_games": 0}
            continue
        res[tag] = {
            "n_games": int(len(sub)),
            "frac_white_was_better": float(np.mean(sub["max_wp"] > cfg.wp_white)),
            "frac_black_was_better": float(np.mean(sub["min_wp"] < cfg.wp_black)),
            "frac_left_draw_basin": float(np.mean((sub["max_wp"] > cfg.wp_white) |
                                                  (sub["min_wp"] < cfg.wp_black))),
        }
    draws = per_game[per_game["outcome"] == "1/2-1/2"]
    if len(draws):
        quiet = (draws["max_wp"] <= cfg.wp_white) & (draws["min_wp"] >= cfg.wp_black)
        white_squandered = draws["max_wp"] > cfg.wp_white
        black_squandered = draws["min_wp"] < cfg.wp_black
        res["draw_classification"] = {
            "n_draws": int(len(draws)),
            "quiet_draws_pct": float(100 * np.mean(quiet)),
            "white_squandered_pct": float(100 * np.mean(white_squandered)),
            "black_squandered_pct": float(100 * np.mean(black_squandered)),
            "note": ("quiet = nie aus dem Remis-Becken hinausgekommen ( sampled plies); "
                     "squandered = Vorsprung >= Becken-Schwelle erreicht, Partie endete 1/2-1/2"),
        }
    # Konversionsraten: wp > 0.55 irgendwann -> wie oft Sieg Weiß?
    had_white_edge = per_game["max_wp"] > cfg.wp_white
    if int(had_white_edge.sum()) > 0:
        conv = per_game[had_white_edge]["outcome"]
        res["white_edge_conversion"] = {
            "n_games_with_white_edge": int(had_white_edge.sum()),
            "pct_won": float(100 * np.mean(conv == "1-0")),
            "pct_draw": float(100 * np.mean(conv == "1/2-1/2")),
            "pct_lost": float(100 * np.mean(conv == "0-1")),
        }
    # Elo-Temperatur (k-ramers-artig): Schrittrauschen vs. Elo
    if "white_elo" in d:
        steps = d.sort_values(["game_id", "ply"]).groupby("game_id")["wp_engine"]
        for lo, hi, tag in ((2200, 2400, "elo_lt2400"), (2400, 2600, "elo_2400_2600"),
                            (2600, 3000, "elo_ge2600")):
            sel = per_game[(per_game["elo_min"] >= lo) & (per_game["elo_min"] < hi)]
            if len(sel) > 3:
                sub_d = d[d["game_id"].isin(sel["game_id"])]
                sp = sub_d.sort_values(["game_id", "ply"]).groupby("game_id")["wp_engine"]
                jump = (sp.max() - sp.min()).abs().mean()
                res["wp_swing_mean_" + tag] = float(jump)
    return res


def tilt_candidates(df: pd.DataFrame, pairs: np.ndarray, top_k: int = 10) -> list:
    """Morse-artige Kipp-Punkte: groesste wp-Aenderungen zwischen aufeinander-
    folgenden Stellungen (Kandidaten fuer kritische Punkte der Partie)."""
    if len(pairs) == 0 or "wp_engine" not in df:
        return []
    wp = df["wp_engine"].to_numpy()
    dwp = np.abs(wp[pairs[:, 1]] - wp[pairs[:, 0]])
    order = np.argsort(-dwp)[:top_k]
    out = []
    for i in order:
        a = pairs[i, 0]
        out.append({"fen": str(df["fen"].iloc[a]), "ply": int(df["ply"].iloc[a]),
                    "d_wp": float(wp[pairs[i, 1]] - wp[pairs[i, 0]])})
    return out


def run_dynamics(cfg, df: pd.DataFrame, emb: np.ndarray) -> dict:
    report = {}
    pairs = trajectory_pairs(df["game_id"].to_numpy() if "game_id" in df else np.full(len(df), np.nan),
                             df["ply"].to_numpy())
    report["n_trajectories_steps"] = int(len(pairs))
    report["drift"] = drift_stats(emb, df["wp_engine"].to_numpy(), pairs,
                                  n_bootstrap=min(cfg.n_bootstrap, 500),
                                  seed=cfg.seed)
    report["draw_cause"] = draw_cause_stats(df, cfg)
    report["tilt_examples"] = tilt_candidates(df, pairs)
    return report
