"""Schritt 3+4: Topologische Analyse des gelernten Manifolds + Hypothesentest.

- Persistente Homologie je Becken (Betti-Kurven, Persistenzdiagramm-Daten)
- Diskrete Geodaeten im Isomap-k-NN-Graphen (scipy.sparse.csgraph.shortest_path)
  von der Startstellung (und fruehen Mittelspielstellungen) zu den Becken
- Kohaerenzmetrik = LOO-k-NN-Beckenklassifikation im Einbettungsraum
- Bootstrap-Test: Startstellung naeher an Remis- als an Sieg-Becken?
- Gegenbeispiel-Suche: topologische Vorhersage vs. Engine-Bewertung
- Baseline: Stockfish depth=1 ("ohne Suche") gegenueber Einbettungs-k-NN
"""
import json

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import Isomap
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from .config import Config

BASINS = ("white", "draw", "black")


def basin_of(wp: float, cfg: Config) -> str:
    if wp > cfg.wp_white:
        return "white"
    if wp < cfg.wp_black:
        return "black"
    return "draw"


def betti_curve(points: np.ndarray, radii, maxdim=1):
    res = ripser(points, maxdim=maxdim)
    out = []
    for r in radii:
        row = {"radius": float(r)}
        for dim in range(maxdim + 1):
            dgm = res["dgms"][dim]
            row[f"betti{dim}"] = int(((dgm[:, 0] <= r) & (dgm[:, 1] > r)).sum())
        h1 = res["dgms"][1]
        row["h1_total_persistence"] = float(np.sum(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0
        out.append(row)
    return out


def ph_summary(points: np.ndarray, maxdim=1):
    res = ripser(points, maxdim=maxdim)
    s = {}
    for dim in range(maxdim + 1):
        dgm = res["dgms"][dim]
        finite = dgm[np.isfinite(dgm[:, 1])]
        pers = finite[:, 1] - finite[:, 0]
        s[f"ph_dim{dim}_num_finite"] = int(len(finite))
        s[f"ph_dim{dim}_total_persistence"] = float(np.sum(pers)) if len(pers) else 0.0
        s[f"ph_dim{dim}_max_persistence"] = float(np.max(pers)) if len(pers) else 0.0
    s["ph_diagrams"] = {f"dim{d}": res["dgms"][d].tolist() for d in range(maxdim + 1)}
    return s


def coherence(emb: np.ndarray, labels: np.ndarray, k: int, seed: int) -> float:
    """Kohaerenzmetrik: LOO-k-NN-Becken-Accuracy im Einbettungsraum."""
    classes, counts = np.unique(labels, return_counts=True)
    ok = classes[counts > k]
    mask = np.isin(labels, ok)
    if len(ok) < 2 or mask.sum() < 2 * k:
        return float("nan")
    clf = KNeighborsClassifier(n_neighbors=k)
    n_splits = int(min(5, min(counts[counts > k])))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, emb[mask], labels[mask], cv=cv)
    return float((pred == labels[mask]).mean())


def _fit_isomap(emb, cfg):
    nn = min(cfg.isomap_neighbors, max(3, len(emb) - 2))
    try:
        iso = Isomap(n_neighbors=nn, n_components=min(8, emb.shape[1], len(emb) - 1))
        iso.fit(emb)
        return iso
    except Exception:
        return None


def _basin_means(dist_row, labels):
    return {b: float(np.mean(dist_row[labels == b]))
            for b in BASINS if (labels == b).any()}


def run_analysis(cfg: Config, df: pd.DataFrame, emb: np.ndarray,
                 start_idx: int, return_diagrams: bool = False):
    labels = np.array([basin_of(w, cfg) for w in df["wp_engine"]])
    report = {"n_positions": int(len(df)),
              "basin_sizes": {b: int((labels == b).sum()) for b in BASINS}}
    diagrams = {}

    # --- Persistente Homologie je Becken (geseedet subgesampelt) ---
    rng = np.random.default_rng(cfg.seed)
    ph = {}
    for b in BASINS:
        pts = emb[labels == b]
        if len(pts) > cfg.ph_subsample:
            pts = pts[rng.choice(len(pts), cfg.ph_subsample, replace=False)]
        if len(pts) < 10:
            ph[b] = {"warning": "zu wenige Punkte"}
            continue
        s = ph_summary(pts, cfg.maxdim)
        s["betti_curve"] = betti_curve(pts, cfg.betti_radii, cfg.maxdim)
        ph[b] = s
        diagrams[b] = s.pop("ph_diagrams")
    report["persistent_homology"] = ph

    # --- Kohaerenz ---
    report["coherence_knn_accuracy"] = coherence(emb, labels, cfg.knn_k, cfg.seed)

    # --- Diskrete Geodaeten im Isomap-k-NN-Graphen ---
    iso = _fit_isomap(emb, cfg)
    centroids = {b: emb[labels == b].mean(axis=0)
                 for b in BASINS if (labels == b).any()}
    d_start = None

    if iso is not None:
        graph = iso.graph_
        d_start = shortest_path(graph, directed=False, indices=[start_idx])[0]
        if not np.all(np.isfinite(d_start)):
            eucl = np.linalg.norm(emb - emb[start_idx], axis=1)
            d_start = np.where(np.isfinite(d_start), d_start, 100.0 * eucl.max() + eucl)
        geo_start = _basin_means(d_start, labels)
        early_idx = np.where(df["early_middlegame"].to_numpy(dtype=bool))[0]
        if len(early_idx) > 200:
            early_idx = rng.choice(early_idx, 200, replace=False)
        if len(early_idx) > 0:
            d_early = shortest_path(graph, directed=False, indices=early_idx)
            d_early = np.where(np.isfinite(d_early), d_early, np.nan)
            report["geodesic_early_mean_to_basins"] = {
                b: float(np.nanmean(d_early[:, labels == b], axis=1).mean())
                for b in BASINS if (labels == b).any()}
    else:
        geo_start = {b: float(np.linalg.norm(emb[start_idx] - c))
                     for b, c in centroids.items()}
        early = emb[df["early_middlegame"].to_numpy(dtype=bool)]
        if len(early):
            report["geodesic_early_mean_to_basins"] = {
                b: float(np.mean(np.linalg.norm(early - c, axis=1)))
                for b, c in centroids.items()}
    report["geodesic_start_to_basins"] = geo_start

    # --- Bootstrap-Test: d(start,draw) - d(start,white/black) ---
    boot = {"draw_minus_white": [], "draw_minus_black": []}
    basin_idx = {b: np.where(labels == b)[0] for b in BASINS}
    if "draw" in geo_start:
        for _ in range(cfg.n_bootstrap):
            means = {}
            for b in BASINS:
                idx = basin_idx[b]
                if len(idx) == 0:
                    means[b] = None
                    continue
                sample = idx[rng.integers(0, len(idx), size=len(idx))]
                if iso is not None:
                    means[b] = float(np.mean(d_start[sample]))
                else:
                    means[b] = float(np.mean(
                        np.linalg.norm(emb[sample] - emb[start_idx], axis=1)))
            if means.get("draw") is not None and means.get("white") is not None:
                boot["draw_minus_white"].append(means["draw"] - means["white"])
            if means.get("draw") is not None and means.get("black") is not None:
                boot["draw_minus_black"].append(means["draw"] - means["black"])
    stats = {}
    for key, vals in boot.items():
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        stats[key] = {"mean": float(vals.mean()),
                      "ci95": [float(lo), float(hi)],
                      "p_one_sided_draw_closer": float(np.mean(vals >= 0))}
    report["bootstrap_geodesic_test"] = stats

    # --- Vorhersagegenauigkeit + Gegenbeispiele ---
    classes, counts = np.unique(labels, return_counts=True)
    ok = classes[counts > cfg.knn_k]
    mask = np.isin(labels, ok)
    contradictions = []
    if len(ok) >= 2 and mask.sum() > cfg.knn_k * 2:
        cv = StratifiedKFold(n_splits=int(min(5, min(counts[counts > cfg.knn_k]))),
                             shuffle=True, random_state=cfg.seed)
        pred = cross_val_predict(KNeighborsClassifier(n_neighbors=cfg.knn_k),
                                 emb[mask], labels[mask], cv=cv)
        report["topo_prediction_accuracy"] = float((pred == labels[mask]).mean())
        wp_arr = df["wp_engine"].to_numpy()[mask]
        fen_arr = df["fen"].to_numpy()[mask]
        ply_arr = df["ply"].to_numpy()[mask]
        for p, wp, fen, ply in zip(pred, wp_arr, fen_arr, ply_arr):
            decisive = abs(wp - 0.5) > cfg.contra_margin
            if (decisive and p == "draw") or \
               (not decisive and p in ("white", "black") and abs(wp - 0.5) > 0.2):
                contradictions.append({"fen": fen, "ply": int(ply),
                                       "wp_engine": float(wp),
                                       "topo_prediction": p,
                                       "true_basin": basin_of(wp, cfg)})
    else:
        report["topo_prediction_accuracy"] = None
    report["n_contradictions"] = int(len(contradictions))
    report["contradiction_examples"] = contradictions[:25]

    # --- Baseline: depth-1-Eval ---
    if "wp_baseline" in df.columns:
        base_pred = np.array([basin_of(w, cfg) for w in df["wp_baseline"].fillna(0.5)])
        report["baseline_depth1_accuracy_all"] = float((base_pred == labels).mean())
        if mask.sum():
            report["baseline_depth1_accuracy_masked"] = float(
                (base_pred[mask] == labels[mask]).mean())
    if return_diagrams:
        report["ph_diagrams_raw"] = diagrams
    return report


def save_report(report: dict, path: str):
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, float) and np.isnan(o):
            return None
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(report), f, indent=2)
