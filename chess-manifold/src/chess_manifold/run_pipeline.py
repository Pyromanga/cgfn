"""Pipeline-Orchestrierung: data -> engine eval -> features -> training -> analyse.

Aufruf: python -m chess_manifold.run_pipeline (Konfiguration via CM_*-Env-Variablen)
"""
import json
import os
import pickle

import chess
import numpy as np
import pandas as pd

from .analysis import basin_of, run_analysis, save_report
from .analysis_dynamics import run_dynamics
from .analysis_openings import opening_stats
from .config import Config
from .data import generate_positions
from .engine import Engine, cp_to_winprob
from .features import extract_features
from .lichess import generate_positions_lichess
from .plots import make_plots
from .train import train_encoder


def stage_data(cfg: Config):
    if cfg.source == "lichess":
        rows, n_games = generate_positions_lichess(cfg)
        print(f"    Quelle: Lichess-API ({n_games} Partien)")
    else:
        rows = generate_positions(cfg)
    boards = [chess.Board(r["fen"]) for r in rows]
    keys = sorted(extract_features(boards[0]).keys())
    feats = [extract_features(b) for b in boards]
    df = pd.DataFrame([{**r, **f} for r, f in zip(rows, feats)])
    return df, keys


def main():
    cfg = Config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("[1/5] Datengenerierung ...")
    df, feature_cols = stage_data(cfg)

    print("[2/5] Engine-Bewertung ...")
    eng = Engine(depth=cfg.engine_depth)
    engine_available = eng.available
    wp_engine, wp_baseline, cps = [], [], []
    for fen in df["fen"]:
        board = chess.Board(fen)
        cp = eng.eval_cp(board, depth=cfg.engine_depth)
        cps.append(cp)
        wp_engine.append(cp_to_winprob(cp))
        wp_baseline.append(cp_to_winprob(eng.eval_cp(board, depth=cfg.baseline_depth)))
    eng.close()
    df["cp_engine"] = cps
    df["wp_engine"] = wp_engine
    df["wp_baseline"] = wp_baseline
    df["basin"] = [basin_of(w, cfg) for w in wp_engine]
    df.to_csv(os.path.join(cfg.data_dir, "positions.csv"), index=False)
    print(f"    Engine verfuegbar: {engine_available}; Positionen: {len(df)}")

    print("[3/5] Contrastive Training ...")
    model, emb, scaler, train_info = train_encoder(cfg, df, feature_cols)
    df_emb = pd.DataFrame(emb, columns=[f"z{i}" for i in range(emb.shape[1])])
    pd.concat([df[["fen", "ply", "wp_engine", "basin", "is_start",
                   "early_middlegame"]], df_emb], axis=1).to_csv(
        os.path.join(cfg.data_dir, "embeddings.csv"), index=False)

    print("[4/5] Topologische Analyse + Hypothesentest ...")
    start_mask = df["is_start"].to_numpy(dtype=bool)
    start_idx = int(np.where(start_mask)[0][0])
    report = run_analysis(cfg, df, emb, start_idx, return_diagrams=True)
    diagrams = report.pop("ph_diagrams_raw", {})
    np.savez_compressed(
        os.path.join(cfg.out_dir, "ph_diagrams.npz"),
        **{f"{b}_{d}": np.asarray(v, dtype=float)
           for b, dg in diagrams.items() for d, v in dg.items()})
    report["engine_available"] = bool(engine_available)
    report["train_info"] = train_info
    report["config"] = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in vars(cfg).items()}

    print("[4b/5] Dynamikanalyse (Trajektorien, Remis-Ursachen-Test) ...")
    try:
        report["dynamics"] = run_dynamics(cfg, df, emb)
    except Exception as e:
        report["dynamics"] = {"error": repr(e)}

    print("[4c/5] Opening-Analyse (quiet draws sichtbar machen) ...")
    try:
        report["openings"] = opening_stats(df, cfg)
    except Exception as e:
        report["openings"] = {"error": repr(e)}
    save_report(report, os.path.join(cfg.out_dir, "report.json"))

    print("[5/5] Plots ...")
    make_plots(cfg, df, emb, report, os.path.join(cfg.out_dir, "plots"), diagrams)

    with open(os.path.join(cfg.out_dir, "model.pkl"), "wb") as f:
        pickle.dump({"model_state": model.state_dict(),
                     "scaler": scaler, "feature_cols": feature_cols}, f)

    summary = {
        "n_positions": report["n_positions"],
        "basin_sizes": report["basin_sizes"],
        "coherence_knn_accuracy": report.get("coherence_knn_accuracy"),
        "topo_prediction_accuracy": report.get("topo_prediction_accuracy"),
        "baseline_depth1_accuracy": report.get("baseline_depth1_accuracy_all"),
        "geodesic_start_to_basins": report.get("geodesic_start_to_basins"),
        "bootstrap_geodesic_test": report.get("bootstrap_geodesic_test"),
        "n_contradictions": report.get("n_contradictions"),
        "dynamics": report.get("dynamics"),
        "openings": report.get("openings"),
        "engine_available": report["engine_available"],
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print("Fertig. Ergebnisse in", cfg.out_dir)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
