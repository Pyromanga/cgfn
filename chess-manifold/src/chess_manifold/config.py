"""Zentrale Konfiguration. Alle Werte via Umgebungsvariablen (CM_*) ueberschreibbar."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- Daten ---
    n_games: int = int(os.environ.get("CM_N_GAMES", "25"))       # Random-Selfplay-Partien
    plies_per_game: int = int(os.environ.get("CM_PLIES", "12"))  # Stellungen pro Partie
    max_game_plies: int = 160
    seed: int = int(os.environ.get("CM_SEED", "42"))
    min_ply: int = 4          # Opening-Rauschen ausschliessen
    early_middlegame: tuple = (8, 20)

    # --- Engine ---
    engine_depth: int = int(os.environ.get("CM_ENGINE_DEPTH", "12"))  # Supervision
    baseline_depth: int = 1                                          # "ohne Suche"-Baseline

    # --- Becken-Definition (Engine-Gewinnwahrscheinlichkeit) ---
    wp_white: float = 0.55   # > 0.55 -> Weiss-Becken
    wp_black: float = 0.45   # < 0.45 -> Schwarz-Becken, sonst Remis-Becken

    # --- Modell ---
    emb_dim: int = int(os.environ.get("CM_EMB_DIM", "16"))
    hidden: int = 256
    epochs: int = int(os.environ.get("CM_EPOCHS", "30"))
    batch_size: int = 256
    temperature: float = 0.07
    supcon_bins: int = 6       # Fein-Supervision: Bins der Gewinnwahrscheinlichkeit
    lr: float = 1e-3

    # --- Topologie / Analyse ---
    maxdim: int = 1
    ph_subsample: int = int(os.environ.get("CM_PH_SUBSAMPLE", "1200"))  # PH-Input-Cap je Becken
    betti_radii: tuple = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    knn_k: int = 10
    isomap_neighbors: int = 10
    n_bootstrap: int = 2000
    contra_margin: float = 0.35  # |wp - 0.5| > margin UND anderes Becken vorhergesagt

    # --- Pfade ---
    data_dir: str = os.environ.get("CM_DATA", "data")
    out_dir: str = os.environ.get("CM_OUT", "results")
