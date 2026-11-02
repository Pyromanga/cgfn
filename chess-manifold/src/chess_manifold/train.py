"""Training des SupCon-Encoders und Berechnung aller Einbettungen."""
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .model import SupConModule


def wp_to_bin(wp: float, bins: int) -> int:
    return min(bins - 1, int(wp * bins))


def standardize(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd, mu, sd


def train_encoder(cfg: Config, df: pd.DataFrame, feature_cols):
    X = df[feature_cols].to_numpy(dtype=np.float64)
    Xz, mu, sd = standardize(X)
    labels = np.array([wp_to_bin(w, cfg.supcon_bins) for w in df["wp_engine"]])

    n = len(df)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_t = torch.tensor(Xz, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)
    train_dl = DataLoader(TensorDataset(X_t[tr_idx], y_t[tr_idx]),
                          batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_t[val_idx], y_t[val_idx]),
                       batch_size=cfg.batch_size)

    model = SupConModule(in_dim=Xz.shape[1], hidden=cfg.hidden,
                         emb_dim=cfg.emb_dim, temperature=cfg.temperature, lr=cfg.lr)
    trainer = Trainer(max_epochs=cfg.epochs, accelerator="cpu",
                     logger=False, enable_checkpointing=False,
                     enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_dl, val_dl)

    model.eval()
    with torch.no_grad():
        emb = model(X_t).numpy()
    val_loss = trainer.validate(model, val_dl, verbose=False)[0]["val_loss"]
    return model, emb, (mu, sd), {"train_size": int(len(tr_idx)),
                                   "val_size": int(len(val_idx)),
                                   "val_supcon_loss": float(val_loss)}


def embed_single(model, mu, sd, x_row: np.ndarray) -> np.ndarray:
    xz = (x_row - mu) / sd
    with torch.no_grad():
        return model(torch.tensor(xz, dtype=torch.float32).unsqueeze(0)).numpy()[0]
