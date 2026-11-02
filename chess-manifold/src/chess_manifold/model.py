"""Encoder + Supervised Contrastive Learning (analog SOLIS).

MLP-Encoder (Feature-Vektor statt Token-Sequenz -- bei ~65 strukturierten
Merkmalen adequate und um Groessenordnungen billiger).
Ziel: L2-normalisierte Einbettung, in der Stellungen aehnlicher
Gewinnwahrscheinlichkeit nahe beieinander liegen (SupCon, Khosla et al.).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020, vereinfachte Form)."""
    z = F.normalize(embeddings, dim=-1)
    sim = z @ z.T / temperature
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(eye, -1e9)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


class SupConModule(pl.LightningModule):
    def __init__(self, in_dim, hidden, emb_dim, temperature, lr):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(in_dim, hidden, emb_dim)
        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = supcon_loss(self.encoder(x), y, self.hparams.temperature)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = supcon_loss(self.encoder(x), y, self.hparams.temperature)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
