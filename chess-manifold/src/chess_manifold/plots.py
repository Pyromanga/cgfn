"""Visualisierungen: Persistenzdiagramme je Becken, Einbettungs-PCA, Geodaeten-Balken."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def make_plots(cfg, df, emb, report, out_dir, diagrams=None):
    diagrams = diagrams or {}
    os.makedirs(out_dir, exist_ok=True)
    labels = df["basin"].to_numpy()

    # --- Embedding-PCA, gefaerbt nach Becken ---
    fig, ax = plt.subplots(figsize=(7, 6))
    p = PCA(n_components=2).fit_transform(emb)
    for b, c in zip(("white", "draw", "black"), ("tab:orange", "tab:gray", "tab:blue")):
        m = labels == b
        if m.any():
            ax.scatter(p[m, 0], p[m, 1], s=8, alpha=0.6, c=c, label=b)
    sm = df["is_start"].to_numpy(dtype=bool)
    if sm.any():
        ax.scatter(p[sm, 0], p[sm, 1], s=200, marker="*", c="red",
                   label="Startstellung", zorder=5)
    ax.legend()
    ax.set_title("Gelerntes Manifold (PCA-Projektion)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "embedding_pca.png"), dpi=150)
    plt.close(fig)

    # --- Persistenzdiagramme je Becken ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, b in zip(axes, ("white", "draw", "black")):
        dg = diagrams.get(b, {})
        if not dg:
            ax.set_title(f"{b}: keine PH")
            continue
        for dim in (0, 1):
            pts = np.asarray(dg.get(f"dim{dim}", []))
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=10, label=f"H{dim}")
        lim = ax.get_xlim()
        ax.plot([0, lim[1]], [0, lim[1]], "k--", lw=0.5)
        ax.set_title(f"Persistenzdiagramm: {b}-Becken")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "persistence_diagrams.png"), dpi=150)
    plt.close(fig)

    # --- Betti-Kurven ---
    ph = report.get("persistent_homology", {})
    fig, ax = plt.subplots(figsize=(7, 5))
    for b, c in zip(("white", "draw", "black"), ("tab:orange", "tab:gray", "tab:blue")):
        bc = ph.get(b, {}).get("betti_curve", [])
        if bc:
            r = [x["radius"] for x in bc]
            ax.plot(r, [x["betti1"] for x in bc], "-o", ms=3, c=c, label=f"beta1 {b}")
    ax.set_xlabel("Radius epsilon")
    ax.set_ylabel("Betti-1")
    ax.legend()
    ax.set_title("Betti-1-Kurven der Becken")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "betti_curves.png"), dpi=150)
    plt.close(fig)

    # --- Geodaeten Start -> Becken ---
    geo = report.get("geodesic_start_to_basins", {})
    if geo:
        fig, ax = plt.subplots(figsize=(6, 4))
        ks = list(geo)
        vs = [geo[k] for k in ks]
        ax.bar(ks, vs, color=["tab:orange", "tab:gray", "tab:blue"][:len(ks)])
        ax.set_title("Geodaetische Distanz Startstellung -> Becken")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "geodesics_start.png"), dpi=150)
        plt.close(fig)
