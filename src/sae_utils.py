"""sae_utils.py
-----------------------------------------------------------------------
Thin, opinionated helper layer around *SAE-Lens* checkpoints
so you can:
• load any pre-trained SAE by (release, sae_id)
• project residual-stream activations → sparse feature space → back
• run quick-n-dirty linear probes in latent or residual space
• compute simple Δ-activation stats (mean shift, cosine, etc.)
• do a one-liner 2-D visualisation (UMAP / t-SNE) for sanity-checks

Dependencies
------------
pip install torch sae-lens scikit-learn umap-learn matplotlib
"""

from __future__ import annotations

import torch as t
from functools import lru_cache
from typing import Literal, Union

from sae_lens import SAE  # https://github.com/sae-lens/sae_lens

# =====================================================================
#                               Loader
# =====================================================================

@lru_cache(maxsize=None)
def load_sae(*, release: str, sae_id: str, device: str = "cpu") -> SAE:
    """Load pre-trained SAE from SAE-Lens by model and SAE ID."""
    return SAE.from_pretrained(release, sae_id)[0].to(device)

# =====================================================================
#                           Thin  OO Wrapper
# =====================================================================

class SAEWrapper:
    def __init__(
        self,
        *,
        release: str,
        sae_id: str,
        device: str = "cpu",
    ) -> None:
        self.sae = load_sae(release=release, sae_id=sae_id, device=device)
        self.device = self.sae.W_dec.device

    def encode(self, x: t.Tensor) -> t.Tensor:
        return self.sae.encode(x)

    def decode(self, z: t.Tensor) -> t.Tensor:
        return self.sae.decode(z)

    def reconstruct(self, x: t.Tensor) -> t.Tensor:
        return self.sae(x)

    @t.no_grad()
    def feature_shift(self, x1: t.Tensor, x2: t.Tensor) -> t.Tensor:
        return (self.encode(x2) - self.encode(x1)).mean(dim=0)

    @t.no_grad()
    def latent_stats(self, x: t.Tensor) -> dict[str, float]:
        z = self.encode(x)
        return {
            "mean_active": (z != 0).float().mean().item(),
            "mean_abs": z.abs().mean().item(),
        }

# =====================================================================
#                     Simple *linear-probe* utilities
# =====================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_linear_probe(z: t.Tensor, y: t.Tensor, *, test_size: float = 0.2, seed: int = 0) -> float:
    X = z.cpu().numpy()
    y_np = y.cpu().numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_np, test_size=test_size, random_state=seed)
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    return accuracy_score(y_te, preds)

# =====================================================================
#                     Quick visualisation helper
# =====================================================================
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def quick_scatter(z: t.Tensor, labels: t.Tensor | None = None, *, method: str = "umap", title: str = ""):
    reducer = TSNE(n_components=2, perplexity=30, random_state=42) if method.lower() == "tsne" \
              else umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    z_np = z.cpu().numpy()
    z_2d = reducer.fit_transform(z_np)

    plt.figure(figsize=(7, 5))
    if labels is None:
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=3, alpha=0.6)
    else:
        plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels.cpu().numpy(), cmap="viridis", s=4, alpha=0.7)
        plt.colorbar(label="label")
    plt.title(title or f"{method.upper()} projection of SAE latents")
    plt.xlabel("comp-1")
    plt.ylabel("comp-2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    release = "llama_scope_lxr_32x"
    sae_id = "l0r_32x"
    sae = load_sae(release=release, sae_id=sae_id, device="cuda:0")
    print("Loaded SAE:", sae)
