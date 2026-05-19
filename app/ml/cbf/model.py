"""
Content-Based Filtering Model Wrapper

Provides similarity search and recommendations using PCA-reduced
SBERT item embeddings.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CBFModel:
    """Content-based filtering model using SBERT + PCA item embeddings."""

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.item_embeddings: Optional[np.ndarray] = None  # PCA-reduced (n_items, 128)
        self.item_embeddings_raw: Optional[np.ndarray] = None  # Full SBERT (n_items, 384)
        self._normed_embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.load(self.model_dir)

    def load(self, model_dir: Optional[Path] = None):
        """Load PCA item embeddings from disk."""
        model_dir = Path(model_dir) if model_dir else self.model_dir

        logger.info(f"Loading CBF model from {model_dir}...")

        self.item_embeddings = np.load(model_dir / "item_pca_embeddings.npy")

        # Pre-compute normalized embeddings for cosine similarity
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._normed_embeddings = self.item_embeddings / norms

        self._loaded = True
        logger.info(f"Loaded CBF: {self.item_embeddings.shape[0]} items, "
                     f"{self.item_embeddings.shape[1]} dims")
        return self

    @classmethod
    def from_pretrained(cls, model_dir: str = "data/models") -> "CBFModel":
        """Load a pre-trained CBF model."""
        instance = cls(model_dir)
        instance.load()
        return instance

    def save(self, model_dir: Optional[str] = None):
        """Save embeddings to disk."""
        model_dir = Path(model_dir) if model_dir else self.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "item_pca_embeddings.npy", self.item_embeddings)
        logger.info(f"Saved CBF embeddings to {model_dir}")

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get the PCA embedding for an item. Shape: (128,)"""
        self._ensure_loaded()
        return self.item_embeddings[item_idx]

    def get_similar_items(self, item_idx: int, n: int = 10) -> list[int]:
        """Find n most similar items using cosine similarity over PCA embeddings."""
        self._ensure_loaded()
        query = self._normed_embeddings[item_idx]

        # Cosine similarity via dot product of normalized vectors
        scores = self._normed_embeddings @ query

        # Exclude the query item
        scores[item_idx] = -np.inf
        top_indices = np.argsort(scores)[::-1][:n]

        return top_indices.tolist()

    def recommend_for_user(
        self,
        user_idx: int,
        interaction_history: list[int],
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Recommend items by averaging the embeddings of items the user
        has interacted with, then finding nearest items.

        Args:
            user_idx: User index (used for consistency, not directly needed)
            interaction_history: List of item indices the user interacted with
            n: Number of recommendations

        Returns:
            List of (item_idx, score) tuples sorted by similarity descending
        """
        self._ensure_loaded()

        if not interaction_history:
            return []

        # Average interacted item embeddings
        history_embeds = self.item_embeddings[interaction_history]
        user_profile = np.mean(history_embeds, axis=0)

        # Normalize
        user_norm = np.linalg.norm(user_profile)
        if user_norm > 1e-10:
            user_profile = user_profile / user_norm

        # Cosine similarity against all items
        scores = self._normed_embeddings @ user_profile

        # Exclude already interacted items
        for idx in interaction_history:
            scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
