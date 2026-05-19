"""
iALS Model Wrapper

Provides a clean interface for ALS model inference: get embeddings,
find similar items, and generate user recommendations.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import load_npz

logger = logging.getLogger(__name__)


class ALSModel:
    """Wrapper around a trained implicit ALS model for inference."""

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.load(self.model_dir)

    def load(self, model_dir: Optional[Path] = None):
        """Load model and embeddings from disk."""
        model_dir = Path(model_dir) if model_dir else self.model_dir

        logger.info(f"Loading ALS model from {model_dir}...")

        # We DO NOT load als_model.pkl during inference because it consumes ~1GB of RAM.
        # We only need the embeddings for similarity and recommendation math.
        
        # mmap_mode='r': only reads the rows needed per-request, not all ~986MB combined
        self.user_embeddings = np.load(model_dir / "user_embeddings_als.npy", mmap_mode='r')
        self.item_embeddings = np.load(model_dir / "item_embeddings_als.npy", mmap_mode='r')

        self._loaded = True
        logger.info(
            f"Loaded ALS: {self.user_embeddings.shape[0]} users, "
            f"{self.item_embeddings.shape[0]} items, "
            f"{self.user_embeddings.shape[1]} factors"
        )
        return self

    @classmethod
    def from_pretrained(cls, model_dir: str = "data/models") -> "ALSModel":
        """Load a pre-trained ALS model."""
        instance = cls(model_dir)
        instance.load()
        return instance

    def save(self, model_dir: Optional[str] = None):
        """Save model and embeddings to disk."""
        model_dir = Path(model_dir) if model_dir else self.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "als_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        np.save(model_dir / "user_embeddings_als.npy", self.user_embeddings)
        np.save(model_dir / "item_embeddings_als.npy", self.item_embeddings)

        logger.info(f"Saved ALS model to {model_dir}")

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get the embedding vector for a user. Shape: (128,)"""
        self._ensure_loaded()
        return self.user_embeddings[user_idx]

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get the embedding vector for an item. Shape: (128,)"""
        self._ensure_loaded()
        return self.item_embeddings[item_idx]

    def get_similar_items(self, item_idx: int, n: int = 10) -> list[int]:
        """Find the n most similar items by cosine similarity in ALS space."""
        self._ensure_loaded()
        item_vec = self.item_embeddings[item_idx]

        # Cosine similarity: normalize then dot product
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        normed = self.item_embeddings / norms

        item_vec_normed = item_vec / max(np.linalg.norm(item_vec), 1e-10)
        scores = normed @ item_vec_normed

        # Exclude the item itself
        scores[item_idx] = -np.inf
        top_indices = np.argsort(scores)[::-1][:n]

        return top_indices.tolist()

    def recommend_for_user(
        self,
        user_idx: int,
        n: int = 10,
        exclude_interacted: bool = True,
        interaction_matrix=None,
    ) -> list[tuple[int, float]]:
        """
        Recommend top-n items for a user.

        Returns list of (item_idx, score) tuples sorted by score descending.
        """
        self._ensure_loaded()
        user_vec = self.user_embeddings[user_idx]

        # Score all items via dot product
        scores = self.item_embeddings @ user_vec

        if exclude_interacted and interaction_matrix is not None:
            # Zero out items the user already interacted with
            interacted = interaction_matrix[user_idx].nonzero()[1]
            scores[interacted] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
