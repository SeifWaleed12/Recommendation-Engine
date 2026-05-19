"""
Ranker Wrapper

Wraps the trained LightGBM LambdaRank model for inference.
Ranks a list of candidate items for a given user.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import pandas as pd
from app.ml.ranking.features import HONEST_FEATURE_NAMES

logger = logging.getLogger(__name__)


class Ranker:
    """Wrapper around a trained LightGBM ranker for inference."""

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.ranker = None
        self._feature_assembler = None
        self._user_idx_map: Optional[dict] = None
        self._item_idx_map: Optional[dict] = None
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.load(self.model_dir)

    def load(self, model_dir: Optional[Path] = None):
        """Load ranker model and dependencies."""
        model_dir = Path(model_dir) if model_dir else self.model_dir

        logger.info(f"Loading ranker from {model_dir}...")

        with open(model_dir / "lgbm_ranker.pkl", "rb") as f:
            self.ranker = pickle.load(f)

        # Load index maps
        with open(model_dir / "user_idx_map.json", "r") as f:
            self._user_idx_map = json.load(f)
        with open(model_dir / "item_idx_map.json", "r") as f:
            self._item_idx_map = json.load(f)

        # Initialize feature assembler
        from app.ml.ranking.features import FeatureAssembler
        self._feature_assembler = FeatureAssembler(str(model_dir))

        self._loaded = True
        logger.info("Ranker loaded successfully")
        return self

    @classmethod
    def from_pretrained(cls, model_dir: str = "data/models") -> "Ranker":
        """Load a pre-trained ranker."""
        instance = cls(model_dir)
        instance.load()
        return instance

    def save(self, model_dir: Optional[str] = None):
        """Save ranker to disk."""
        model_dir = Path(model_dir) if model_dir else self.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "lgbm_ranker.pkl", "wb") as f:
            pickle.dump(self.ranker, f)
        logger.info(f"Saved ranker to {model_dir}")

    def rank_candidates(
        self,
        user_id: str,
        candidate_item_ids: list[str],
        context: dict = None,
    ) -> list[tuple[str, float]]:
        """
        Rank candidate items for a user.

        Args:
            user_id: User external_id
            candidate_item_ids: List of item external_ids to rank
            context: Optional context dictionary (unused for now)

        Returns:
            List of (item_id, score) tuples sorted by predicted
            relevance score descending.
        """
        self._ensure_loaded()

        if not candidate_item_ids:
            return []

        # Build feature vectors for all pairs
        pairs = [(user_id, item_id) for item_id in candidate_item_ids]
        features = self._feature_assembler.assemble_ranking_features_batch(
            pairs,
            user_idx_map=self._user_idx_map,
            item_idx_map=self._item_idx_map,
            context=context,
        )

        # Predict scores
        X = pd.DataFrame(features, columns=HONEST_FEATURE_NAMES)
        scores = self.ranker.predict(X)

        # Sort by score descending
        ranked = sorted(
            zip(candidate_item_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(item_id, float(score)) for item_id, score in ranked]


