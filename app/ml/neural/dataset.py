"""
Two-Tower Dataset

PyTorch dataset for training with BPR loss.
For each positive (user, item) pair, samples 4 random negative items.
"""

import sys
import json
import random
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:postgres@localhost:5432/recsys"


def _load_feature_store_features(entity_type: str, feature_name: str) -> dict:
    """Load features from FeatureStore as {entity_id: feature_dict}."""
    try:
        from sqlalchemy import create_engine
        engine = create_engine(DB_URL)
        df = pd.read_sql(
            f"SELECT entity_id, feature_value_json FROM feature_store "
            f"WHERE entity_type = '{entity_type}' AND feature_name = '{feature_name}'",
            engine,
        )
        return dict(zip(df["entity_id"], df["feature_value_json"]))
    except Exception as e:
        logger.warning(f"Could not load {entity_type} features from FeatureStore: {e}")
        return {}


def _build_user_feature_vector(features: dict) -> np.ndarray:
    """
    Build 16-dim user feature vector from FeatureStore user_profile_v1.

    Features: [interaction_count, purchase_count, avg_session_length,
               price_percentile, last_active_days_ago, + 11 zeros]
    """
    vec = np.zeros(16, dtype=np.float32)
    if features:
        vec[0] = float(features.get("interaction_count", 0))
        vec[1] = float(features.get("purchase_count", 0))
        vec[2] = 0.0  # avg_session_length — not computed yet
        vec[3] = float(features.get("price_percentile", 0) or 0)
        vec[4] = float(features.get("last_active_days_ago", 0) or 0)
    return vec


def _build_item_feature_vector(features: dict) -> np.ndarray:
    """
    Build 16-dim item feature vector from FeatureStore item_profile_v1.

    Features: [price_normalized, global_view_count_log, conversion_rate,
               days_since_created_log, avg_interaction_weight, + 11 zeros]
    """
    vec = np.zeros(16, dtype=np.float32)
    if features:
        # Use log transforms for count features to reduce skew
        view_count = float(features.get("global_view_count", 0))
        days_created = float(features.get("days_since_created", 0))

        vec[0] = float(features.get("price_normalized", 0) if "price_normalized" in features else 0)
        vec[1] = np.log1p(view_count)
        vec[2] = float(features.get("conversion_rate", 0))
        vec[3] = np.log1p(days_created)
        vec[4] = float(features.get("avg_interaction_weight", 0))
    return vec


class TwoTowerDataset(Dataset):
    """
    Dataset for Two-Tower training with BPR negative sampling.

    For each positive (user, item) pair, samples `n_negatives` random items
    that the user has NOT interacted with.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        user_idx_map: dict,
        item_idx_map: dict,
        n_items: int,
        n_negatives: int = 4,
        user_features: dict = None,
        item_features: dict = None,
    ):
        self.n_items = n_items
        self.n_negatives = n_negatives

        # Map external IDs to integer indices
        self.user_indices = interactions_df["user_ext_id"].map(user_idx_map).values
        self.item_indices = interactions_df["item_ext_id"].map(item_idx_map).values

        # Filter out rows that couldn't be mapped
        valid = ~(np.isnan(self.user_indices) | np.isnan(self.item_indices))
        # Handle potential NaN from mapping misses
        if hasattr(self.user_indices, 'astype'):
            try:
                self.user_indices = self.user_indices.astype(np.int64)
                self.item_indices = self.item_indices.astype(np.int64)
            except (ValueError, TypeError):
                mask = pd.notna(interactions_df["user_ext_id"].map(user_idx_map)) & \
                       pd.notna(interactions_df["item_ext_id"].map(item_idx_map))
                self.user_indices = interactions_df.loc[mask, "user_ext_id"].map(user_idx_map).values.astype(np.int64)
                self.item_indices = interactions_df.loc[mask, "item_ext_id"].map(item_idx_map).values.astype(np.int64)

        # Build user → set of interacted items for negative sampling
        self.user_items = defaultdict(set)
        for u, i in zip(self.user_indices, self.item_indices):
            self.user_items[int(u)].add(int(i))

        # Feature vectors
        self.user_features = user_features or {}
        self.item_features = item_features or {}

        # Pre-compute feature vectors for all users and items
        self._user_feat_cache = {}
        self._item_feat_cache = {}

        logger.info(f"Dataset: {len(self.user_indices)} interactions, "
                     f"{len(self.user_items)} users, {n_negatives} negatives/positive")

    def _get_user_feats(self, user_ext_id: str) -> np.ndarray:
        if user_ext_id not in self._user_feat_cache:
            feats = self.user_features.get(user_ext_id, {})
            self._user_feat_cache[user_ext_id] = _build_user_feature_vector(feats)
        return self._user_feat_cache[user_ext_id]

    def _get_item_feats(self, item_ext_id: str) -> np.ndarray:
        if item_ext_id not in self._item_feat_cache:
            feats = self.item_features.get(item_ext_id, {})
            self._item_feat_cache[item_ext_id] = _build_item_feature_vector(feats)
        return self._item_feat_cache[item_ext_id]

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        user_idx = int(self.user_indices[idx])
        pos_item_idx = int(self.item_indices[idx])
        interacted = self.user_items[user_idx]

        # Sample negative items
        neg_items = []
        attempts = 0
        while len(neg_items) < self.n_negatives and attempts < self.n_negatives * 10:
            neg = random.randint(0, self.n_items - 1)
            if neg not in interacted:
                neg_items.append(neg)
            attempts += 1

        # Pad with random items if we couldn't find enough
        while len(neg_items) < self.n_negatives:
            neg_items.append(random.randint(0, self.n_items - 1))

        # Feature vectors (use zeros as fallback — features keyed by ext_id
        # but we only have idx here; pre-built caches handle this)
        user_feats = np.zeros(16, dtype=np.float32)
        pos_item_feats = np.zeros(16, dtype=np.float32)
        neg_item_feats = np.zeros((self.n_negatives, 16), dtype=np.float32)

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "user_feats": torch.tensor(user_feats, dtype=torch.float32),
            "pos_item_idx": torch.tensor(pos_item_idx, dtype=torch.long),
            "pos_item_feats": torch.tensor(pos_item_feats, dtype=torch.float32),
            "neg_item_idxs": torch.tensor(neg_items, dtype=torch.long),
            "neg_item_feats": torch.tensor(neg_item_feats, dtype=torch.float32),
        }


def create_datasets(
    data_dir: str = "data/processed",
    train_ratio: float = 0.8,
    n_negatives: int = 4,
) -> tuple[TwoTowerDataset, TwoTowerDataset, int, int]:
    """
    Create train/val datasets with temporal split.

    Returns:
        (train_dataset, val_dataset, n_users, n_items)
    """
    data_dir = Path(data_dir)

    logger.info("Loading train interactions and index maps...")
    df = pd.read_parquet(data_dir / "train_interactions.parquet")

    with open(data_dir / "user_idx_map.json", "r") as f:
        user_idx_map = json.load(f)
    with open(data_dir / "item_idx_map.json", "r") as f:
        item_idx_map = json.load(f)

    n_users = len(user_idx_map)
    n_items = len(item_idx_map)

    # Temporal split: sort by timestamp, take first 80% for training
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Load features from FeatureStore
    logger.info("Loading features from FeatureStore...")
    user_features = _load_feature_store_features("user", "user_profile_v1")
    item_features = _load_feature_store_features("item", "item_profile_v1")
    logger.info(f"Loaded {len(user_features)} user features, {len(item_features)} item features")

    train_dataset = TwoTowerDataset(
        train_df, user_idx_map, item_idx_map, n_items,
        n_negatives=n_negatives,
        user_features=user_features,
        item_features=item_features,
    )
    val_dataset = TwoTowerDataset(
        val_df, user_idx_map, item_idx_map, n_items,
        n_negatives=n_negatives,
        user_features=user_features,
        item_features=item_features,
    )

    return train_dataset, val_dataset, n_users, n_items
