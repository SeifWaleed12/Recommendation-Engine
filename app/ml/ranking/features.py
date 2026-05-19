"""
Ranking Feature Assembly

Builds 25-dimensional feature vectors for (user, item) pairs.
Optimized for Batch JIT loading to balance speed and RAM.
"""

import sys
import logging
import traceback
from pathlib import Path
from typing import Optional
from functools import lru_cache

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:postgres@localhost:5432/recsys"

FEATURE_NAMES = [
    "user_interaction_count",       # 0
    "user_purchase_count",          # 1
    "user_avg_session_length",      # 2
    "user_price_percentile",        # 3
    "user_last_active_days_ago",    # 4
    "item_price",                   # 5
    "item_global_view_count",       # 6
    "item_global_purchase_count",   # 7
    "item_conversion_rate",         # 8
    "item_unique_user_count",       # 9
    "item_days_since_created",      # 10
    "cross_category_match",         # 11
    "cross_price_in_range",         # 12
    "cross_als_score",              # 13
    "cross_cbf_score",              # 14
    "item_trending_score",          # 15
    "user_recency_decay",           # 16
    "cross_neural_two_tower_score", # 17
    "user_item_category_rank",      # 18
    "price_elasticity_score",       # 19
    "session_item_view_count",      # 20
    "item_ctr_normalized",          # 21
    "user_history_similarity",      # 22
    "item_diversity_score",         # 23
    "user_category_affinity_score", # 24
    "retrieval_sasrec_hit",         # 25
    "retrieval_als_hit",            # 26
    "retrieval_cbf_hit",            # 27
    "retrieval_neural_hit",         # 28
    "retrieval_pop_hit"             # 29
]

# Final Feature List (26 items) after dropping [2, 4, 12, 14]
HONEST_FEATURE_NAMES = [f for i, f in enumerate(FEATURE_NAMES) if i not in [2, 4, 12, 14]]

class FeatureAssembler:
    """
    Assembles ranking features for (user, item) pairs.
    Uses Batch JIT fetching for high-performance data retrieval.
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self._engine = None
        self._item_features: dict = {}
        self._item_info: dict = {}
        self._user_cache: dict = {} # Local short-term cache
        
        self._als_user_embeddings: Optional[np.ndarray] = None
        self._als_item_embeddings: Optional[np.ndarray] = None
        self._neural_user_embeddings: Optional[np.ndarray] = None
        self._neural_item_embeddings: Optional[np.ndarray] = None
        self._cbf_item_embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        logger.info("Initializing Batch-JIT Feature Assembler...")
        self._engine = create_engine(DB_URL)

        try:
            # Pre-load item metadata (small enough for RAM)
            item_df = pd.read_sql(
                "SELECT entity_id, feature_value_json FROM feature_store "
                "WHERE entity_type = 'item' AND feature_name = 'item_profile_v1'",
                self._engine,
            )
            self._item_features = dict(zip(item_df["entity_id"], item_df["feature_value_json"]))

            items_info_df = pd.read_sql(
                "SELECT external_id, category, price FROM items",
                self._engine,
            )
            self._item_info = items_info_df.set_index("external_id").to_dict("index")
        except Exception as e:
            logger.error(f"Error loading item metadata: {e}")

        def load_npy(name):
            path = self.model_dir / name
            return np.load(path, mmap_mode="r") if path.exists() else None

        self._als_user_embeddings = load_npy("user_embeddings_als.npy")
        self._als_item_embeddings = load_npy("item_embeddings_als.npy")
        self._neural_user_embeddings = load_npy("user_embeddings_neural.npy")
        self._neural_item_embeddings = load_npy("item_embeddings_neural.npy")
        self._cbf_item_embeddings = load_npy("item_pca_embeddings.npy")

        self._loaded = True

    def _fetch_user_batch(self, user_ids: list[str]) -> dict:
        """Fetch multiple user profiles in ONE database request."""
        # Check cache first
        missing = [uid for uid in user_ids if uid not in self._user_cache]
        if not missing:
            return {uid: self._user_cache[uid] for uid in user_ids}

        # Batch fetch from DB
        try:
            # SQL IN clause for efficiency
            ids_str = "', '".join(missing)
            query = f"SELECT entity_id, feature_value_json FROM feature_store WHERE entity_type = 'user' AND entity_id IN ('{ids_str}')"
            df = pd.read_sql(query, self._engine)
            
            # Update cache
            batch_data = dict(zip(df["entity_id"], df["feature_value_json"]))
            self._user_cache.update(batch_data)
            
            # Fill remaining missing with empty dicts to avoid re-querying
            for uid in missing:
                if uid not in self._user_cache:
                    self._user_cache[uid] = {}
        except Exception as e:
            logger.error(f"Batch DB Fetch failed: {e}")

        return {uid: self._user_cache.get(uid, {}) for uid in user_ids}

    def assemble_ranking_features_batch(
        self,
        pairs: list[tuple[str, str]],
        user_idx_map: dict,
        item_idx_map: dict,
        context: dict = None,
        chunk_size: int = 10000, 
    ) -> np.ndarray:
        self._ensure_loaded()
        n = len(pairs)
        if n == 0: return np.zeros((0, len(HONEST_FEATURE_NAMES)), dtype=np.float32)

        all_feats = np.zeros((n, len(HONEST_FEATURE_NAMES)), dtype=np.float32)
        
        # Parse retrieval metadata if present (context should map item_id to hit dict)
        if context is None:
            context = {}

        for start_idx in range(0, n, chunk_size):
            # Clear cache between massive chunks to save RAM
            if start_idx % 100000 == 0:
                self._user_cache = {}

            end_idx = min(start_idx + chunk_size, n)
            chunk_pairs = pairs[start_idx:end_idx]
            
            user_ids = [str(p[0]) for p in chunk_pairs]
            item_ids = [str(p[1]) for p in chunk_pairs]
            
            # ── BATCH FETCH USERS (The 100x speedup) ───────────────────
            user_data_map = self._fetch_user_batch(list(set(user_ids)))
            
            u_idxs = np.array([user_idx_map.get(uid, -1) for uid in user_ids], dtype=np.int32)
            i_idxs = np.array([item_idx_map.get(iid, -1) for iid in item_ids], dtype=np.int32)
            
            feats = np.zeros((len(chunk_pairs), len(FEATURE_NAMES)), dtype=np.float32)

            user_data = [user_data_map.get(uid, {}) for uid in user_ids]
            item_data = [self._item_features.get(iid, {}) for iid in item_ids]
            item_info = [self._item_info.get(iid, {}) for iid in item_ids]

            # Vectorized assignments
            feats[:, 0] = [float(u.get("interaction_count", 0)) for u in user_data]
            feats[:, 1] = [float(u.get("purchase_count", 0)) for u in user_data]
            feats[:, 3] = [float(u.get("price_percentile", 0) or 0) for u in user_data]
            feats[:, 4] = [float(u.get("last_active_days_ago", 0) or 0) for u in user_data]

            feats[:, 5] = [float(i.get("price", 0) or 0) for i in item_info]
            feats[:, 6] = [float(i.get("global_view_count", 0)) for i in item_data]
            feats[:, 7] = [float(i.get("global_purchase_count", 0)) for i in item_data]
            feats[:, 8] = [float(i.get("conversion_rate", 0)) for i in item_data]
            feats[:, 9] = [float(i.get("unique_user_count", 0)) for i in item_data]
            feats[:, 10] = [float(i.get("days_since_created", 0)) for i in item_data]

            valid_idxs = (u_idxs >= 0) & (i_idxs >= 0)
            if self._als_user_embeddings is not None and self._als_item_embeddings is not None:
                u_emb = self._als_user_embeddings[u_idxs[valid_idxs]]
                i_emb = self._als_item_embeddings[i_idxs[valid_idxs]]
                feats[valid_idxs, 13] = (u_emb * i_emb).sum(axis=1)

            if self._cbf_item_embeddings is not None:
                i_emb_c = self._cbf_item_embeddings[i_idxs[valid_idxs]]
                feats[valid_idxs, 14] = (i_emb_c * i_emb_c).sum(axis=1) * 0.5

            if self._neural_user_embeddings is not None and self._neural_item_embeddings is not None:
                u_emb_n = self._neural_user_embeddings[u_idxs[valid_idxs]]
                i_emb_n = self._neural_item_embeddings[i_idxs[valid_idxs]]
                feats[valid_idxs, 17] = (u_emb_n * i_emb_n).sum(axis=1)

            user_cats = [u.get("preferred_category") for u in user_data]
            item_cats = [i.get("category") for i in item_info]
            feats[:, 11] = [1.0 if (uc and uc == ic) else 0.0 for uc, ic in zip(user_cats, item_cats)]
            feats[:, 15] = feats[:, 6] / (feats[:, 10] + 1) 
            feats[:, 19] = np.abs(feats[:, 5] - feats[:, 3])
            feats[:, 21] = (feats[:, 7] + 1.0) / (feats[:, 6] + 10.0)
            
            # Retrieval hits from context
            for j, (uid, iid) in enumerate(chunk_pairs):
                # context can be mapped by uid -> iid -> features, or just iid if training
                hit_data = {}
                if uid in context and iid in context[uid]:
                    hit_data = context[uid][iid]
                elif iid in context and isinstance(context[iid], dict) and "sasrec" in context[iid]:
                    hit_data = context[iid]
                
                feats[j, 25] = float(hit_data.get("sasrec", 0))
                feats[j, 26] = float(hit_data.get("als", 0))
                feats[j, 27] = float(hit_data.get("cbf", 0))
                feats[j, 28] = float(hit_data.get("neural", 0))
                feats[j, 29] = float(hit_data.get("pop", 0))

            # ── DROP LEAKY/BROKEN FEATURES (2, 4, 12, 14) ────────────────
            drop_indices = [2, 4, 12, 14]
            feats_honest = np.delete(feats, drop_indices, axis=1)

            all_feats[start_idx:end_idx] = np.nan_to_num(
                feats_honest,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        return all_feats

    def assemble_ranking_features(self, *args, **kwargs) -> np.ndarray:
        return self.assemble_ranking_features_batch([args[:2]], args[2], args[3])[0]
