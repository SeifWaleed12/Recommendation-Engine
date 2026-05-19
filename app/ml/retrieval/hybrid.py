"""
Hybrid Retrieval

Combines ALS, CBF, and Neural retrieval channels using FAISS indexes.
Optimized for JIT interaction fetching to save RAM.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:postgres@localhost:5432/recsys"


class HybridRetriever:
    """
    Multi-channel retriever optimized for low memory usage.
    Fetches user interactions JIT from the database.
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self._engine = None
        self._loaded = False

        # FAISS searchers
        self._als_searcher = None
        self._cbf_searcher = None
        self._neural_searcher = None
        
        # Embeddings (Memory-Mapped)
        self._als_user_embeddings: Optional[np.ndarray] = None
        self._cbf_item_embeddings: Optional[np.ndarray] = None
        
        # Fallbacks
        self._popularity_ranking: Optional[list] = None
        self._item_idx_map: Optional[dict] = None

    def _ensure_loaded(self):
        if self._loaded: return
        self.load()

    def load(self):
        """Load FAISS indexes and pre-load small metadata."""
        from app.ml.retrieval.faiss_index import FaissSearcher
        logger.info("Initializing Hybrid Retriever (Memory Optimized)...")
        
        self._engine = create_engine(DB_URL)

        # ── FAISS Indexes (mmap enabled in FaissSearcher) ───────────
        def load_searcher(name):
            path = self.model_dir / name
            if path.exists():
                s = FaissSearcher()
                s.load_index(str(path))
                return s
            return None

        self._als_searcher = load_searcher("faiss_als.index")
        self._cbf_searcher = load_searcher("faiss_cbf.index")
        self._neural_searcher = load_searcher("faiss_neural.index")

        # ── Embeddings (mmap) ───────────────────────────────────────
        def load_npy(name):
            path = self.model_dir / name
            return np.load(path, mmap_mode='r') if path.exists() else None

        self._als_user_embeddings = load_npy("user_embeddings_als.npy")
        self._cbf_item_embeddings = load_npy("item_pca_embeddings.npy")

        # ── Item ID Map (Needed for JIT) ────────────────────────────
        with open(self.model_dir / "item_idx_map.json", "r") as f:
            self._item_idx_map = json.load(f)

        # ── SASRec Sequential Channel ───────────────────────────────
        self._sasrec_model = None
        self._sasrec_searcher = None
        sasrec_path = self.model_dir / "sasrec_retriever.pt"
        if sasrec_path.exists():
            import torch
            from app.ml.retrieval.sasrec_model import SASRec
            from app.ml.retrieval.faiss_index import FaissSearcher
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._sasrec_model = SASRec(item_count=len(self._item_idx_map), max_seq_len=50, hidden_units=128)
            self._sasrec_model.load_state_dict(torch.load(sasrec_path, map_location=self.device))
            self._sasrec_model.to(self.device)
            self._sasrec_model.eval()
            
            # Extract item embeddings and build index
            with torch.no_grad():
                item_embs = self._sasrec_model.item_emb.weight.data.cpu().numpy()
            self._sasrec_searcher = FaissSearcher()
            self._sasrec_searcher.add_vectors(item_embs, normalize=True)
            logger.info("Built SASRec FAISS index for sequential retrieval")

        # ── Popularity ranking (Pre-built) ──────────────────────────
        self._popularity_ranking = self._build_popularity_ranking()

        self._loaded = True
        logger.info("Hybrid retriever initialized successfully")

    def _build_popularity_ranking(self) -> list[int]:
        """Build item popularity ranking from DB (limit to top 1000)."""
        try:
            # Query top 1000 items only to save RAM
            query = """
                SELECT entity_id, (feature_value_json->>'global_purchase_count')::int as purchases
                FROM feature_store 
                WHERE entity_type = 'item' AND feature_name = 'item_profile_v1'
                ORDER BY purchases DESC LIMIT 1000
            """
            df = pd.read_sql(query, self._engine)
            idxs = []
            for eid in df["entity_id"]:
                idx = self._item_idx_map.get(str(eid))
                if idx is not None: idxs.append(int(idx))
            return idxs
        except Exception as e:
            logger.warning(f"Could not build popularity ranking: {e}")
            return []

    @lru_cache(maxsize=10000)
    def _get_user_recent_interactions(self, user_id: str) -> list[int]:
        """Fetch user's recent items from DB JIT (Cached)."""
        try:
            # Query the interactions table directly for this specific user
            query = f"""
                SELECT item_ext_id FROM interactions 
                WHERE user_ext_id = '{user_id}' 
                ORDER BY timestamp DESC LIMIT 20
            """
            df = pd.read_sql(query, self._engine)
            
            idxs = []
            for eid in df["item_ext_id"]:
                idx = self._item_idx_map.get(str(eid))
                if idx is not None: idxs.append(int(idx))
            return idxs
        except Exception as e:
            logger.debug(f"DB History fetch failed for {user_id}: {e}")
            return []

    # Local in-memory state for real-time benchmark updates
    _realtime_history = defaultdict(list)

    def add_interaction(self, user_id: str, item_idx: int):
        """Update local state for current session."""
        uid = str(user_id)
        self._realtime_history[uid].append(int(item_idx))
        if len(self._realtime_history[uid]) > 20:
            self._realtime_history[uid] = self._realtime_history[uid][-20:]

    def retrieve(self, user_id: str, user_idx: Optional[int] = None, k: int = 500, return_meta: bool = False):
        self._ensure_loaded()
        
        # Combine JIT DB history + Real-time session history
        db_history = self._get_user_recent_interactions(str(user_id))
        session_history = self._realtime_history.get(str(user_id), [])
        
        db_history_ordered = db_history[::-1]
        combined = db_history_ordered + session_history
        
        recent_history = []
        seen = set()
        for idx in reversed(combined):
            if idx not in seen:
                seen.add(idx)
                recent_history.insert(0, idx)
        
        if not recent_history and user_idx is None:
            res = self._popularity_ranking[:k]
            if return_meta:
                meta = {str(idx): {"pop": 1, "als": 0, "sasrec": 0, "cbf": 0, "neural": 0} for idx in res}
                return res, meta
            return res

        ordered_candidates = []
        seen_candidates = set()
        
        # Metadata tracking
        meta = defaultdict(lambda: {"pop": 0, "als": 0, "sasrec": 0, "cbf": 0, "neural": 0})
        
        def add_candidates(indices, channel):
            for idx in indices:
                idx = int(idx)
                if idx >= 0:
                    meta[str(idx)][channel] = 1
                    if idx not in seen_candidates:
                        ordered_candidates.append(idx)
                        seen_candidates.add(idx)

        # 1. SASRec
        if self._sasrec_model and self._sasrec_searcher and recent_history:
            try:
                import torch
                seq = [idx + 1 for idx in recent_history[-50:]]
                if len(seq) < 50:
                    seq = [0] * (50 - len(seq)) + seq
                seq_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    user_emb = self._sasrec_model(seq_tensor).cpu().numpy()[0]
                _, indices = self._sasrec_searcher.search(user_emb, k=300)
                add_candidates([idx - 1 for idx in indices[indices > 0]], "sasrec")
            except Exception as e:
                logger.error(f"SASRec retrieval failed: {e}")

        # 2. CBF
        if self._cbf_searcher and self._cbf_item_embeddings is not None and recent_history:
            try:
                history_embeds = self._cbf_item_embeddings[recent_history]
                user_profile = np.mean(history_embeds, axis=0)
                _, indices = self._cbf_searcher.search(user_profile, k=100)
                add_candidates(indices[indices >= 0], "cbf")
            except Exception: pass

        # 3. ALS
        if user_idx is not None and self._als_searcher and self._als_user_embeddings is not None:
            try:
                user_vec = self._als_user_embeddings[user_idx]
                _, indices = self._als_searcher.search(user_vec, k=250)
                add_candidates(indices[indices >= 0], "als")
            except Exception: pass

        # 4. Neural
        if user_idx is not None and self._neural_searcher and self._als_user_embeddings is not None:
            try:
                user_vec = self._als_user_embeddings[user_idx]
                _, indices = self._neural_searcher.search(user_vec, k=150)
                add_candidates(indices[indices >= 0], "neural")
            except Exception: pass

        # 5. Padding/Deduplication
        result = ordered_candidates[:k]
        if len(result) < k and self._popularity_ranking:
            add_candidates(self._popularity_ranking, "pop")
            result = ordered_candidates[:k]
        
        if return_meta:
            return result, dict(meta)
        return result
