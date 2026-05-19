"""
End-to-End Recommendation Pipeline

Single entry point for generating recommendations. This is the function
the API calls. Handles retrieval, ranking, diversity re-ranking,
business rules, and cold start.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:postgres@localhost:5432/recsys"

# ── Module-level singletons (loaded once) ────────────────────────────
_retriever = None
_ranker = None
_als_model = None
_user_idx_map: Optional[dict] = None
_item_idx_map: Optional[dict] = None
_idx_to_item: Optional[dict] = None
_idx_to_user: Optional[dict] = None
_active_items: Optional[set] = None
_item_categories: Optional[dict] = None
_initialized = False
_init_lock = threading.Lock()
_cached_cold_start: Optional[list[dict]] = None

def _sanitize_id(val) -> str:
    try:
        f_val = float(val)
        if f_val.is_integer():
            return str(int(f_val))
        return str(val)
    except (ValueError, TypeError):
        return str(val)

def _initialize():
    """Load all models and data once at startup."""
    global _retriever, _ranker, _als_model
    global _user_idx_map, _item_idx_map, _idx_to_item, _idx_to_user
    global _active_items, _item_categories, _initialized

    if _initialized:
        return
        
    with _init_lock:
        if _initialized:
            return

        try:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            model_dir = project_root / "data" / "models"

            logger.info("Initializing recommendation pipeline...")

            # Load index maps
            with open(model_dir / "user_idx_map.json", "r") as f:
                _user_idx_map = json.load(f)
            with open(model_dir / "item_idx_map.json", "r") as f:
                _item_idx_map = json.load(f)

            _idx_to_item = {v: k for k, v in _item_idx_map.items()}
            _idx_to_user = {v: k for k, v in _user_idx_map.items()}

            # Load hybrid retriever
            from app.ml.retrieval.hybrid import HybridRetriever
            _retriever = HybridRetriever(str(model_dir))
            _retriever.load()

            # Load ranker
            from app.ml.ranking.model import Ranker
            _ranker = Ranker(str(model_dir))
            _ranker.load()
            _ranker._feature_assembler._ensure_loaded()

            # Load ALS model (for MMR diversity)
            from app.ml.als.model import ALSModel
            _als_model = ALSModel(str(model_dir))
            _als_model.load()

            # Load active items and categories from DB
            try:
                from sqlalchemy import create_engine
                engine = create_engine(DB_URL)
                items_df = pd.read_sql(
                    "SELECT external_id, is_active, category FROM items",
                    engine,
                )
                items_df["external_id"] = items_df["external_id"].apply(_sanitize_id)
                
                _active_items = set(
                    items_df[items_df["is_active"] == True]["external_id"].values
                )
                _item_categories = dict(
                    zip(items_df["external_id"], items_df["category"])
                )
            except Exception as e:
                logger.warning(f"Could not load item metadata from DB: {e}")
                _active_items = None
                _item_categories = {}

            _initialized = True
            logger.info("Recommendation pipeline initialized")
        except Exception as e:
            logger.error(f"FATAL: Pipeline failed to initialize: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Still set initialized so we don't spam logs in a loop
            _initialized = True


def _mmr_rerank(
    items: list[tuple[str, float]],
    n: int,
    lambda_param: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance re-ranking for diversity.
    Optimized with NumPy vectorization.
    """
    if not items:
        return []
    if len(items) <= 1:
        return items[:n]

    # 1. Prepare data
    item_ids = [it[0] for it in items]
    rel_scores = np.array([it[1] for it in items], dtype=np.float32)

    # Normalize relevance scores to [0, 1]
    max_rel = rel_scores.max()
    min_rel = rel_scores.min()
    rel_range = max_rel - min_rel if max_rel != min_rel else 1.0
    norm_rel = (rel_scores - min_rel) / rel_range

    # 2. Get and normalize embeddings
    embeddings_list = []
    dim = 128  # Default fallback dimension
    for item_id in item_ids:
        idx = _item_idx_map.get(item_id)
        emb = None
        if idx is not None and _als_model is not None:
            try:
                emb = _als_model.get_item_embedding(int(idx))
                dim = emb.shape[0]
            except (IndexError, TypeError, AttributeError):
                pass

        if emb is not None:
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                embeddings_list.append(emb / norm)
            else:
                embeddings_list.append(None)
        else:
            embeddings_list.append(None)

    # Convert None to zero vectors of correct dimension
    final_embeddings = []
    for emb in embeddings_list:
        if emb is None:
            final_embeddings.append(np.zeros(dim, dtype=np.float32))
        else:
            final_embeddings.append(emb.astype(np.float32))

    embeddings = np.array(final_embeddings)  # Shape: (M, D)

    # 3. Iterative selection
    selected_indices = []
    remaining_indices = list(range(len(items)))

    # First item: highest relevance
    first_idx = int(np.argmax(norm_rel))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    # Keep track of max similarity to selected items for each item
    max_sims = np.dot(embeddings, embeddings[first_idx])

    while len(selected_indices) < n and remaining_indices:
        rem_array = np.array(remaining_indices)

        # mmr = λ * norm_rel - (1 - λ) * max_sim
        mmr_scores = (lambda_param * norm_rel[rem_array] - 
                      (1 - lambda_param) * max_sims[rem_array])

        best_rem_idx = np.argmax(mmr_scores)
        best_idx = int(rem_array[best_rem_idx])

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # Update max_sims: max(current_max, sim to new item)
        new_sims = np.dot(embeddings, embeddings[best_idx])
        max_sims = np.maximum(max_sims, new_sims)

    return [items[idx] for idx in selected_indices]


def _cold_start_recommend(n: int) -> list[dict]:
    """
    Cold start recommendations: top items by conversion rate
    from top-3 categories globally.
    Cached after first computation to avoid 15s+ DB query.
    """
    global _cached_cold_start
    if _cached_cold_start is not None:
        return _cached_cold_start[:n]

    try:
        from sqlalchemy import create_engine
        engine = create_engine(DB_URL)

        cat_df = pd.read_sql(
            "SELECT entity_id, feature_value_json FROM feature_store "
            "WHERE entity_type = 'item' AND feature_name = 'item_profile_v1'",
            engine,
        )

        items_data = []
        for _, row in cat_df.iterrows():
            ext_id = _sanitize_id(row["entity_id"])
            feats = row["feature_value_json"]
            category = _item_categories.get(ext_id, "unknown")
            conv_rate = feats.get("conversion_rate", 0)
            items_data.append({
                "item_id": ext_id,
                "category": category,
                "conversion_rate": conv_rate,
            })

        items_df = pd.DataFrame(items_data)

        # Top 3 categories
        top_cats = (
            items_df.groupby("category")["conversion_rate"]
            .mean()
            .nlargest(3)
            .index.tolist()
        )

        # Top items from those categories
        results = (
            items_df[items_df["category"].isin(top_cats)]
            .nlargest(200, "conversion_rate")  # Cache top 200
        )

        _cached_cold_start = [
            {
                "item_id": row["item_id"],
                "score": float(row["conversion_rate"]),
                "rank": i + 1,
                "explanation": f"Trending in {row['category']}",
                "retrieval_source": "cold_start",
            }
            for i, (_, row) in enumerate(results.iterrows())
        ]
        return _cached_cold_start[:n]
    except Exception as e:
        logger.error(f"Cold start recommendation failed: {e}")
        return []


def recommend(
    user_id: str,
    n: int = 10,
    exclude_interacted: bool = True,
    context: dict = None,
) -> list[dict]:
    """
    Generate recommendations for a user.
    Single entry point for API and Benchmark.
    """
    _initialize()
    context = context or {}

    # 1. Look up user index
    user_idx = _user_idx_map.get(user_id)
    if user_idx is not None:
        user_idx = int(user_idx)

    # 2. Retrieve candidates
    k_retrieve = max(150, n * 2)
    candidate_idxs, meta = _retriever.retrieve(user_id, user_idx, k=k_retrieve, return_meta=True)

    # 3. Filter interacted items
    if exclude_interacted:
        interacted = set(_retriever._realtime_history.get(user_id, []))
        candidate_idxs = [idx for idx in candidate_idxs if idx not in interacted]

    # 3.5 Catalog constraint for cold users to prevent cross-contamination
    if user_idx is None:
        session_items = _retriever._realtime_history.get(user_id, [])
        if session_items:
            # Check the catalog of the most recently interacted item
            last_item_id = str(_idx_to_item.get(session_items[-1], ""))
            is_amz_session = last_item_id.startswith("amz_")
            
            filtered_candidates = []
            for idx in candidate_idxs:
                c_id = str(_idx_to_item.get(idx, ""))
                if is_amz_session and c_id.startswith("amz_"):
                    filtered_candidates.append(idx)
                elif not is_amz_session and not c_id.startswith("amz_"):
                    filtered_candidates.append(idx)
            
            # Use filtered if we have enough, otherwise fallback to mixed
            if len(filtered_candidates) >= n:
                candidate_idxs = filtered_candidates

    # 4. Business rules
    if _active_items is not None:
        candidate_idxs = [
            idx for idx in candidate_idxs
            if _idx_to_item.get(idx, "") in _active_items
        ]

    if not candidate_idxs:
        return _cold_start_recommend(n)

    # 5. Rank candidates
    candidate_item_ids = [_idx_to_item[idx] for idx in candidate_idxs if idx in _idx_to_item]
    
    # Map meta index to item_id for context
    context = {}
    if meta:
        context = {user_id: {str(_idx_to_item.get(int(idx))): m for idx, m in meta.items() if _idx_to_item.get(int(idx))}}
    
    # If the user is completely new (no ALS/Neural embedding) but has clicked on items,
    # the ranker will have 0 personalization features and will just sort by global Amazon popularity.
    # To prevent this, we bypass the ranker and rely on the retriever's order (SASRec/CBF).
    if user_idx is None and _retriever._realtime_history.get(user_id):
        # Give them dummy scores matching their retrieval rank so MMR still works
        ranked = []
        for i, item_id in enumerate(candidate_item_ids):
            ranked.append((item_id, float(len(candidate_item_ids) - i)))
    else:
        ranked = _ranker.rank_candidates(user_id, candidate_item_ids, context)

    # 6. Take top candidates for MMR
    top_candidates = ranked[:n * 3]

    # 7. MMR diversity re-ranking
    diversified = _mmr_rerank(top_candidates, n, lambda_param=0.5)

    # 8. Build result dicts
    results = []
    for rank, (item_id, score) in enumerate(diversified, 1):
        category = _item_categories.get(item_id, "unknown")
        results.append({
            "item_id": item_id,
            "score": float(score),
            "rank": rank,
            "explanation": f"Based on your interest in {category}",
            "retrieval_source": "hybrid",
        })

    return results

def notify_interaction(user_id: str, item_id: str):
    """Real-time retriever update."""
    _initialize()
    item_idx = _item_idx_map.get(item_id)
    logger.info(f"notify_interaction called for user {user_id}, item {item_id}. item_idx={item_idx}")
    if item_idx is not None and _retriever is not None:
        _retriever.add_interaction(user_id, int(item_idx))
        logger.info(f"Successfully added interaction to retriever for user {user_id}.")
    else:
        logger.warning(f"Failed to add interaction. item_idx={item_idx}, _retriever_exists={_retriever is not None}")

def get_similar_items(item_id: str, n: int = 10) -> list[dict]:
    """Similar items based on ALS embedding space."""
    _initialize()
    item_idx = _item_idx_map.get(item_id)
    if item_idx is None: return []
    try:
        similar_indices = _als_model.get_similar_items(int(item_idx), n=n)
    except: return []
    return [{"item_id": _idx_to_item.get(idx)} for idx in similar_indices if _idx_to_item.get(idx)]
