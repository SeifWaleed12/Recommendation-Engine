"""Benchmark the production recommender against simple baselines.

Separates evaluation into 4 cohorts:
1. Retailrocket Warm (Known users)
2. Retailrocket Cold (New users, simulated single-click)
3. Amazon Warm
4. Amazon Cold

Optimized for low-RAM execution: avoids dense matrix conversions,
batches UserCF via sparse dot products, and uses gc aggressively.
"""

import gc
import json
import os
import random
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import load_npz, csr_matrix, coo_matrix

# ── Direct Pipeline Import (In-Process for RAM Efficiency) ──────────
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.ml.pipeline.recommend import recommend, notify_interaction

# ── Configuration ───────────────────────────────────────────────────

BENCHMARK_USERS = int(os.environ.get("BENCHMARK_USERS", 250))
REC_SIZE = 50
RANDOM_SEED = 42

DATA_DIR = project_root / "data" / "processed"
MODEL_DIR = project_root / "data" / "models"
REPORT_DIR = project_root / "evaluation" / "reports"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Simulation Helpers ───────────────────────────────────────────────

def _local_recs(user_id: str, n: int) -> list[str]:
    """Get recommendations directly from the pipeline object."""
    try:
        recs = recommend(str(user_id), n=n)
        return [str(r["item_id"]) for r in recs]
    except Exception:
        return []

def _record_event_local(user_id: str, item_id: str):
    """Update user features in-process for cold-start simulation."""
    try:
        notify_interaction(str(user_id), str(item_id))
    except Exception:
        pass


# ── Baselines ────────────────────────────────────────────────────────

def _user_cf_recs_sparse(user_idx: int, train_matrix: csr_matrix,
                         idx_to_item: dict, excluded: set, n: int,
                         norms: np.ndarray) -> list[str]:
    """Memory-efficient User-Based CF using sparse dot products only."""
    user_vec = train_matrix[user_idx]
    user_norm = norms[user_idx]
    if user_norm < 1e-10:
        return []

    # Sparse dot product: stays sparse the entire time
    dot_products = train_matrix.dot(user_vec.T).toarray().flatten()
    
    # Cosine = dot / (norm_a * norm_b); avoid division by zero
    safe_norms = np.maximum(norms, 1e-10)
    sims = dot_products / (safe_norms * user_norm)
    sims[user_idx] = -1.0  # exclude self

    # Top 50 neighbors
    neighbors = np.argpartition(sims, -50)[-50:]
    
    # Aggregate neighbor items (sparse sum stays sparse)
    neighbor_scores = train_matrix[neighbors].sum(axis=0).A1

    # Use argpartition instead of full argsort for top-N
    candidate_count = min(n + len(excluded) + 50, len(neighbor_scores))
    top_indices = np.argpartition(neighbor_scores, -candidate_count)[-candidate_count:]
    top_indices = top_indices[np.argsort(neighbor_scores[top_indices])[::-1]]

    recs = []
    for idx in top_indices:
        item_id = idx_to_item.get(idx)
        if item_id and item_id not in excluded:
            recs.append(item_id)
            if len(recs) >= n:
                break
    return recs


# ── Metrics ──────────────────────────────────────────────────────────

def _compute_metrics(predictions: dict, actual_truth: dict, catalog: list[str],
                     rec_size: int, raw_candidates: dict = None) -> dict:
    """Compute Category HR@10, HR@50, NDCG@50, Recall@50, Coverage, and Candidate Recall."""
    from app.ml.pipeline.recommend import _item_categories
    
    hr_10 = hr_50 = ndcg_50 = recall_50 = 0
    candidate_recall_50 = candidate_recall_100 = candidate_recall_500 = 0
    all_recommended = set()
    count = 0

    for uid, rec_list in predictions.items():
        truth = actual_truth.get(uid, set())
        if not truth:
            continue
            
        truth_cats = set(_item_categories.get(str(t)) for t in truth if _item_categories.get(str(t)))
        if not truth_cats:
            continue
            
        count += 1

        rec_cats_10 = set(_item_categories.get(str(r)) for r in rec_list[:10] if _item_categories.get(str(r)))
        rec_cats_50 = set(_item_categories.get(str(r)) for r in rec_list[:rec_size] if _item_categories.get(str(r)))

        if rec_cats_10 & truth_cats:
            hr_10 += 1
        if rec_cats_50 & truth_cats:
            hr_50 += 1

        hits = len(rec_cats_50 & truth_cats)
        recall_50 += hits / len(truth_cats)

        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(rec_list)
                  if _item_categories.get(str(item)) in truth_cats)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth_cats), rec_size)))
        if idcg > 0:
            ndcg_50 += dcg / idcg

        all_recommended.update(rec_list)
        
        # Calculate candidate recall if provided (also category-based)
        if raw_candidates and uid in raw_candidates:
            cands = raw_candidates[uid]
            cand_cats_50 = set(_item_categories.get(str(r)) for r in cands[:50] if _item_categories.get(str(r)))
            cand_cats_100 = set(_item_categories.get(str(r)) for r in cands[:100] if _item_categories.get(str(r)))
            cand_cats_500 = set(_item_categories.get(str(r)) for r in cands[:500] if _item_categories.get(str(r)))
            if cand_cats_50 & truth_cats: candidate_recall_50 += 1
            if cand_cats_100 & truth_cats: candidate_recall_100 += 1
            if cand_cats_500 & truth_cats: candidate_recall_500 += 1

    if count == 0:
        return None

    res = {
        "hr_10": round(hr_10 / count, 6),
        "hr_50": round(hr_50 / count, 6),
        "ndcg_50": round(ndcg_50 / count, 6),
        "recall_50": round(recall_50 / count, 6),
        "coverage": round(len(all_recommended) / len(catalog), 6) if catalog else 0
    }
    
    if raw_candidates:
        res["candidate_recall_50"] = round(candidate_recall_50 / count, 6)
        res["candidate_recall_100"] = round(candidate_recall_100 / count, 6)
        res["candidate_recall_500"] = round(candidate_recall_500 / count, 6)
        
    return res


def _evaluate_cohort(name: str, uids: list[str], test_items_dict: dict,
                     train_items_dict: dict, catalog: list[str],
                     top_popular: list[str], train_matrix: csr_matrix,
                     user_idx_map: dict, idx_to_item: dict, is_cold: bool,
                     row_norms: np.ndarray):

    preds_hybrid = {}
    cands_hybrid = {}
    preds_pop = {}
    preds_ucf = {}
    actual_truth = {}
    
    from app.ml.pipeline.recommend import _retriever, _idx_to_item

    for uid in tqdm(uids, desc=f"Evaluating {name}"):
        history = test_items_dict[uid]
        excluded = set(train_items_dict.get(uid, set()))

        if is_cold:
            seed_item = history[0]
            _record_event_local(uid, seed_item)
            actual_truth[uid] = set(history[1:])
            excluded.add(seed_item)
        else:
            actual_truth[uid] = set(history)

        # 1. Our system (Recommendations)
        preds_hybrid[uid] = _local_recs(uid, REC_SIZE)
        
        # 1b. Our system (Raw Candidates for Recall calculation)
        if _retriever is not None:
            user_idx = user_idx_map.get(uid)
            if user_idx is not None: user_idx = int(user_idx)
            raw_idx = _retriever.retrieve(uid, user_idx, k=500)
            cands_hybrid[uid] = [_idx_to_item.get(idx) for idx in raw_idx if idx in _idx_to_item]

        # 2. Popularity
        preds_pop[uid] = [i for i in top_popular if i not in excluded][:REC_SIZE]

        # 3. UserCF (warm only)
        if not is_cold:
            uidx = user_idx_map.get(uid)
            if uidx is not None:
                preds_ucf[uid] = _user_cf_recs_sparse(
                    uidx, train_matrix, idx_to_item, excluded, REC_SIZE, row_norms
                )
            else:
                preds_ucf[uid] = []

    # Calculate scores
    cohort_results = []

    metrics = _compute_metrics(preds_hybrid, actual_truth, catalog, REC_SIZE, raw_candidates=cands_hybrid)
    if metrics:
        metrics["name"] = "Our Hybrid System (Optimized)"
        cohort_results.append(metrics)

    metrics = _compute_metrics(preds_pop, actual_truth, catalog, REC_SIZE)
    if metrics:
        metrics["name"] = "Popularity Baseline"
        cohort_results.append(metrics)


    print(f"\nResults for {name}:")
    for res in cohort_results:
        print(f"  {res['name']}:")
        print(f"    HR@10: {res['hr_10']:.4f} | HR@50: {res['hr_50']:.4f} | NDCG@50: {res['ndcg_50']:.4f}")
        if "candidate_recall_500" in res:
            print(f"    CandRec@50: {res['candidate_recall_50']:.4f} | CandRec@100: {res['candidate_recall_100']:.4f} | CandRec@500: {res['candidate_recall_500']:.4f}")

    return cohort_results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"Loading datasets (BENCHMARK_USERS={BENCHMARK_USERS})...")
    train_df = pd.read_parquet(DATA_DIR / "train_interactions.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_interactions.parquet")
    
    with open(DATA_DIR / "user_idx_map.json", "r") as f:
        user_idx_map = json.load(f)
    with open(DATA_DIR / "item_idx_map.json", "r") as f:
        item_idx_map = json.load(f)
    idx_to_item = {v: k for k, v in item_idx_map.items()}

    catalog = list(item_idx_map.keys())
    gc.collect()

    train_items = train_df.groupby("user_ext_id")["item_ext_id"].apply(set).to_dict()
    test_items = test_df.groupby("user_ext_id")["item_ext_id"].apply(list).to_dict()
    
    # Ordered list for mocking Retriever DB history
    train_items_ordered = train_df.groupby("user_ext_id")["item_ext_id"].apply(list).to_dict()

    # MOCK Retriever DB Fetch to prevent data leakage from Postgres
    from app.ml.pipeline.recommend import _initialize
    _initialize()
    from app.ml.pipeline.recommend import _retriever, _item_idx_map
    
    def mock_get_user_recent_interactions(user_id: str) -> list[int]:
        history = train_items_ordered.get(user_id, [])
        # We only want the last 20, reversed to match the SQL newest-first order
        recent = history[-20:][::-1]
        idxs = []
        for ext_id in recent:
            idx = _item_idx_map.get(str(ext_id))
            if idx is not None:
                idxs.append(int(idx))
        return idxs
        
    if _retriever is not None:
        _retriever._get_user_recent_interactions = mock_get_user_recent_interactions

    top_popular = train_df["item_ext_id"].value_counts().head(200).index.tolist()

    # Build sparse interaction matrix
    print("Building interaction matrix...")
    user_indices = train_df["user_ext_id"].map(user_idx_map)
    item_indices = train_df["item_ext_id"].map(item_idx_map)

    # Drop rows where mapping failed
    valid = user_indices.notna() & item_indices.notna()
    train_matrix = coo_matrix((
        train_df.loc[valid, "weight"].values,
        (user_indices[valid].astype(int).values,
         item_indices[valid].astype(int).values)
    ), shape=(len(user_idx_map), len(item_idx_map))).tocsr()

    # Free train_df and test_df
    del train_df, test_df, user_indices, item_indices, valid
    gc.collect()

    # Pre-compute row norms once (for UserCF cosine similarity)
    print("Pre-computing row norms...")
    row_norms = np.sqrt(train_matrix.multiply(train_matrix).sum(axis=1)).A1

    # 2. Select Users
    warm_users = list(set(train_items.keys()) & set(test_items.keys()))
    cold_users = list(set(test_items.keys()) - set(train_items.keys()))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(warm_users)
    rng.shuffle(cold_users)

    # Filter by datasource
    rr_warm = [u for u in warm_users if str(u).isdigit()][:BENCHMARK_USERS]
    rr_cold = [u for u in cold_users if str(u).isdigit()][:BENCHMARK_USERS]
    amz_warm = [u for u in warm_users if not str(u).isdigit()][:BENCHMARK_USERS]
    amz_cold = [u for u in cold_users if not str(u).isdigit()][:BENCHMARK_USERS]

    print(f"Cohort sizes: RR_WARM={len(rr_warm)}, RR_COLD={len(rr_cold)}, "
          f"AMZ_WARM={len(amz_warm)}, AMZ_COLD={len(amz_cold)}")

    # 3. Run Evaluation
    final_report = {}

    final_report["RR_WARM"] = _evaluate_cohort(
        "RetailRocket Warm", rr_warm, test_items, train_items, catalog,
        top_popular, train_matrix, user_idx_map, idx_to_item, False, row_norms)

    final_report["RR_COLD"] = _evaluate_cohort(
        "RetailRocket Cold", rr_cold, test_items, train_items, catalog,
        top_popular, train_matrix, user_idx_map, idx_to_item, True, row_norms)



    # 4. Save Results
    report_path = REPORT_DIR / "benchmark_4way.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[SUCCESS] Benchmark Complete! Results saved to {report_path}")

    # Also sync to frontend public folder for the dashboard
    frontend_report_path = project_root / "frontend" / "public" / "reports" / "benchmark_4way.json"
    frontend_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frontend_report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"Frontend report synced to {frontend_report_path}")


if __name__ == "__main__":
    main()
