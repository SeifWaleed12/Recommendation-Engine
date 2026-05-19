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
from scipy.sparse import csr_matrix, coo_matrix

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.ml.pipeline.recommend import _initialize, recommend, _retriever, _item_idx_map, _idx_to_item
from evaluation.benchmark import _compute_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BENCHMARK_USERS = 250
REC_SIZE = 50
RANDOM_SEED = 42
DATA_DIR = project_root / "data" / "processed"

def run_ablation(channel_name: str, uids: list[str], test_items_dict: dict, train_items_dict: dict, user_idx_map: dict):
    print(f"\n--- Running Ablation: {channel_name} ---")
    
    preds = {}
    cands = {}
    actual_truth = {}
    
    from app.ml.pipeline.recommend import _retriever, _idx_to_item
    
    # Configure Retriever for this ablation
    orig_als = _retriever._als_searcher
    orig_cbf = _retriever._cbf_searcher
    orig_neural = _retriever._neural_searcher
    orig_sasrec = _retriever._sasrec_searcher
    
    _retriever._als_searcher = orig_als if channel_name in ["ALS", "ALL", "ALS+SASRec", "ALS+CBF+SASRec"] else None
    _retriever._cbf_searcher = orig_cbf if channel_name in ["CBF", "ALL", "ALS+CBF+SASRec"] else None
    _retriever._neural_searcher = orig_neural if channel_name in ["Neural", "ALL"] else None
    _retriever._sasrec_searcher = orig_sasrec if channel_name in ["SASRec", "ALL", "ALS+SASRec", "ALS+CBF+SASRec"] else None

    # We do NOT use the ranker for ablation of candidate recall, but we need predictions to see ranker impact.
    # We will just run the standard `recommend` pipeline and capture both.
    
    for uid in tqdm(uids, desc=f"Evaluating {channel_name}"):
        history = test_items_dict[uid]
        actual_truth[uid] = set(history)
        
        # 1. Raw Candidates (k=500)
        user_idx = user_idx_map.get(uid)
        if user_idx is not None: user_idx = int(user_idx)
        raw_idx = _retriever.retrieve(uid, user_idx, k=500)
        cands[uid] = [_idx_to_item.get(idx) for idx in raw_idx if idx in _idx_to_item]
        
        # 2. Final Recommendations (Ranker output)
        try:
            recs = recommend(uid, n=REC_SIZE)
            preds[uid] = [str(r["item_id"]) for r in recs]
        except Exception:
            preds[uid] = []

    # Restore retriever
    _retriever._als_searcher = orig_als
    _retriever._cbf_searcher = orig_cbf
    _retriever._neural_searcher = orig_neural
    _retriever._sasrec_searcher = orig_sasrec

    metrics = _compute_metrics(preds, actual_truth, [], REC_SIZE, raw_candidates=cands)
    if metrics:
        print(f"  CandRec@500: {metrics.get('candidate_recall_500', 0):.4f}")
        print(f"  HR@10:       {metrics['hr_10']:.4f}")
        print(f"  NDCG@50:     {metrics['ndcg_50']:.4f}")
    return metrics

def main():
    _initialize()
    from app.ml.pipeline.recommend import _retriever, _item_idx_map
    
    print("Loading datasets...")
    train_df = pd.read_parquet(DATA_DIR / "train_interactions.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_interactions.parquet")
    
    with open(DATA_DIR / "user_idx_map.json", "r") as f:
        user_idx_map = json.load(f)

    train_items = train_df.groupby("user_ext_id")["item_ext_id"].apply(set).to_dict()
    test_items = test_df.groupby("user_ext_id")["item_ext_id"].apply(list).to_dict()
    
    # Mock retriever DB fetch (Prevent leakage)
    train_items_ordered = train_df.groupby("user_ext_id")["item_ext_id"].apply(list).to_dict()
    def mock_get_user_recent_interactions(user_id: str) -> list[int]:
        history = train_items_ordered.get(user_id, [])
        recent = history[-20:][::-1]
        idxs = []
        for ext_id in recent:
            idx = _item_idx_map.get(str(ext_id))
            if idx is not None: idxs.append(int(idx))
        return idxs
    _retriever._get_user_recent_interactions = mock_get_user_recent_interactions

    # Select Users
    warm_users = list(set(train_items.keys()) & set(test_items.keys()))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(warm_users)
    rr_warm = [u for u in warm_users if not str(u).isdigit()][:BENCHMARK_USERS]

    print(f"Running Ablation on {len(rr_warm)} RR_WARM users...")

    configs = [
        "ALS",
        "CBF",
        "Neural",
        "SASRec",
        "ALS+SASRec",
        "ALS+CBF+SASRec",
        "ALL"
    ]
    
    results = {}
    for cfg in configs:
        res = run_ablation(cfg, rr_warm, test_items, train_items, user_idx_map)
        results[cfg] = res

    report_path = project_root / "evaluation" / "reports" / "ablation_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results to {report_path}")

if __name__ == "__main__":
    main()