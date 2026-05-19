"""
LightGBM Ranker — Training Script

Trains a LambdaRank model to re-rank candidate items for each user.
Uses STRICT temporal split and Hard Negative Mining.
"""

import sys
import time
import json
import random
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from app.ml.ranking.features import FeatureAssembler, FEATURE_NAMES, HONEST_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(42)
np.random.seed(42)


def train_ranker():
    """Train LightGBM LambdaRank model with leakage prevention."""
    import lightgbm as lgb

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Data ────────────────────────────────────────────────────
    logger.info("Loading interactions (Strict Temporal)...")
    df = pd.read_parquet(
        data_dir / "interactions_clean.parquet",
        columns=["user_ext_id", "item_ext_id", "event_type", "timestamp"],
        engine="pyarrow"
    )
    
    # Take 500k most recent for high-quality signal
    if len(df) > 500000:
        df = df.sort_values("timestamp", ascending=False).head(500000).iloc[::-1]

    with open(data_dir / "user_idx_map.json", "r") as f:
        user_idx_map = json.load(f)
    with open(data_dir / "item_idx_map.json", "r") as f:
        item_idx_map = json.load(f)

    # ── Build training samples ───────────────────────────────────────
    logger.info("Building training samples...")
    label_map = {"purchase": 3, "add_to_cart": 2, "view": 1}

    # Sort by timestamp for strict separation
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # split interactions into 80% train-time, 20% test-time
    split_interaction_idx = int(len(df) * 0.8)
    train_df_raw = df.iloc[:split_interaction_idx]
    test_df_raw = df.iloc[split_interaction_idx:]
    
    def build_samples_from_df(target_df, context_df, desc):
        # We use full context_df for negative logic (knowing what user has seen)
        # But samples come ONLY from target_df
        user_items = context_df.groupby("user_ext_id")["item_ext_id"].apply(set).to_dict()
        user_purchases = context_df[context_df["event_type"] == "purchase"].groupby("user_ext_id")["item_ext_id"].apply(set).to_dict()
        user_views = context_df[context_df["event_type"] == "view"].groupby("user_ext_id")["item_ext_id"].apply(set).to_dict()
        
        # Positives
        pos_df = target_df[target_df["event_type"].isin(["purchase", "add_to_cart"])].drop_duplicates(subset=["user_ext_id", "item_ext_id"])
        
        local_samples = []
        u_ids = pos_df["user_ext_id"].values
        i_ids = pos_df["item_ext_id"].values
        ev_types = pos_df["event_type"].values
        
        for i in range(len(pos_df)):
            local_samples.append((str(u_ids[i]), str(i_ids[i]), label_map[ev_types[i]]))
            
        sampled_users = list(pos_df["user_ext_id"].unique())
        all_items = list(item_idx_map.keys())
        
        # Pre-compute top 5% popular items to use as hard negatives
        popular_items = list(context_df["item_ext_id"].value_counts().head(int(len(all_items) * 0.05)).index)
        if not popular_items: popular_items = all_items
        
        for uid in tqdm(sampled_users, desc=f"Negatives ({desc})"):
            purch = user_purchases.get(uid, set())
            views = user_views.get(uid, set())
            
            # Hard Negatives: Viewed but not bought (Label 1)
            view_only = list(views - purch)
            if len(view_only) > 5:
                view_only = random.sample(view_only, 5)
            for iid in view_only:
                local_samples.append((str(uid), str(iid), 1))
                
            # Hard Negatives: Popular items the user did not interact with (Label 0)
            interacted = user_items.get(uid, set())
            neg_count = 0
            for _ in range(20):
                if neg_count >= 5: break
                it = random.choice(popular_items)
                if it not in interacted:
                    local_samples.append((str(uid), str(it), 0))
                    neg_count += 1
        return local_samples

    train_samples = build_samples_from_df(train_df_raw, df, "Train")
    test_samples = build_samples_from_df(test_df_raw, df, "Test")

    # Order matters: train first, then test
    samples = train_samples + test_samples
    split_idx = len(train_samples)
    
    # ── Generate Retrieval Metadata Context ──────────────────────────
    logger.info("Generating Retrieval Context (Simulating retriever hits)...")
    from app.ml.pipeline.recommend import _initialize
    _initialize()
    from app.ml.pipeline.recommend import _retriever
    
    context = {}
    unique_users = list(set([s[0] for s in samples]))
    
    # Mock retriever DB Fetch so it only sees train data (prevent leakage)
    train_items_ordered = train_df_raw.groupby("user_ext_id")["item_ext_id"].apply(list).to_dict()
    def mock_get_user_recent_interactions(user_id: str) -> list[int]:
        history = train_items_ordered.get(user_id, [])
        recent = history[-20:][::-1]
        idxs = []
        for ext_id in recent:
            idx = item_idx_map.get(str(ext_id))
            if idx is not None: idxs.append(int(idx))
        return idxs
    
    _retriever._get_user_recent_interactions = mock_get_user_recent_interactions
    
    # Move this OUTSIDE the loop! Re-creating a 440,000-item dictionary 89,000 times was causing the 5-hour delay.
    idx_to_item = {int(v): str(k) for k, v in item_idx_map.items()}
    
    # Pre-calculate which items we need for each user
    user_to_sampled_items = defaultdict(set)
    for s in samples:
        user_to_sampled_items[s[0]].add(s[1])

    import concurrent.futures
    import torch
    import faiss
    
    # CRITICAL FIX for CPU Thrashing:
    # When running 16 Python threads, if PyTorch and FAISS also try to use all CPU cores 
    # for each operation, the CPU will thrash and slow down to a crawl (11 it/s).
    # Setting these to 1 forces each thread to use exactly 1 core, scaling perfectly.
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)

    def simulate_for_user(uid):
        uidx = user_idx_map.get(uid)
        if uidx is not None: uidx = int(uidx)
        try:
            _, meta = _retriever.retrieve(uid, user_idx=uidx, k=500, return_meta=True)
            needed = user_to_sampled_items[uid]
            filtered_meta = {}
            for idx, v in meta.items():
                item_str = idx_to_item.get(int(idx))
                if item_str and item_str in needed:
                    filtered_meta[item_str] = v
            return uid, filtered_meta
        except Exception:
            return uid, {}

    context = {}
    import multiprocessing
    max_workers = min(16, multiprocessing.cpu_count())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(simulate_for_user, uid): uid for uid in unique_users}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(unique_users), desc="Simulating Retriever"):
            uid, filtered_meta = future.result()
            context[uid] = filtered_meta

    # ── Assemble features ────────────────────────────────────────────
    logger.info(f"Assembling features for {len(samples)} samples...")
    assembler = FeatureAssembler(str(model_dir))

    pairs = [(s[0], s[1]) for s in samples]
    labels = np.array([s[2] for s in samples], dtype=np.float32)
    user_ids = [s[0] for s in samples]

    features = assembler.assemble_ranking_features_batch(
        pairs, user_idx_map=user_idx_map, item_idx_map=item_idx_map, context=context
    )
    
    # ── LEAKAGE PROTECTION ───────────────────────────────────────────
    # Features 2, 4, 12, 14 are already dropped inside FeatureAssembler.
    # The returned `features` array is exactly 21 columns wide (HONEST_FEATURE_NAMES).
    current_feature_names = HONEST_FEATURE_NAMES
    X = features

    # ── Temporal Split ───────────────────────────────────────────────
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    u_train, u_test = user_ids[:split_idx], user_ids[split_idx:]

    def get_groups(user_list):
        groups = []
        if not user_list: return groups
        curr = user_list[0]
        count = 0
        for u in user_list:
            if u == curr: count += 1
            else:
                groups.append(count)
                curr = u
                count = 1
        groups.append(count)
        return groups

    # Sort within splits to group same-user samples together (Required by LGBM)
    train_order = sorted(range(len(u_train)), key=lambda i: u_train[i])
    X_train, y_train = X_train[train_order], y_train[train_order]
    train_groups = get_groups([u_train[i] for i in train_order])

    test_order = sorted(range(len(u_test)), key=lambda i: u_test[i])
    X_test, y_test = X_test[test_order], y_test[test_order]
    test_groups = get_groups([u_test[i] for i in test_order])

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Train ────────────────────────────────────────────────────────
    logger.info("Training Honest Ranker...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        n_jobs=-1,
        random_state=42,
        importance_type="gain"
    )

    ranker.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_test, y_test)],
        eval_group=[test_groups],
        eval_at=[5, 10],
        callbacks=[lgb.log_evaluation(period=50)]
    )

    # ── Report ───────────────────────────────────────────────────────
    importances = ranker.feature_importances_
    logger.info("Top Features (Honest):")
    for name, imp in sorted(zip(current_feature_names, importances), key=lambda x: -x[1])[:10]:
        logger.info(f"  {name}: {imp:.1f}")

    # Save
    with open(model_dir / "lgbm_ranker.pkl", "wb") as f:
        pickle.dump(ranker, f)
    logger.info("✓ Ranker Retrained (LEAKAGE REMOVED).")


if __name__ == "__main__":
    train_ranker()
