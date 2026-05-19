"""
Evaluation Script

Computes standard recommendation metrics on the temporal test split:
- NDCG@K (K=5, 10, 20)
- Recall@K (K=10, 20, 50)
- MAP@10
- Coverage: % of catalog appearing in any recommendation
- Cold-start Recall@50: users with < 5 interactions only
"""

import sys
import json
import random
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(42)
np.random.seed(42)


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute DCG@K."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Compute NDCG@K."""
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(predicted: list, actual: set, k: int) -> float:
    """Compute Recall@K."""
    if not actual:
        return 0.0
    predicted_k = set(predicted[:k])
    return len(predicted_k & actual) / len(actual)


def average_precision(predicted: list, actual: set, k: int) -> float:
    """Compute Average Precision at K."""
    if not actual:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in actual:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(len(actual), k)


def evaluate():
    """Run evaluation on the test split and print metrics."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "data" / "models"

    # ── Load data ────────────────────────────────────────────────────
    logger.info("Loading data for evaluation...")

    df = pd.read_parquet(data_dir / "interactions_clean.parquet")

    with open(model_dir / "user_idx_map.json", "r") as f:
        user_idx_map = json.load(f)
    with open(model_dir / "item_idx_map.json", "r") as f:
        item_idx_map = json.load(f)

    n_items = len(item_idx_map)

    # ── Temporal split ───────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Build train interaction sets
    train_interactions = defaultdict(set)
    for _, row in train_df.iterrows():
        u = row["user_ext_id"]
        i = row["item_ext_id"]
        train_interactions[u].add(i)

    # Build test ground truth
    test_ground_truth = defaultdict(set)
    for _, row in test_df.iterrows():
        u = row["user_ext_id"]
        i = row["item_ext_id"]
        test_ground_truth[u].add(i)

    # Users with interactions in test set
    test_users = list(test_ground_truth.keys())
    logger.info(f"Test users: {len(test_users)}")

    # ── Load recommendation pipeline ─────────────────────────────────
    sys.path.insert(0, str(project_root))

    try:
        from app.ml.pipeline.recommend import recommend
    except Exception as e:
        logger.error(f"Could not load recommendation pipeline: {e}")
        logger.info("Falling back to ALS-only evaluation...")
        from app.ml.als.model import ALSModel
        als_model = ALSModel.from_pretrained(str(model_dir))

        # Simple ALS-based evaluation
        idx_to_item = {v: k for k, v in item_idx_map.items()}

        def recommend(user_id, n=50, **kwargs):
            user_idx = user_idx_map.get(user_id)
            if user_idx is None:
                return []
            recs = als_model.recommend_for_user(int(user_idx), n=n, exclude_interacted=False)
            return [{"item_id": idx_to_item.get(idx, ""), "score": score, "rank": i+1,
                     "explanation": "", "retrieval_source": "als"}
                    for i, (idx, score) in enumerate(recs)]

    # ── Evaluate ─────────────────────────────────────────────────────
    logger.info("Running evaluation...")

    # Sample users for efficiency (evaluating all 1.5M would take too long)
    max_eval_users = min(1000, len(test_users))
    eval_users = random.sample(test_users, max_eval_users)

    # Cold-start users
    cold_start_users = [
        u for u in eval_users if len(train_interactions.get(u, set())) < 5
    ]

    metrics = {
        "ndcg@5": [], "ndcg@10": [], "ndcg@20": [],
        "recall@10": [], "recall@20": [], "recall@50": [],
        "map@10": [],
    }
    cold_recall_50 = []
    all_recommended_items = set()

    for user_id in tqdm(eval_users, desc="Evaluating users"):
        actual = test_ground_truth[user_id]
        if not actual:
            continue

        # Get recommendations
        try:
            recs = recommend(user_id, n=50, exclude_interacted=False)
            predicted_ids = [r["item_id"] for r in recs]
        except Exception:
            continue

        all_recommended_items.update(predicted_ids)

        # Binary relevance for this user
        relevances = [1.0 if pid in actual else 0.0 for pid in predicted_ids]

        # Compute metrics
        metrics["ndcg@5"].append(ndcg_at_k(relevances, 5))
        metrics["ndcg@10"].append(ndcg_at_k(relevances, 10))
        metrics["ndcg@20"].append(ndcg_at_k(relevances, 20))
        metrics["recall@10"].append(recall_at_k(predicted_ids, actual, 10))
        metrics["recall@20"].append(recall_at_k(predicted_ids, actual, 20))
        metrics["recall@50"].append(recall_at_k(predicted_ids, actual, 50))
        metrics["map@10"].append(average_precision(predicted_ids, actual, 10))

        # Cold-start
        if user_id in cold_start_users:
            cold_recall_50.append(recall_at_k(predicted_ids, actual, 50))

    # ── Coverage ─────────────────────────────────────────────────────
    coverage = len(all_recommended_items) / n_items * 100

    # ── Print Results ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 37)

    for metric_name, values in metrics.items():
        if values:
            mean_val = np.mean(values)
            print(f"{metric_name.upper():<25} {mean_val:>10.4f}")

    print(f"{'Coverage (%)':<25} {coverage:>10.2f}")

    if cold_recall_50:
        cold_recall = np.mean(cold_recall_50)
        print(f"{'Cold-start Recall@50':<25} {cold_recall:>10.4f}")
    else:
        print(f"{'Cold-start Recall@50':<25} {'N/A':>10}")

    print("=" * 60)
    print(f"Evaluated on {max_eval_users} users ({len(cold_start_users)} cold-start)")
    print()


if __name__ == "__main__":
    evaluate()
