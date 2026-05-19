"""Compare LightGBM and DCN rankers on shared candidate pools."""

import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.ml.ranking.model import DCNRanker, Ranker
from evaluation.metrics import average_precision, ndcg_at_k, recall_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
EVAL_USERS = int(os.environ.get("RANKER_EVAL_USERS", 500))
NEGATIVES_PER_USER = int(os.environ.get("RANKER_EVAL_NEGATIVES", 200))
K = 10


def precision_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    if not predicted:
        return 0.0
    return len(set(predicted[:k]) & actual) / min(k, len(predicted))


def build_candidate_pool(
    actual: set[str],
    excluded: set[str],
    catalog: list[str],
    rng: random.Random,
) -> list[str]:
    candidates = set(actual)
    attempts = 0
    max_attempts = NEGATIVES_PER_USER * 20

    while len(candidates) < len(actual) + NEGATIVES_PER_USER and attempts < max_attempts:
        item_id = rng.choice(catalog)
        attempts += 1
        if item_id not in actual and item_id not in excluded:
            candidates.add(item_id)

    candidate_list = list(candidates)
    rng.shuffle(candidate_list)
    return candidate_list


def score_ranker(ranker, users: list[str], ground_truth: dict, train_items: dict,
                 catalog: list[str], name: str) -> dict:
    rng = random.Random(RANDOM_SEED)
    metrics = {
        "ndcg@10": [],
        "recall@10": [],
        "precision@10": [],
        "map@10": [],
    }
    latencies_ms = []
    failures = 0

    for user_id in tqdm(users, desc=f"Evaluating {name}"):
        actual = ground_truth[user_id]
        excluded = train_items.get(user_id, set())
        candidates = build_candidate_pool(actual, excluded, catalog, rng)

        try:
            start = time.perf_counter()
            ranked = ranker.rank_candidates(str(user_id), candidates)
            latencies_ms.append((time.perf_counter() - start) * 1000)
        except Exception as exc:
            failures += 1
            logger.warning("%s failed for user %s: %s", name, user_id, exc)
            continue

        predicted = [item_id for item_id, _ in ranked]
        relevances = [1.0 if item_id in actual else 0.0 for item_id in predicted]

        metrics["ndcg@10"].append(ndcg_at_k(relevances, K))
        metrics["recall@10"].append(recall_at_k(predicted, actual, K))
        metrics["precision@10"].append(precision_at_k(predicted, actual, K))
        metrics["map@10"].append(average_precision(predicted, actual, K))

    result = {
        metric: float(np.mean(values)) if values else 0.0
        for metric, values in metrics.items()
    }
    result["latency_ms_mean"] = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    result["latency_ms_p95"] = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    result["evaluated_users"] = len(latencies_ms)
    result["failures"] = failures
    return result


def print_results(results: dict) -> None:
    print("\n" + "=" * 72)
    print("RANKER COMPARISON")
    print("=" * 72)
    print(f"{'Metric':<20} {'LightGBM':>14} {'DCN':>14} {'Winner':>14}")
    print("-" * 72)

    higher_is_better = {"ndcg@10", "recall@10", "precision@10", "map@10"}
    for metric in [
        "ndcg@10",
        "recall@10",
        "precision@10",
        "map@10",
        "latency_ms_mean",
        "latency_ms_p95",
    ]:
        lgbm = results["lightgbm"][metric]
        dcn = results["dcn"][metric]
        if metric in higher_is_better:
            winner = "LightGBM" if lgbm >= dcn else "DCN"
        else:
            winner = "LightGBM" if lgbm <= dcn else "DCN"
        print(f"{metric:<20} {lgbm:>14.6f} {dcn:>14.6f} {winner:>14}")

    print("-" * 72)
    print(f"LightGBM users: {results['lightgbm']['evaluated_users']} failures: {results['lightgbm']['failures']}")
    print(f"DCN users:      {results['dcn']['evaluated_users']} failures: {results['dcn']['failures']}")
    print("=" * 72)


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "data" / "models"

    logger.info("Loading temporal split data...")
    interactions = pd.read_parquet(
        data_dir / "interactions_clean.parquet",
        columns=["user_ext_id", "item_ext_id", "timestamp"],
    ).sort_values("timestamp").reset_index(drop=True)

    with open(data_dir / "item_idx_map.json", "r") as f:
        item_idx_map = json.load(f)

    split_idx = int(len(interactions) * 0.8)
    train_df = interactions.iloc[:split_idx]
    test_df = interactions.iloc[split_idx:]

    train_items = {
        str(user_id): {str(item_id) for item_id in item_ids}
        for user_id, item_ids in train_df.groupby("user_ext_id")["item_ext_id"].apply(set).items()
    }
    ground_truth = defaultdict(set)
    for user_id, item_id in zip(test_df["user_ext_id"].values, test_df["item_ext_id"].values):
        ground_truth[str(user_id)].add(str(item_id))

    users = [user_id for user_id, actual in ground_truth.items() if actual]
    rng.shuffle(users)
    users = users[: min(EVAL_USERS, len(users))]
    catalog = list(item_idx_map.keys())

    logger.info("Loading rankers...")
    lgbm_ranker = Ranker.from_pretrained(str(model_dir))
    dcn_ranker = DCNRanker.from_pretrained(str(model_dir))

    results = {
        "lightgbm": score_ranker(lgbm_ranker, users, ground_truth, train_items, catalog, "LightGBM"),
        "dcn": score_ranker(dcn_ranker, users, ground_truth, train_items, catalog, "DCN"),
    }

    print_results(results)


if __name__ == "__main__":
    main()
