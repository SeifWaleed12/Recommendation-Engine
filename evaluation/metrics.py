"""Reusable recommendation evaluation metrics."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    values = list(relevances)[:k]
    return float(sum(rel / math.log2(idx + 2) for idx, rel in enumerate(values)))


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return float(dcg / ideal) if ideal > 0 else 0.0


def recall_at_k(predicted: Sequence[str], actual: set[str], k: int) -> float:
    if not actual:
        return 0.0
    return float(len(set(predicted[:k]) & actual) / len(actual))


def average_precision(predicted: Sequence[str], actual: set[str], k: int) -> float:
    if not actual:
        return 0.0
    hits = 0
    score = 0.0
    for idx, item_id in enumerate(predicted[:k], start=1):
        if item_id in actual:
            hits += 1
            score += hits / idx
    return float(score / min(len(actual), k))


def mean_average_precision(
    predictions: Iterable[Sequence[str]],
    actuals: Iterable[set[str]],
    k: int,
) -> float:
    values = [average_precision(pred, actual, k) for pred, actual in zip(predictions, actuals)]
    return float(np.mean(values)) if values else 0.0


def catalog_coverage(recommended_items: Iterable[str], catalog_items: Iterable[str]) -> float:
    catalog = set(catalog_items)
    if not catalog:
        return 0.0
    return float(len(set(recommended_items)) / len(catalog))


def intra_list_diversity(
    recommended_items: Sequence[str],
    item_embeddings: dict[str, np.ndarray] | None = None,
) -> float:
    if len(recommended_items) < 2:
        return 0.0
    if not item_embeddings:
        return float(len(set(recommended_items)) / len(recommended_items))

    distances = []
    for i, item_a in enumerate(recommended_items):
        emb_a = item_embeddings.get(item_a)
        if emb_a is None:
            continue
        for item_b in recommended_items[i + 1:]:
            emb_b = item_embeddings.get(item_b)
            if emb_b is None:
                continue
            denom = max(float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b)), 1e-10)
            similarity = float(np.dot(emb_a, emb_b) / denom)
            distances.append(1.0 - similarity)
    return float(np.mean(distances)) if distances else 0.0


MAP = mean_average_precision
