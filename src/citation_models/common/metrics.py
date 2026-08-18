from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    seen: set[str] = set()
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant and item not in seen:
            hits += 1
            score += hits / rank
            seen.add(item)
    # GCR-GAN Equation 10 divides by all ground-truth positives (GTP).
    return score / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant and item not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(item)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    truth: Mapping[str, set[str]],
    *,
    recall_ks: Iterable[int] = (20, 40, 60, 80, 100),
    map_k: int = 10,
    ndcg_k: int = 100,
) -> dict[str, float | int]:
    recall_ks = tuple(recall_ks)
    if map_k <= 0 or ndcg_k <= 0 or any(k <= 0 for k in recall_ks):
        raise ValueError("All metric cutoffs must be positive")
    query_ids = sorted(set(rankings) & set(truth))
    query_ids = [qid for qid in query_ids if truth[qid]]
    if not query_ids:
        raise ValueError("No evaluable queries have at least one relevant candidate")

    output: dict[str, float | int] = {"queries": len(query_ids)}
    output[f"MAP@{map_k}"] = sum(
        average_precision_at_k(rankings[q], truth[q], map_k) for q in query_ids
    ) / len(query_ids)
    output[f"nDCG@{ndcg_k}"] = sum(
        ndcg_at_k(rankings[q], truth[q], ndcg_k) for q in query_ids
    ) / len(query_ids)
    for k in recall_ks:
        output[f"Recall@{k}"] = sum(
            recall_at_k(rankings[q], truth[q], k) for q in query_ids
        ) / len(query_ids)
    return output
