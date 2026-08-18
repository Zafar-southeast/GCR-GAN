from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def required_ranking_depth(
    candidate_count: int,
    *,
    recall_ks: Iterable[int],
    map_k: int,
    ndcg_k: int,
) -> int:
    """Return the smallest ranking depth that preserves every requested metric.

    MAP@k, nDCG@k, and Recall@k never inspect ranks below their respective
    cutoffs. Keeping only this prefix avoids serializing a query-by-candidate
    matrix of paper IDs on large corpora without changing any reported value.
    """
    recall_ks = tuple(int(value) for value in recall_ks)
    cutoffs = (*recall_ks, int(map_k), int(ndcg_k))
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("All ranking cutoffs must be positive")
    return min(candidate_count, max(cutoffs))


def stable_top_k_indices(scores: np.ndarray, k: int | None = None) -> np.ndarray:
    """Return descending score indices with stable, deterministic tie handling.

    For a proper top-k request this is O(n) selection plus O(k log k) sorting,
    rather than sorting all candidates. At the cutoff boundary, tied items are
    selected by their original candidate order, exactly matching NumPy's stable
    full sort.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")
    if scores.size == 0:
        raise ValueError("scores cannot be empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    if k is None:
        k = int(scores.size)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), int(scores.size))
    if k == scores.size:
        return np.argsort(-scores, kind="stable")

    # np.partition identifies the kth score without ordering all candidates.
    # Explicit handling of the equality boundary makes the result reproducible
    # even when many candidates receive the same score.
    threshold = np.partition(scores, scores.size - k)[scores.size - k]
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)
    selected = np.concatenate((greater, equal[: k - greater.size]))
    order = np.lexsort((selected, -scores[selected]))
    return selected[order]
