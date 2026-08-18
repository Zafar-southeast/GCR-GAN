from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .records import PaperRecord


@dataclass(frozen=True)
class PaperSplit:
    train: tuple[str, ...]
    test: tuple[str, ...]


def random_paper_split(
    records: Sequence[PaperRecord], train_ratio: float = 0.8, seed: int = 1203
) -> PaperSplit:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must lie strictly between 0 and 1")
    ids = sorted(record.id for record in records)
    random.Random(seed).shuffle(ids)
    boundary = max(1, min(len(ids) - 1, int(len(ids) * train_ratio)))
    return PaperSplit(tuple(sorted(ids[:boundary])), tuple(sorted(ids[boundary:])))


def evaluable_truth(
    query: PaperRecord, candidate_ids: set[str], *, minimum_relevant: int = 1
) -> set[str]:
    truth = set(query.references) & candidate_ids
    return truth if len(truth) >= minimum_relevant else set()

