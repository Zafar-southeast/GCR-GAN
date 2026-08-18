#!/usr/bin/env python3
"""Generate deterministic synthetic citation corpora from 128 to many thousands of papers.

The corpus is an execution/evaluation fixture, not a substitute for AMiner. Every
held-out query receives eight citations into the training candidate set so all
configured ranking metrics are evaluable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 1203
TRAIN_RATIO = 0.80

TOPICS = (
    {
        "field": ("F01", "Recommender Systems"),
        "venue": ("V01", "ACM RecSys"),
        "concepts": (
            ("C01", "Citation Recommendation"),
            ("C02", "Collaborative Filtering"),
            ("C03", "Personalization"),
        ),
        "terms": ("citation ranking", "reader preference", "scholarly recommendation"),
    },
    {
        "field": ("F02", "Graph Representation Learning"),
        "venue": ("V02", "NeurIPS"),
        "concepts": (
            ("C04", "Graph Neural Network"),
            ("C05", "Node Embedding"),
            ("C06", "Heterogeneous Network"),
        ),
        "terms": ("message passing", "graph structure", "node representation"),
    },
    {
        "field": ("F03", "Natural Language Processing"),
        "venue": ("V03", "ACL"),
        "concepts": (
            ("C07", "Scientific Language Model"),
            ("C08", "Document Embedding"),
            ("C09", "Masked Language Modeling"),
        ),
        "terms": ("scientific text", "transformer encoder", "document semantics"),
    },
    {
        "field": ("F04", "Knowledge Graphs"),
        "venue": ("V04", "ISWC"),
        "concepts": (
            ("C10", "Knowledge Graph Embedding"),
            ("C11", "Ontology Learning"),
            ("C12", "Graph Reasoning"),
        ),
        "terms": ("relational evidence", "ontology structure", "multi-hop reasoning"),
    },
    {
        "field": ("F05", "Information Retrieval"),
        "venue": ("V05", "SIGIR"),
        "concepts": (
            ("C13", "Neural Ranking"),
            ("C14", "Dense Retrieval"),
            ("C15", "Learning to Rank"),
        ),
        "terms": ("candidate retrieval", "ranking objective", "relevance estimation"),
    },
    {
        "field": ("F06", "Data Mining"),
        "venue": ("V06", "KDD"),
        "concepts": (
            ("C16", "Generative Adversarial Network"),
            ("C17", "Denoising Autoencoder"),
            ("C18", "Representation Learning"),
        ),
        "terms": ("adversarial learning", "denoising reconstruction", "latent features"),
    },
    {
        "field": ("F07", "Bibliometrics"),
        "venue": ("V07", "Scientometrics"),
        "concepts": (
            ("C19", "Citation Network"),
            ("C20", "Scientific Impact"),
            ("C21", "Research Analytics"),
        ),
        "terms": ("citation network", "research impact", "publication analysis"),
    },
    {
        "field": ("F08", "Machine Learning"),
        "venue": ("V08", "ICML"),
        "concepts": (
            ("C22", "Contrastive Learning"),
            ("C23", "Metric Learning"),
            ("C24", "Representation Evaluation"),
        ),
        "terms": ("contrastive objective", "metric space", "representation quality"),
    },
)

TITLE_PATTERNS = (
    "Robust {term} for global scholarly discovery",
    "Learning {term} from structured scientific evidence",
    "A scalable study of {term} in research graphs",
    "Improving {term} with semantic and relational signals",
    "Reliable {term} under sparse bibliographic observations",
)

FIRST_NAMES = ("Ada", "Ben", "Chen", "Dina", "Evan", "Fatima", "Grace", "Haris")
LAST_NAMES = ("Ali", "Brown", "Chen", "Garcia", "Kim", "Lee", "Noor", "Shah")


def split_ids(ids: list[str]) -> tuple[set[str], set[str]]:
    shuffled = sorted(ids)
    random.Random(SEED).shuffle(shuffled)
    boundary = int(len(shuffled) * TRAIN_RATIO)
    return set(shuffled[:boundary]), set(shuffled[boundary:])


def author(author_index: int) -> dict[str, str]:
    first = FIRST_NAMES[author_index % len(FIRST_NAMES)]
    last = LAST_NAMES[(author_index // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return {
        "id": f"A{author_index + 1:02d}",
        "name": f"{first} {last}",
        "affiliation": f"Institute {(author_index % 8) + 1}",
    }


def choose_references(
    paper_number: int,
    paper_id: str,
    all_ids: list[str],
    train_ids: set[str],
    test_ids: set[str],
) -> list[str]:
    topic_index = (paper_number - 1) % len(TOPICS)
    local_rng = random.Random(SEED + paper_number)

    def topic_of(candidate: str) -> int:
        return (int(candidate[1:]) - 1) % len(TOPICS)

    if paper_id in test_ids:
        ordered_train = sorted(train_ids)
        same_topic = [
            candidate for candidate in ordered_train if topic_of(candidate) == topic_index
        ]
        cross_topic = [
            candidate for candidate in ordered_train if topic_of(candidate) != topic_index
        ]
        local_rng.shuffle(same_topic)
        local_rng.shuffle(cross_topic)
        return sorted(same_topic[:6] + cross_topic[:2])

    earlier_train = [
        candidate
        for candidate in all_ids
        if candidate in train_ids and int(candidate[1:]) < paper_number
    ]
    same_topic = [candidate for candidate in earlier_train if topic_of(candidate) == topic_index]
    cross_topic = [candidate for candidate in earlier_train if topic_of(candidate) != topic_index]
    local_rng.shuffle(same_topic)
    local_rng.shuffle(cross_topic)
    return sorted(same_topic[:5] + cross_topic[:2])


def build_record(
    paper_number: int,
    all_ids: list[str],
    train_ids: set[str],
    test_ids: set[str],
) -> dict[str, object]:
    paper_id = f"P{paper_number:03d}"
    topic_index = (paper_number - 1) % len(TOPICS)
    topic = TOPICS[topic_index]
    term = topic["terms"][(paper_number // len(TOPICS)) % len(topic["terms"])]
    pattern = TITLE_PATTERNS[(paper_number - 1) % len(TITLE_PATTERNS)]
    title = pattern.format(term=term)
    concepts = topic["concepts"]
    method = concepts[(paper_number - 1) % len(concepts)][1]
    abstract = (
        f"We study {term} for scholarly information access. The method combines {method} "
        f"with {topic['terms'][1]} and evaluates ranking quality on structured citation data. "
        "Results examine relevance, robustness, and representation quality."
    )
    author_base = topic_index * 8
    topic_cycle = (paper_number - 1) // len(TOPICS)
    authors = [
        author(author_base + (topic_cycle % 8)),
        author(author_base + ((topic_cycle + 3) % 8)),
    ]
    if paper_id in test_ids and paper_number % 13 == 0:
        authors = []
    record: dict[str, object] = {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "year": 2024 if paper_id in test_ids else 2014 + (paper_number - 1) // 20,
        "citation_count": 10 + (paper_number * 17) % 250,
        "authors": authors,
        "venue": {"id": topic["venue"][0], "name": topic["venue"][1]},
        "references": choose_references(
            paper_number, paper_id, all_ids, train_ids, test_ids
        ),
        "fields": [{"id": topic["field"][0], "name": topic["field"][1]}],
        "concepts": [{"id": item[0], "name": item[1]} for item in concepts[:2]],
        "keywords": list(dict.fromkeys((term, topic["terms"][2]))),
    }
    if paper_number % 4 == 0:
        record["facets"] = {
            "objective": [f"Improve {term} for scientific recommendation."],
            "method": [f"Combine {method} with attributed graph evidence."],
            "result": [f"Evaluate {term} using held-out citation ranking."],
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=int, default=160)
    parser.add_argument(
        "--output", type=Path, default=Path("examples/benchmark_160/papers.jsonl")
    )
    args = parser.parse_args()
    if args.papers < 128:
        raise ValueError("Use at least 128 papers so an 80/20 split has over 100 candidates")
    all_ids = [f"P{number:03d}" for number in range(1, args.papers + 1)]
    train_ids, test_ids = split_ids(all_ids)
    records = [
        build_record(number, all_ids, train_ids, test_ids)
        for number in range(1, args.papers + 1)
    ]
    unevaluable = [
        record["id"]
        for record in records
        if record["id"] in test_ids and not (set(record["references"]) & train_ids)
    ]
    if unevaluable:
        raise RuntimeError("Every held-out paper must cite at least one training candidate")
    payload = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "papers": len(records),
                "train_candidates": len(train_ids),
                "test_queries": len(test_ids),
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
