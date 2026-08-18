from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from citation_models.common.records import NamedEntity, PaperRecord


def paper_node(paper_id: str) -> str:
    return f"paper::{paper_id}"


def author_node(author_id: str) -> str:
    return f"author::{author_id}"


def topic_node(topic_id: str) -> str:
    return f"topic::{topic_id}"


@dataclass
class HeterogeneousBibliographicNetwork:
    node_ids: list[str]
    node_types: list[str]
    texts: list[str]
    adjacency: sparse.csr_matrix

    @property
    def node_to_index(self) -> dict[str, int]:
        return {node_id: i for i, node_id in enumerate(self.node_ids)}


def build_hbn(
    records: Iterable[PaperRecord],
    *,
    undirected: bool = True,
    include_coauthor_edges: bool = True,
) -> HeterogeneousBibliographicNetwork:
    """Build the paper/author/topic HBN defined in Section 3 of GCR-GAN.

    Topic nodes combine fields of study, concepts, and keywords because the two
    AMiner releases expose topical metadata under different keys.
    """
    records = list(records)
    paper_ids = {record.id for record in records}
    author_texts: dict[str, list[str]] = {}
    topic_labels: dict[str, str] = {}
    node_type: dict[str, str] = {}
    node_text: dict[str, str] = {}
    edges: set[tuple[str, str]] = set()

    def connect(source: str, target: str) -> None:
        if source == target:
            return
        edges.add((source, target))
        if undirected:
            edges.add((target, source))

    for paper in records:
        pid = paper_node(paper.id)
        node_type[pid] = "paper"
        node_text[pid] = paper.specter_text

        author_nodes: list[str] = []
        for author in paper.authors:
            aid = author_node(author.id)
            author_nodes.append(aid)
            node_type[aid] = "author"
            author_texts.setdefault(aid, []).append(paper.specter_text)
            connect(pid, aid)
        if include_coauthor_edges:
            for i, source in enumerate(author_nodes):
                for target in author_nodes[i + 1 :]:
                    connect(source, target)

        topical_entities = [*paper.fields, *paper.concepts]
        for keyword in paper.keywords:
            topical_entities.append(NamedEntity(id=keyword.casefold(), name=keyword))
        for topic in topical_entities:
            tid = topic_node(topic.id)
            node_type[tid] = "topic"
            topic_labels[tid] = topic.name
            connect(pid, tid)

        for cited_id in paper.references:
            if cited_id in paper_ids:
                connect(pid, paper_node(cited_id))

    for aid, documents in author_texts.items():
        node_text[aid] = " [SEP] ".join(documents)
    node_text.update(topic_labels)

    node_ids = sorted(node_type)
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    rows = np.fromiter((index[s] for s, _ in sorted(edges)), dtype=np.int64)
    cols = np.fromiter((index[t] for _, t in sorted(edges)), dtype=np.int64)
    values = np.ones(len(rows), dtype=np.float32)
    adjacency = sparse.csr_matrix((values, (rows, cols)), shape=(len(node_ids), len(node_ids)))
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return HeterogeneousBibliographicNetwork(
        node_ids=node_ids,
        node_types=[node_type[x] for x in node_ids],
        texts=[node_text.get(x, x) for x in node_ids],
        adjacency=adjacency,
    )


def query_adjacency(
    paper: PaperRecord, node_to_index: dict[str, int], *, include_citations: bool = False
) -> sparse.csr_matrix:
    """Create a held-out query row over the training HBN.

    Citation targets are excluded by default because they are the evaluation
    ground truth. Author/topic links remain observable query metadata.
    """
    indices: set[int] = set()
    for author in paper.authors:
        idx = node_to_index.get(author_node(author.id))
        if idx is not None:
            indices.add(idx)
    for topic in [*paper.fields, *paper.concepts]:
        idx = node_to_index.get(topic_node(topic.id))
        if idx is not None:
            indices.add(idx)
    for keyword in paper.keywords:
        idx = node_to_index.get(topic_node(keyword.casefold()))
        if idx is not None:
            indices.add(idx)
    if include_citations:
        for cited in paper.references:
            idx = node_to_index.get(paper_node(cited))
            if idx is not None:
                indices.add(idx)
    cols = np.array(sorted(indices), dtype=np.int64)
    rows = np.zeros(len(cols), dtype=np.int64)
    data = np.ones(len(cols), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(1, len(node_to_index)))
