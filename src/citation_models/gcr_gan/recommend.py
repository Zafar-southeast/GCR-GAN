from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse

from citation_models.common.ranking import stable_top_k_indices
from citation_models.common.records import PaperRecord

from .features import GCRFeatures
from .hbn import author_node, paper_node, topic_node


def encode_training_nodes(
    model, features: GCRFeatures, device, batch_size: int = 256
) -> np.ndarray:
    import torch

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(features.node_ids), batch_size):
            indices = np.arange(start, min(start + batch_size, len(features.node_ids)))
            batch = torch.from_numpy(features.dense_rows(indices)).to(device)
            embeddings.append(model.discriminator.encode(batch, corrupt=False).cpu().numpy())
    matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Non-finite GCR-GAN node embeddings")
    return matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def _mean_known_nodes(
    node_names: list[str], node_to_index: dict[str, int], embeddings: np.ndarray
) -> np.ndarray:
    indices = [node_to_index[x] for x in node_names if x in node_to_index]
    if not indices:
        return np.zeros(embeddings.shape[1], dtype=np.float32)
    return embeddings[indices].mean(axis=0)


def _cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    if np.linalg.norm(vector) <= 1e-12:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return _normalize(matrix) @ _normalize(vector[None, :])[0]


def combine_gcr_score_terms(
    candidate_papers: np.ndarray,
    candidate_authors: np.ndarray,
    candidate_topics: np.ndarray,
    query_content: np.ndarray,
    query_author: np.ndarray,
    query_topic: np.ndarray,
    *,
    personalized: bool,
    score_variant: str,
) -> np.ndarray:
    """Pure NumPy implementation of the Equation 8 score terms."""
    if score_variant not in {"semantic", "literal_equation_8"}:
        raise ValueError("score_variant must be 'semantic' or 'literal_equation_8'")
    candidates = (candidate_papers, candidate_authors, candidate_topics)
    queries = (query_content, query_author, query_topic)
    if any(matrix.ndim != 2 for matrix in candidates) or any(
        vector.ndim != 1 for vector in queries
    ):
        raise ValueError("Candidate score inputs must be 2D and query inputs must be 1D")
    expected_shape = candidate_papers.shape
    if any(matrix.shape != expected_shape for matrix in candidates) or any(
        vector.shape[0] != expected_shape[1] for vector in queries
    ):
        raise ValueError("All Equation 8 score terms must share candidate/embedding dimensions")
    if not all(np.isfinite(array).all() for array in (*candidates, *queries)):
        raise ValueError("Equation 8 score inputs contain NaN or infinity")
    scores = _cosine_scores(candidate_papers, query_content)
    if personalized:
        scores += _cosine_scores(candidate_authors, query_author)
        topic_candidates = (
            candidate_topics if score_variant == "semantic" else candidate_authors
        )
        scores += _cosine_scores(topic_candidates, query_topic)
    return scores


def rank_gcr_queries(
    model,
    features: GCRFeatures,
    training_embeddings: np.ndarray,
    query_records: Sequence[PaperRecord],
    query_content_embeddings: np.ndarray,
    candidate_records: Sequence[PaperRecord],
    device,
    *,
    personalized: bool = True,
    score_variant: str = "semantic",
    top_k: int | None = None,
    query_batch_size: int = 32,
) -> dict[str, list[str]]:
    """Rank with the cosine terms described around GCR-GAN Equation 8.

    ``semantic`` follows the surrounding prose: paper/content + author/author
    + topic/topic. ``literal_equation_8`` reproduces the printed symbols,
    whose third term repeats the candidate-author matrix against query topics.
    The paper does not resolve this inconsistency, so both are explicit.
    """
    import torch

    if score_variant not in {"semantic", "literal_equation_8"}:
        raise ValueError("score_variant must be 'semantic' or 'literal_equation_8'")
    if query_batch_size <= 0 or (top_k is not None and top_k <= 0):
        raise ValueError("query_batch_size and top_k must be positive")
    if training_embeddings.ndim != 2 or training_embeddings.shape[0] != len(
        features.node_ids
    ):
        raise ValueError("Training embeddings must be a 2D matrix aligned with HBN nodes")
    if query_content_embeddings.shape != (len(query_records), features.content.shape[1]):
        raise ValueError("Query content embeddings are not aligned with queries/features")
    if not np.isfinite(training_embeddings).all() or not np.isfinite(
        query_content_embeddings
    ).all():
        raise ValueError("Recommendation embeddings contain NaN or infinity")

    node_to_index = {node_id: i for i, node_id in enumerate(features.node_ids)}
    candidates = [paper for paper in candidate_records if paper_node(paper.id) in node_to_index]
    if not candidates:
        raise ValueError("No candidate papers are present in the trained HBN")
    candidate_ids = [paper.id for paper in candidates]
    candidate_indices = np.array([node_to_index[paper_node(pid)] for pid in candidate_ids])
    candidate_paper_vectors = training_embeddings[candidate_indices]
    candidate_author_vectors = np.stack(
        [
            _mean_known_nodes(
                [author_node(author.id) for author in paper.authors],
                node_to_index,
                training_embeddings,
            )
            for paper in candidates
        ]
    )
    candidate_topic_vectors = np.stack(
        [
            _mean_known_nodes(
                [topic_node(x.id) for x in [*paper.fields, *paper.concepts]]
                + [topic_node(x.casefold()) for x in paper.keywords],
                node_to_index,
                training_embeddings,
            )
            for paper in candidates
        ]
    )
    candidate_paper_normalized = _normalize(candidate_paper_vectors)
    candidate_author_normalized = _normalize(candidate_author_vectors)
    candidate_topic_normalized = _normalize(candidate_topic_vectors)
    requested_depth = len(candidate_ids) if top_k is None else min(top_k, len(candidate_ids))
    rankings: dict[str, list[str]] = {}
    # N_Qc is the query-content representation in Equation 8. Query citations
    # are ground truth and author/topic preferences are separate score terms,
    # so the content projection has an all-zero HBN adjacency row.
    with torch.no_grad():
        for start in range(0, len(query_records), query_batch_size):
            stop = min(start + query_batch_size, len(query_records))
            papers = query_records[start:stop]
            content = query_content_embeddings[start:stop]
            content_only_adjacency = sparse.csr_matrix(
                (len(papers), len(node_to_index)), dtype=np.float32
            )
            rows = features.query_row(content, content_only_adjacency)
            query_vectors = (
                model.discriminator.encode(torch.from_numpy(rows).to(device), corrupt=False)
                .cpu()
                .numpy()
            )
            scores = _normalize(query_vectors) @ candidate_paper_normalized.T
            if personalized:
                query_authors = np.stack(
                    [
                        _mean_known_nodes(
                            [author_node(author.id) for author in paper.authors],
                            node_to_index,
                            training_embeddings,
                        )
                        for paper in papers
                    ]
                )
                query_topics = np.stack(
                    [
                        _mean_known_nodes(
                            [topic_node(x.id) for x in [*paper.fields, *paper.concepts]]
                            + [topic_node(x.casefold()) for x in paper.keywords],
                            node_to_index,
                            training_embeddings,
                        )
                        for paper in papers
                    ]
                )
                scores += _normalize(query_authors) @ candidate_author_normalized.T
                topic_candidates = (
                    candidate_topic_normalized
                    if score_variant == "semantic"
                    else candidate_author_normalized
                )
                scores += _normalize(query_topics) @ topic_candidates.T
            if not np.isfinite(scores).all():
                raise FloatingPointError("Non-finite GCR-GAN ranking scores")
            for paper, paper_scores in zip(papers, scores):
                order = stable_top_k_indices(paper_scores, requested_depth)
                rankings[paper.id] = [candidate_ids[i] for i in order]
    return rankings
