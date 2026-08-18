from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .hbn import HeterogeneousBibliographicNetwork


@dataclass
class MinMaxScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> MinMaxScaler:
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("Content embeddings must be a non-empty 2D matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("Content embeddings contain NaN or infinity")
        return cls(matrix.min(axis=0), matrix.max(axis=0))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[1] != self.minimum.shape[0]:
            raise ValueError("Content embedding dimensions do not match the fitted scaler")
        if not np.isfinite(matrix).all():
            raise ValueError("Content embeddings contain NaN or infinity")
        denominator = np.maximum(self.maximum - self.minimum, 1e-12)
        # The sigmoid generator has support [0, 1]. Held-out SPECTER values can
        # lie outside the training extrema, so clip them to the same support.
        return np.clip((matrix - self.minimum) / denominator, 0.0, 1.0).astype(np.float32)


@dataclass
class GCRFeatures:
    content: np.ndarray
    adjacency: sparse.csr_matrix
    node_ids: list[str]
    node_types: list[str]
    content_scaler: MinMaxScaler

    def __post_init__(self) -> None:
        rows = len(self.node_ids)
        if self.content.ndim != 2 or self.content.shape[0] != rows:
            raise ValueError("Content rows must align with node_ids")
        if self.adjacency.shape != (rows, rows):
            raise ValueError("Adjacency must be square and aligned with node_ids")
        if len(self.node_types) != rows:
            raise ValueError("node_types must align with node_ids")
        if len(set(self.node_ids)) != rows:
            raise ValueError("node_ids contains duplicates")
        if self.content_scaler.minimum.shape != (self.content.shape[1],) or (
            self.content_scaler.maximum.shape != (self.content.shape[1],)
        ):
            raise ValueError("Saved content scaler does not match content dimensions")
        if not self.all_finite():
            raise ValueError("GCR features contain NaN or infinity")

    @property
    def input_dim(self) -> int:
        return self.content.shape[1] + self.adjacency.shape[1]

    def dense_rows(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("Feature indices must be one-dimensional")
        if indices.size == 0:
            return np.empty((0, self.input_dim), dtype=np.float32)
        if indices.min() < 0 or indices.max() >= len(self.node_ids):
            raise IndexError("Feature index is outside the HBN")
        return np.concatenate(
            [
                np.asarray(self.content[indices], dtype=np.float32),
                self.adjacency[indices].toarray().astype(np.float32, copy=False),
            ],
            axis=1,
        )

    def query_row(self, content: np.ndarray, adjacency: sparse.csr_matrix) -> np.ndarray:
        if adjacency.shape != (content.shape[0], len(self.node_ids)):
            raise ValueError("Query adjacency rows must align with query content and HBN nodes")
        scaled = self.content_scaler.transform(content)
        return np.concatenate(
            [scaled, adjacency.toarray().astype(np.float32, copy=False)], axis=1
        )

    def dense_batch_megabytes(self, batch_size: int) -> float:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return batch_size * self.input_dim * np.dtype(np.float32).itemsize / (1024**2)

    def all_finite(self) -> bool:
        return _all_finite_rows(self.content) and np.isfinite(self.adjacency.data).all()

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "content.npy", self.content)
        np.save(directory / "content_min.npy", self.content_scaler.minimum)
        np.save(directory / "content_max.npy", self.content_scaler.maximum)
        sparse.save_npz(directory / "adjacency.npz", self.adjacency)
        (directory / "nodes.json").write_text(
            json.dumps(
                [{"id": i, "type": t} for i, t in zip(self.node_ids, self.node_types)],
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, *, mmap_content: bool = True) -> GCRFeatures:
        directory = Path(directory)
        nodes = json.loads((directory / "nodes.json").read_text(encoding="utf-8"))
        return cls(
            content=np.load(directory / "content.npy", mmap_mode="r" if mmap_content else None),
            adjacency=sparse.load_npz(directory / "adjacency.npz").tocsr(),
            node_ids=[x["id"] for x in nodes],
            node_types=[x["type"] for x in nodes],
            content_scaler=MinMaxScaler(
                np.load(directory / "content_min.npy"), np.load(directory / "content_max.npy")
            ),
        )


def _all_finite_rows(matrix: np.ndarray, chunk_rows: int = 16384) -> bool:
    """Validate ordinary arrays and memory maps without a full-size boolean copy."""
    if matrix.ndim != 2:
        return False
    for start in range(0, matrix.shape[0], chunk_rows):
        if not np.isfinite(matrix[start : start + chunk_rows]).all():
            return False
    return True


def create_features(
    network: HeterogeneousBibliographicNetwork, content_embeddings: np.ndarray
) -> GCRFeatures:
    if content_embeddings.shape[0] != len(network.node_ids):
        raise ValueError("Content embedding rows do not match HBN nodes")
    scaler = MinMaxScaler.fit(content_embeddings)
    return GCRFeatures(
        content=scaler.transform(content_embeddings),
        adjacency=network.adjacency.astype(np.float32),
        node_ids=network.node_ids,
        node_types=network.node_types,
        content_scaler=scaler,
    )
