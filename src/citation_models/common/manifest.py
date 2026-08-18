from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "scipy", "torch", "transformers", "datasets", "rdflib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def write_run_manifest(
    path: str | Path,
    *,
    model: str,
    dataset_path: str | Path,
    config: dict[str, Any],
    train_ids: Iterable[str],
    test_ids: Iterable[str],
) -> dict[str, Any]:
    """Persist data, split, config, and environment evidence for a run."""
    dataset = Path(dataset_path)
    train_ids = tuple(train_ids)
    test_ids = tuple(test_ids)
    manifest = {
        "model": model,
        "dataset": {
            "configured_path": str(dataset_path),
            "size_bytes": dataset.stat().st_size,
            "sha256": file_sha256(dataset),
        },
        "split": {
            "train_count": len(train_ids),
            "test_count": len(test_ids),
            "train_ids_sha256": ids_sha256(train_ids),
            "test_ids_sha256": ids_sha256(test_ids),
        },
        "config": config,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _versions(),
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
