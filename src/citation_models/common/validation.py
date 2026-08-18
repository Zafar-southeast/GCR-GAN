from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .records import PaperRecord


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, int | float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Dataset validation failed:\n- " + "\n- ".join(self.errors))


def validate_records(records: Iterable[PaperRecord]) -> ValidationReport:
    records = list(records)
    report = ValidationReport()
    ids = [r.id for r in records]
    duplicate_ids = sorted(x for x, n in Counter(ids).items() if n > 1)
    if duplicate_ids:
        report.errors.append(f"Duplicate paper ids: {duplicate_ids[:20]}")
    empty_titles = sum(not r.title for r in records)
    if empty_titles:
        report.errors.append(f"{empty_titles} paper records have empty titles")
    known = set(ids)
    dangling = sum(ref not in known for r in records for ref in r.references)
    self_citations = sum(r.id in r.references for r in records)
    if dangling:
        report.warnings.append(
            f"{dangling} references point outside the loaded corpus; they are ignored in evaluation"
        )
    if self_citations:
        report.warnings.append(f"{self_citations} self-referential paper edges will be removed")
    report.statistics = {
        "papers": len(records),
        "authors": len({a.id for r in records for a in r.authors}),
        "venues": len({r.venue.id for r in records if r.venue}),
        "fields": len({f.id for r in records for f in r.fields}),
        "concepts": len({c.id for r in records for c in r.concepts}),
        "citation_edges_in_corpus": sum(
            ref in known and ref != r.id for r in records for ref in r.references
        ),
        "cold_start_missing_authors": sum(r.cold_start for r in records),
    }
    return report


def validate_split(train_ids: Iterable[str], test_ids: Iterable[str], known_ids: set[str]) -> None:
    train, test = set(train_ids), set(test_ids)
    overlap = train & test
    if overlap:
        raise ValueError(f"Train/test leakage: {len(overlap)} paper ids overlap")
    unknown = (train | test) - known_ids
    if unknown:
        raise ValueError(f"Split contains unknown ids: {sorted(unknown)[:20]}")
    missing = known_ids - (train | test)
    if missing:
        raise ValueError(f"Split omits known ids: {sorted(missing)[:20]}")
    if not train or not test:
        raise ValueError("Both train and test splits must be non-empty")
