from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class NamedEntity:
    id: str
    name: str
    affiliation: str = ""

    @classmethod
    def from_raw(cls, value: Any, prefix: str) -> NamedEntity:
        if isinstance(value, str):
            name = _clean(value)
            return cls(id=f"{prefix}:{name.casefold()}", name=name)
        if not isinstance(value, dict):
            raise TypeError(f"Expected string or mapping for {prefix}, got {type(value)!r}")
        name = _clean(value.get("name") or value.get("raw") or value.get("label"))
        raw_id = value.get("id") or value.get("_id") or value.get("uri")
        entity_id = _clean(raw_id) or f"{prefix}:{name.casefold()}"
        affiliation = _clean(value.get("affiliation") or value.get("org"))
        return cls(id=entity_id, name=name or entity_id, affiliation=affiliation)


@dataclass
class PaperRecord:
    id: str
    title: str
    abstract: str = ""
    year: int | None = None
    citation_count: int | None = None
    authors: list[NamedEntity] = field(default_factory=list)
    venue: NamedEntity | None = None
    references: list[str] = field(default_factory=list)
    fields: list[NamedEntity] = field(default_factory=list)
    concepts: list[NamedEntity] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    facets: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PaperRecord:
        paper_id = _clean(raw.get("id") or raw.get("_id") or raw.get("index"))
        if not paper_id:
            raise ValueError("Paper record has no id/_id/index")

        authors_raw = raw.get("authors", [])
        if isinstance(authors_raw, (str, dict)):
            authors_raw = [authors_raw]
        authors = [NamedEntity.from_raw(x, "author") for x in authors_raw if x]
        venue_raw = raw.get("venue")
        venue = NamedEntity.from_raw(venue_raw, "venue") if venue_raw else None

        fos = raw.get("fields") or raw.get("fos") or raw.get("fields_of_study") or []
        if isinstance(fos, (str, dict)):
            fos = [fos]
        fields = [NamedEntity.from_raw(x, "field") for x in fos if x]
        concepts_raw = raw.get("concepts") or []
        if isinstance(concepts_raw, (str, dict)):
            concepts_raw = [concepts_raw]
        concepts = [NamedEntity.from_raw(x, "concept") for x in concepts_raw if x]
        refs = raw.get("references") or raw.get("refs") or raw.get("citations") or []
        if isinstance(refs, (str, dict)):
            refs = [refs]
        keywords = raw.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [x.strip() for x in keywords.split(",") if x.strip()]
        facets = raw.get("facets") or {}

        year_raw = raw.get("year")
        try:
            year = int(year_raw) if year_raw not in (None, "") else None
        except (TypeError, ValueError):
            year = None
        citation_count_raw = raw.get("citation_count", raw.get("n_citation"))
        try:
            citation_count = (
                int(citation_count_raw) if citation_count_raw not in (None, "") else None
            )
        except (TypeError, ValueError):
            citation_count = None

        def reference_id(value: Any) -> str:
            if isinstance(value, dict):
                return _clean(value.get("id") or value.get("_id") or value.get("paper_id"))
            return _clean(value)

        return cls(
            id=paper_id,
            title=_clean(raw.get("title")),
            abstract=_clean(raw.get("abstract")),
            year=year,
            citation_count=citation_count,
            authors=authors,
            venue=venue,
            references=sorted({reference_id(x) for x in refs if reference_id(x)}),
            fields=fields,
            concepts=concepts,
            keywords=[_clean(x) for x in keywords if _clean(x)],
            facets={
                str(k): [_clean(x) for x in (v if isinstance(v, list) else [v]) if _clean(x)]
                for k, v in facets.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def specter_text(self) -> str:
        return f"{self.title} [SEP] {self.abstract}".strip()

    @property
    def cold_start(self) -> bool:
        return not self.authors


def index_records(records: Iterable[PaperRecord]) -> dict[str, PaperRecord]:
    result: dict[str, PaperRecord] = {}
    for record in records:
        if record.id in result:
            raise ValueError(f"Duplicate paper id: {record.id}")
        result[record.id] = record
    return result
