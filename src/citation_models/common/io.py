from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .records import PaperRecord


def _iter_json_array(stream, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Incrementally parse a JSON array without loading the archive twice."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = ended = False
    while True:
        chunk = stream.read(chunk_size)
        eof = not chunk
        buffer += chunk
        cursor = 0
        while True:
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1
            if not started:
                if cursor >= len(buffer):
                    break
                if buffer[cursor] != "[":
                    raise ValueError("Expected a JSON array")
                started = True
                cursor += 1
                continue
            while cursor < len(buffer) and (
                buffer[cursor].isspace() or buffer[cursor] == ","
            ):
                cursor += 1
            if cursor >= len(buffer):
                break
            if buffer[cursor] == "]":
                ended = True
                cursor += 1
                break
            try:
                value, end = decoder.raw_decode(buffer, cursor)
            except json.JSONDecodeError:
                break
            if not isinstance(value, dict):
                raise TypeError("Every item in the paper JSON array must be an object")
            yield value
            cursor = end
        buffer = buffer[cursor:]
        if ended:
            if buffer.strip():
                raise ValueError("Unexpected content after the JSON array")
            return
        if eof:
            raise ValueError("Incomplete or malformed JSON array")


def _iter_json_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first = ""
        while not first:
            first = stream.readline()
            if not first:
                return
            first = first.strip()
        if first.startswith("["):
            stream.seek(0)
            yield from _iter_json_array(stream)
            return
        yield json.loads(first.rstrip(","))
        for line in stream:
            line = line.strip().rstrip(",")
            if line and line not in ("]", "["):
                yield json.loads(line)


def _iter_legacy_aminer(path: Path) -> Iterator[dict]:
    current: dict = {}

    def finish() -> dict | None:
        nonlocal current
        if not current:
            return None
        record = current
        current = {}
        return record

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if not line:
                record = finish()
                if record:
                    yield record
            elif line.startswith("#index"):
                current["id"] = line[6:].strip()
            elif line.startswith("#*"):
                current["title"] = line[2:].strip()
            elif line.startswith("#@"):
                current["authors"] = [x.strip() for x in line[2:].split(",") if x.strip()]
            elif line.startswith("#t"):
                current["year"] = line[2:].strip()
            elif line.startswith("#c"):
                current["venue"] = line[2:].strip()
            elif line.startswith("#%"):
                current.setdefault("references", []).append(line[2:].strip())
            elif line.startswith("#!"):
                current["abstract"] = line[2:].strip()
        record = finish()
        if record:
            yield record


def iter_papers(path: str | Path, fmt: str = "auto") -> Iterator[PaperRecord]:
    path = Path(path)
    if fmt == "auto":
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            prefix = stream.read(32).lstrip()
        fmt = "json" if prefix.startswith(("{", "[")) else "aminer"
    iterator = _iter_json_records(path) if fmt == "json" else _iter_legacy_aminer(path)
    for raw in iterator:
        yield PaperRecord.from_dict(raw)


def load_papers(path: str | Path, fmt: str = "auto") -> list[PaperRecord]:
    return list(iter_papers(path, fmt))


def save_papers(records: Iterable[PaperRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_id_list(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip()]


def save_id_list(ids: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in ids), encoding="utf-8")
