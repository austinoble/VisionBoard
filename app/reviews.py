"""Persistent, timestamped GT-instance reviews stored as one JSON file per image."""
from __future__ import annotations

import csv
import io
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import store

_lock = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _root(source: str) -> Path:
    src = store.get_source(source)
    if not src:
        raise ValueError(f"source {source} not found")
    configured = src.get("reviews")
    root = Path(configured) if configured else Path(src["images"]).parent / "reviews"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path(source: str, stem: str) -> Path:
    # The scene lookup establishes that stem belongs to this source; use only its basename
    # as an additional guard against path traversal.
    if Path(stem).name != stem:
        raise ValueError("invalid scene name")
    return _root(source) / f"{stem}.json"


def _empty(source: str, stem: str) -> dict:
    return {"source": source, "image": stem, "instances": {}}


def _read(source: str, stem: str) -> dict:
    path = _path(source, stem)
    if not path.is_file():
        return _empty(source, stem)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid review file: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("instances", {}), dict):
        raise ValueError(f"invalid review structure: {path}")
    data.setdefault("source", source)
    data.setdefault("image", stem)
    data.setdefault("instances", {})
    return data


def _write(source: str, stem: str, data: dict) -> None:
    path = _path(source, stem)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def list_instance(source: str, stem: str, instance_id: str) -> dict:
    with _lock:
        data = _read(source, stem)
        entries = list(data["instances"].get(instance_id, []))
    entries.sort(key=lambda item: item.get("created_at_utc", ""), reverse=True)
    return {"source": source, "image": stem, "instance_id": instance_id,
            "entries": entries}


def create(source: str, stem: str, instance_id: str, text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("review text is empty")
    now = _utc_now()
    entry = {
        "id": uuid.uuid4().hex,
        "created_at_utc": now,
        "updated_at_utc": now,
        "text": text,
    }
    with _lock:
        data = _read(source, stem)
        data["instances"].setdefault(instance_id, []).append(entry)
        _write(source, stem, data)
    return entry


def update(source: str, stem: str, instance_id: str, entry_id: str, text: str) -> dict:
    """Update an entry, or remove it when its edited text is empty."""
    text = text.strip()
    with _lock:
        data = _read(source, stem)
        entries = data["instances"].get(instance_id, [])
        index = next((i for i, item in enumerate(entries) if item.get("id") == entry_id), None)
        if index is None:
            raise KeyError(entry_id)
        if not text:
            removed = entries.pop(index)
            if not entries:
                data["instances"].pop(instance_id, None)
            _write(source, stem, data)
            return {"removed": True, "entry": removed}
        entry = entries[index]
        entry["text"] = text
        entry["updated_at_utc"] = _utc_now()
        _write(source, stem, data)
        return {"removed": False, "entry": entry}


def remove(source: str, stem: str, instance_id: str, entry_id: str) -> dict:
    return update(source, stem, instance_id, entry_id, "")


def export(source: str, fmt: str) -> tuple[bytes, str, str]:
    rows = []
    with _lock:
        for path in sorted(_root(source).glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for instance_id, entries in data.get("instances", {}).items():
                for entry in entries:
                    rows.append({
                        "source": source,
                        "image": data.get("image", path.stem),
                        "instance_id": instance_id,
                        "entry_id": entry.get("id", ""),
                        "created_at_utc": entry.get("created_at_utc", ""),
                        "updated_at_utc": entry.get("updated_at_utc", ""),
                        "text": entry.get("text", ""),
                    })
    rows.sort(key=lambda row: (row["created_at_utc"], row["image"], row["instance_id"]))
    if fmt == "json":
        body = json.dumps({"source": source, "reviews": rows}, indent=2,
                          ensure_ascii=False).encode()
        return body, "application/json", f"{source}_reviews.json"
    output = io.StringIO()
    fields = ["source", "image", "instance_id", "entry_id", "created_at_utc",
              "updated_at_utc", "text"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode(), "text/csv; charset=utf-8", f"{source}_reviews.csv"
