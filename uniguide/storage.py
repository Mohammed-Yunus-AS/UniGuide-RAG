from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import KNOWLEDGE_BASE_PATH


def load_chunks(path: Path = KNOWLEDGE_BASE_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = payload.get("chunks", [])
        return chunks if isinstance(chunks, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_chunks(chunks: list[dict[str, Any]], path: Path = KNOWLEDGE_BASE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "chunks": chunks}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def remove_sources(chunks: list[dict], source_names: set[str]) -> list[dict]:
    return [chunk for chunk in chunks if chunk.get("source") not in source_names]
