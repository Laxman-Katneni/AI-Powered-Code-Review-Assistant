# indexing/index_metadata.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from config import INDEXES_DIR


def _safe_repo_id(repo_id: str) -> str:
    # Same convention as your vector_store
    return repo_id.replace("/", "__")


def _meta_path(repo_id: str) -> Path:
    return INDEXES_DIR / _safe_repo_id(repo_id) / "meta.json"


#Persist simple metadata about an index run
def save_index_metadata(
    repo_id: str,
    *,
    file_count: int,
    chunk_count: int,
    commit_hash: Optional[str] = None,
) -> None:

    path = _meta_path(repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "repo_id": repo_id,
        "file_count": file_count,
        "chunk_count": chunk_count,
        "commit_hash": commit_hash,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Return metadata dict if present, otherwise None
def load_index_metadata(repo_id: str) -> Optional[Dict[str, Any]]:
    path = _meta_path(repo_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
