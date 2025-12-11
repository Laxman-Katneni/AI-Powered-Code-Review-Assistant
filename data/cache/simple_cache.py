# cache/simple_cache.py
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional

from config import DATA_DIR

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key_to_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def set_cache(key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
    path = _key_to_path(key)
    payload = {
        "created_at": time.time(),
        "ttl": ttl_seconds,
        "value": value,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def get_cache(key: str) -> Optional[Any]:
    path = _key_to_path(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    ttl = payload.get("ttl")
    created_at = payload.get("created_at", 0)
    if ttl is not None and time.time() > created_at + ttl:
        # expired
        try:
            path.unlink()
        except OSError:
            pass
        return None

    return payload.get("value")
