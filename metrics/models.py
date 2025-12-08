# metrics/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ReviewRun:
    id: str
    repo_id: str
    pr_number: int
    created_at: str           # ISO timestamp
    summary: str
    comment_count: int
    stats: Dict[str, Any]     # e.g. {"by_severity": {...}, "by_category": {...}}
