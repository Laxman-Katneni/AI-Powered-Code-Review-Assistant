# pr/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# pr metadata
@dataclass
class PRInfo:
    repo_id: str          # e.g. "owner/name"
    number: int
    title: str
    author: str
    html_url: str
    base_branch: str
    head_branch: str
    body: Optional[str] = None

# code changes
@dataclass
class DiffChunk:
    id: str
    repo_id: str
    pr_number: int
    file_path: str
    status: str           # "added" | "modified" | "removed" | "renamed" | ...
    hunk_header: str      # e.g. "@@ -10,5 +12,8 @@"
    new_start: int        # starting line in new file
    new_end: int          # end line in new file (approx)
    patch_text: str       # the diff hunk text


@dataclass
class ReviewComment:
    pr_number: int
    file_path: str
    line: int
    severity: str         # "info" | "warning" | "critical"
    category: str         # "architecture" | "security" | "bug-risk" | "performance" | "readability" | ...
    body: str             # the actual comment text
    rationale: str        # why this matters
    suggestion: Optional[str] = None
    extra: Dict[str, Any] = None
