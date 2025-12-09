# Simple Json per repo
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from config import DATA_DIR
from metrics.models import ReviewRun
from pr.models import ReviewComment


from collections import Counter
import uuid


def _metrics_dir_for_repo(repo_id: str) -> Path:
    safe_repo = repo_id.replace("/", "__")
    d = DATA_DIR / "metrics" / safe_repo
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reviews_path(repo_id: str) -> Path:
    return _metrics_dir_for_repo(repo_id) / "reviews.jsonl"



# Append a review run record to the repo's JSONL file

def save_review_run(
    repo_id: str,
    pr_number: int,
    summary: str,
    comments: List[ReviewComment],
) -> None:


    now = datetime.now(timezone.utc).isoformat()
    by_severity = Counter(c.severity for c in comments)
    by_category = Counter(c.category for c in comments)

    run = ReviewRun(
        id=str(uuid.uuid4()),
        repo_id=repo_id,
        pr_number=pr_number,
        created_at=now,
        summary=summary,
        comment_count=len(comments),
        stats={
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
        },
    )

    path = _reviews_path(repo_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run.__dict__) + "\n")


def load_review_runs(repo_id: str) -> List[ReviewRun]:
    path = _reviews_path(repo_id)
    if not path.exists():
        return []

    runs: List[ReviewRun] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data: Dict[str, Any] = json.loads(line)
                runs.append(
                    ReviewRun(
                        id=data["id"],
                        repo_id=data["repo_id"],
                        pr_number=data["pr_number"],
                        created_at=data["created_at"],
                        summary=data["summary"],
                        comment_count=data["comment_count"],
                        stats=data.get("stats", {}),
                    )
                )
            except Exception:
                continue
    return runs
