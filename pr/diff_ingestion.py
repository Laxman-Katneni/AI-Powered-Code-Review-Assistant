
from __future__ import annotations

import uuid
from typing import List, Dict, Any, Tuple

from pr.models import DiffChunk

"""
Very simple unified diff header parser
Looks for first line starting with '@@' and parses the +c,d section as new_start/new_len.
Returns (header_line, new_start, new_end).
If parsing fails, returns ('', 1, 1).
"""
def _parse_unified_header(patch: str) -> Tuple[str, int, int]:

    if not patch:
        return ("", 1, 1)

    for line in patch.splitlines():
        line = line.strip()
        if line.startswith("@@"):
            # Example: @@ -10,5 +12,8 @@
            try:
                parts = line.split(" ")
                plus_part = [p for p in parts if p.startswith("+")][0]
                plus_part = plus_part.lstrip("+")
                if "," in plus_part:
                    start_str, length_str = plus_part.split(",", 1)
                    new_start = int(start_str)
                    new_len = int(length_str)
                else:
                    new_start = int(plus_part)
                    new_len = 1
                new_end = new_start + new_len - 1
                return (line, new_start, new_end)
            except Exception:
                return (line, 1, 1)
    return ("", 1, 1)


"""
Convert GitHub PR files (API response) into DiffChunk objects.
V1: one DiffChunk per file, using the entire patch.
"""
def build_diff_chunks_from_github_files(
    repo_id: str,
    pr_number: int,
    files_json: List[Dict[str, Any]],
) -> List[DiffChunk]:

    chunks: List[DiffChunk] = []

    for f in files_json:
        patch = f.get("patch")
        if not patch:
            # No textual diff (binary, large file, etc.)
            continue

        header, new_start, new_end = _parse_unified_header(patch)
        chunks.append(
            DiffChunk(
                id=str(uuid.uuid4()),
                repo_id=repo_id,
                pr_number=pr_number,
                file_path=f["filename"],
                status=f.get("status", "modified"),
                hunk_header=header,
                new_start=new_start,
                new_end=new_end,
                patch_text=patch,
            )
        )

    return chunks
