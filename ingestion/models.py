# ingestion/models.py
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CodeChunk:
    id: str
    repo_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
