# Chunking Logic for RAG

# Currently going to use line-based chunking for simplicity and deterministic behaviour
# Layer in AST-based Heirarchy chunking

import uuid
from pathlib import Path
from typing import List

from ingestion.models import CodeChunk

"""
Very naive language guess based on extension
TODO: Need to swap this out later with something smarter
"""
def guess_language(path:Path)->str:

    ext = path.suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rb": "ruby",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".php": "php",
        ".scala": "scala",
    }
    return mapping.get(ext, "text")


"""
Split a file into overlapping line-based chunks
Simple but robust and easy to reason about
TODO: Need to change later to AST
"""
def chunk_file(
        repo_id: str,
        file_path: Path,
        max_lines_per_chunk: int = 250,
        overlap_lines: int = 25,
) -> List[CodeChunk]:
    text = file_path.read_text(encoding='utf-8', errors='ignore')
    lines = text.splitlines()
    language = guess_language(file_path)

    chunks: List[CodeChunk] = []
    start = 0

    while start < len(lines):
        end = min(start + max_lines_per_chunk, len(lines))
        content = '\n'.join(lines[start:end])
        if content.strip():
            chunk_id = str(uuid.uuid4())
            chunks.append(
                CodeChunk(
                    id = chunk_id,
                    repo_id = repo_id,
                    file_path= str(file_path),
                    language = language,
                    start_line = start + 1, # 1-based line number
                    end_line = end,
                    content = content,
                    metadata = {}
                )
            )
        if end == len(lines):
            break
        start = end - overlap_lines

    return chunks

# Chunk all files in a repo into CodeChunks
def chunk_repository(
        repo_id: str,
        file_paths:List[Path],
        max_lines_per_chunk: int = 250,
        overlap_lines: int = 25
) -> List[CodeChunk]:
    
    all_chunks: List[CodeChunk] = []
    for path in file_paths:
        all_chunks.extend(
            chunk_file(
                repo_id=repo_id,
                file_path=path,
                max_lines_per_chunk=max_lines_per_chunk,
                overlap_lines=overlap_lines,
            )
        )
    return all_chunks