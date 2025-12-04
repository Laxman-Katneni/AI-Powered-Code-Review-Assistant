from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import INDEXES_DIR
from ingestion.models import CodeChunk
from llm.embeddings import get_embedding_client


def _repo_index_path(repo_id:str)->Path:
    safe_id = repo_id.replace("/", "__")
    return INDEXES_DIR / safe_id

"""
Build (or rebuild) a Chroma index for a repo from CodeChunks.
This is idempotent: we can blow away and recreate per repo when needed.
"""
def build_index(repo_id: str, chunks: List[CodeChunk])->Chroma:
    persist_dir = _repo_index_path(repo_id)
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    for chunk in chunks:
        metadata = {
            "repo_id": chunk.repo_id,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "chunk_id": chunk.id,
        }
        docs.append(Document(page_content = chunk.content, metadata = metadata))
    
    embeddings = get_embedding_client()

    vectordb = Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        persist_directory = str(persist_dir)
    )
    return vectordb

"""
Load an existing Chroma index for a repo.
Raises if it doesn't exist.
"""
def load_index(repo_id: str)->Chroma:
    persist_dir = _repo_index_path(repo_id)
    if not persist_dir.exists():
        raise FileNotFoundError(f"No index found for repo_id={repo_id}")

    embeddings = get_embedding_client()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    return vectordb