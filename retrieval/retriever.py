from typing import List, Dict, Any

from indexing.vector_store import load_index


"""
Retrieve top-k relevant chunks for a given query and repo.
Returns a list of {content, metadata} dicts.
"""
def retrieve_chunks(
        repo_id: str,
        query: str,
        k: int = 8
) -> List[Dict[str, Any]]:
    
    vectordb = load_index(repo_id)
    results = vectordb.similarity_search_with_score(query, k = k)

    formatted = []
    for doc, score in results:
        entry = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        formatted.append(entry)
    return formatted