# For Answer Generation

from typing import List, Dict, Any

# Format retrieved chunks into a readable context block for the LLM
def build_context_block(chunks: List[Dict[str, Any]])->str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        header = f"[{i}] file: {meta['file_path']} (lines {meta['start_line']}-{meta['end_line']})"
        blocks.append(f"{header}\n{chunk['content']}")
    return "\n\n".join(blocks)

def build_rag_prompt(
    question: str,
    chunks: List[Dict[str, Any]],
) -> str:
    context_block = build_context_block(chunks)
    return f"""You are an AI assistant that helps developers understand a codebase.

You are answering questions about a specific repository. Use ONLY the context below.
If the answer is not in the context, say you are not sure instead of guessing.

When you reference code, cite it in the format [file:line_start-line_end].

Context:
{context_block}

User question:
{question}

Answer:"""