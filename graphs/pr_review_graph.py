# graphs/pr_review_graph.py
from __future__ import annotations

import json
from typing import List, Dict, Any, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, CHAT_MODEL
from pr.models import PRInfo, DiffChunk, ReviewComment
from retrieval.retriever import retrieve_chunks


class PRReviewState(TypedDict):
    repo_id: str
    owner: str
    name: str
    pr: PRInfo
    diff_chunks: List[DiffChunk]

    # filled by nodes:
    context_snippets: Dict[str, List[str]]
    raw_model_output: str | None
    summary_text: str | None
    comments: List[ReviewComment]
    error: str | None


def _build_pr_review_prompt(
    pr: PRInfo,
    diff_chunks: List[DiffChunk],
    context_snippets: Dict[str, List[str]],
) -> str:
    # Build a concise diff section
    diff_parts = []
    for chunk in diff_chunks:
        diff_parts.append(
            f"File: {chunk.file_path} (status: {chunk.status}, lines {chunk.new_start}-{chunk.new_end})\n"
            f"{chunk.patch_text[:4000]}"  # truncate to avoid huge prompts
        )
    diff_section = "\n\n".join(diff_parts)

    # Build context section from retrieved code blocks
    ctx_parts = []
    for file_path, snippets in context_snippets.items():
        for i, snip in enumerate(snippets, start=1):
            ctx_parts.append(f"[{file_path} context {i}]\n{snip[:2000]}")
    context_section = "\n\n".join(ctx_parts) if ctx_parts else "No additional context."

    instructions = """
You are acting as a senior staff engineer performing a deep code review on a pull request.

Focus on:
- Architecture and design problems (layering, duplication, boundaries).
- Security issues (injection, auth, secrets, unsafe deserialization, insecure configs).
- Bug risks (edge cases, null handling, wrong assumptions).
- Performance concerns (obvious inefficiencies, n+1 queries, O(n^2) in hot paths).
- Readability and maintainability (only when high-impact).

Do NOT nitpick minor style issues or formatting unless they hide a bug or create confusion.

You must return STRICT JSON with this shape:

{
  "summary": [
    "short bullet point summary item 1",
    "short bullet point summary item 2"
  ],
  "comments": [
    {
      "file_path": "path/in/repo.py",
      "line": 123,
      "severity": "info | warning | critical",
      "category": "architecture | security | bug-risk | performance | readability | testing",
      "body": "Human-friendly review comment text.",
      "rationale": "Explain why this matters.",
      "suggestion": "Concrete suggestion for improvement (optional)."
    }
  ]
}

If you are not sure about the exact line, pick the closest approximate line in the new code that the comment refers to.
If there are no meaningful issues, return an empty comments array but still fill 'summary' appropriately.
"""

    prompt = f"""{instructions}

Pull Request:
- Repo: {pr.repo_id}
- PR #{pr.number}: {pr.title}
- Author: {pr.author}
- Base branch: {pr.base_branch}
- Head branch: {pr.head_branch}
- Description:
{pr.body or "(no description provided)"}

=== DIFFS (NEW CODE) ===
{diff_section}

=== RELEVANT EXISTING CODE CONTEXT ===
{context_section}

Now produce the JSON described above.
"""
    return prompt


def _parse_review_output(raw: str, pr_number: int) -> tuple[str, List[ReviewComment]]:
    """
    Parse the model JSON into (summary_text, comments).
    Falls back gracefully if JSON is malformed.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON between braces if it's wrapped in text
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except Exception:
            # Fall back: no structured comments
            return (f"Model did not return valid JSON. Raw output:\n\n{raw}", [])

    summary_items = data.get("summary") or []
    if isinstance(summary_items, list):
        summary_text = "- " + "\n- ".join(str(s) for s in summary_items)
    else:
        summary_text = str(summary_items)

    comments_json = data.get("comments") or []
    comments: List[ReviewComment] = []

    for c in comments_json:
        try:
            comments.append(
                ReviewComment(
                    pr_number=pr_number,
                    file_path=c.get("file_path", ""),
                    line=int(c.get("line", 1)),
                    severity=str(c.get("severity", "info")).lower(),
                    category=str(c.get("category", "readability")).lower(),
                    body=c.get("body", ""),
                    rationale=c.get("rationale", ""),
                    suggestion=c.get("suggestion"),
                    extra={k: v for k, v in c.items() if k not in {
                        "file_path",
                        "line",
                        "severity",
                        "category",
                        "body",
                        "rationale",
                        "suggestion",
                    }},
                )
            )
        except Exception:
            continue

    return summary_text, comments


# ---------- Graph Nodes ----------
# Node1
def gather_context_node(state: PRReviewState) -> PRReviewState:
    """
    For each touched file, retrieve a few relevant chunks from the existing repo index.
    """
    repo_id = state["repo_id"]
    diff_chunks = state["diff_chunks"]

    context_snippets: Dict[str, List[str]] = {}
    for chunk in diff_chunks:
        q = f"Context for changes in file {chunk.file_path}"
        try:
            retrieved = retrieve_chunks(repo_id, q, k=2)
        except Exception:
            retrieved = []

        snippets: List[str] = []
        for r in retrieved:
            snippets.append(r["content"])
        if snippets:
            context_snippets.setdefault(chunk.file_path, []).extend(snippets)

    state["context_snippets"] = context_snippets
    return state

# Node 2
def call_llm_node(state: PRReviewState) -> PRReviewState:
    """
    Call the chat model with instructions + diff + context.
    """
    pr = state["pr"]
    diff_chunks = state["diff_chunks"]
    ctx = state.get("context_snippets", {})

    prompt = _build_pr_review_prompt(pr, diff_chunks, ctx)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,  # deterministic reviews
    )

    messages = [
        SystemMessage(content="You are a highly experienced senior engineer performing a code review."),
        HumanMessage(content=prompt),
    ]

    try:
        resp = llm.invoke(messages)
        state["raw_model_output"] = resp.content or ""
        state["error"] = None
    except Exception as e:
        state["raw_model_output"] = ""
        state["error"] = f"LLM call failed: {e}"

    return state

# Node 3
def parse_output_node(state: PRReviewState) -> PRReviewState:
    """
    Parse the JSON output into structured ReviewComment objects.
    """
    if state.get("error"):
        return state

    raw = state.get("raw_model_output") or ""
    summary_text, comments = _parse_review_output(raw, pr_number=state["pr"].number)
    state["summary_text"] = summary_text
    state["comments"] = comments
    return state


# ---------- Graph Builder ----------

def build_pr_review_graph():
    g = StateGraph(PRReviewState)

    g.add_node("gather_context", gather_context_node)
    g.add_node("call_llm", call_llm_node)
    g.add_node("parse_output", parse_output_node)

    g.set_entry_point("gather_context")
    g.add_edge("gather_context", "call_llm")
    g.add_edge("call_llm", "parse_output")
    g.add_edge("parse_output", END)

    return g.compile()
