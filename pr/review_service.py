# pr/review_service.py
from __future__ import annotations

import json
from typing import List, Tuple, Dict, Any

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL
from pr.models import PRInfo, DiffChunk, ReviewComment
from retrieval.retriever import retrieve_chunks


def _get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


"""
Build a prompt that asks the model to do a serious code review:
architecture, security, bug risk, performance, readability.
"""
def _build_pr_review_prompt(
    pr: PRInfo,
    diff_chunks: List[DiffChunk],
    context_snippets: Dict[str, List[str]],
) -> str:

    # Build a concise diff section
    diff_section_parts = []
    for chunk in diff_chunks:
        diff_section_parts.append(
            f"File: {chunk.file_path} (status: {chunk.status}, lines {chunk.new_start}-{chunk.new_end})\n"
            f"{chunk.patch_text[:4000]}"  # truncate per chunk to avoid insane context
        )
    diff_section = "\n\n".join(diff_section_parts)

    # Build context section from retrieved code blocks
    ctx_parts = []
    for file_path, snippets in context_snippets.items():
        for i, snip in enumerate(snippets, start=1):
            ctx_parts.append(
                f"[{file_path} context {i}]\n{snip[:2000]}"
            )
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


"""
Run an LLM-powered review over the PR diffs + context.
Returns (summary_text, comments).
"""
def run_pr_review(
    repo_id: str,
    owner: str,
    name: str,
    pr: PRInfo,
    diff_chunks: List[DiffChunk],
) -> Tuple[str, List[ReviewComment]]:

    if not diff_chunks:
        return ("No textual diffs found in this PR.", [])

    # Very simple context strategy:
    # For each diff file, retrieve a few relevant code chunks from the existing repo index.
    context_snippets: Dict[str, List[str]] = {}
    for chunk in diff_chunks:
        # Use filename as a query; could be improved with semantic queries
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

    client = _get_client()
    prompt = _build_pr_review_prompt(pr, diff_chunks, context_snippets)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a highly experienced senior engineer performing a code review."},
            {"role": "user", "content": prompt},
        ],
        # no temperature, since some models only allow default
    )

    raw = response.choices[0].message.content or ""
    # Try to parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON between braces if the model wrapped it in text
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
                    pr_number=pr.number,
                    file_path=c.get("file_path", ""),
                    line=int(c.get("line", 1)),
                    severity=str(c.get("severity", "info")).lower(),
                    category=str(c.get("category", "readability")).lower(),
                    body=c.get("body", ""),
                    rationale=c.get("rationale", ""),
                    suggestion=c.get("suggestion"),
                    extra={k: v for k, v in c.items() if k not in {"file_path", "line", "severity", "category", "body", "rationale", "suggestion"}},
                )
            )
        except Exception:
            continue

    return (summary_text, comments)
