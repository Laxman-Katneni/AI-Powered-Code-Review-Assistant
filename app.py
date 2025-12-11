import streamlit as st
from pathlib import Path

from auth.github_auth import (
    generate_state,
    get_authorize_url,
    exchange_code_for_token,
    get_user,
    get_user_repos,
)
from config import validate_config
from ingestion.file_discovery import list_code_files
from ingestion.parser import chunk_repository
from indexing.vector_store import build_index, load_index
from retrieval.retriever import retrieve_chunks
from llm.chat_llm import answer_with_rag
from indexing.index_metadata import save_index_metadata, load_index_metadata
from ingestion.github_client import clone_or_update_repo, get_repo_local_path

from auth.github_pr_client import list_pull_requests, get_pull_request_files, post_pr_issue_comment
from pr.diff_ingestion import build_diff_chunks_from_github_files
# from pr.review_service import run_pr_review  <-- REMOVED (Replaced by Graph)
from metrics.store import save_review_run, load_review_runs
from pr.models import PRInfo, ReviewComment
import pandas as pd
from typing import Optional, Dict, Any, List
from math import sqrt
from llm.embeddings import embed_query
from graphs.pr_review_graph import build_pr_review_graph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graphs.chat_graph import build_chat_graph



# ------------- Helpers -------------

def ensure_index_exists(repo_id: str) -> bool:
    try:
        load_index(repo_id)
        return True
    except FileNotFoundError:
        return False


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ------------- Main App -------------

def main():
    # Validate config - Error early if any API Key is missing
    st.set_page_config(
        page_title="AI Code Review Assistant",
        layout="wide",
    )
    try:
        validate_config()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


    st.title("RepoMind - An AI Code Review Assistant")
    st.write("Welcome! This is the starting point for your codebase assistant")

    # --- Handle OAuth callback ---
    query_params = st.query_params

    # Check for auth code indicating a return from GitHub
    if "code" in query_params and "gh_access_token" not in st.session_state:
        code = query_params["code"]
        returned_state = query_params.get("state")
        expected_state = st.session_state.get("gh_state")

        # Validate state to prevent CSRF
        if expected_state and returned_state != expected_state:
            st.error("State mismatch during GitHub OAuth. Please try logging in again.")
            st.query_params.clear() # Clear the bad params
        else:
            try:
                access_token = exchange_code_for_token(code)
                user = get_user(access_token)
                
                # Save to session
                st.session_state["gh_access_token"] = access_token
                st.session_state["gh_user"] = user

                # CRITICAL FIX 1: Clear params and RERUN immediately
                st.success(f"Logged in as {user.login}")
                st.query_params.clear()
                st.rerun() 
                
            except Exception as e:
                # CRITICAL FIX 2: If auth fails (e.g. old code), clear params so we don't loop
                st.error(f"GitHub OAuth failed: {e}")
                st.query_params.clear()

    st.sidebar.header("GitHub")

    # --- If not logged in, show login button ---
    if "gh_access_token" not in st.session_state:
        st.sidebar.write("Connect your GitHub account to index your repos.")

        if "gh_state" not in st.session_state:
            st.session_state["gh_state"] = generate_state()

        auth_url = get_authorize_url(st.session_state["gh_state"])
        
        st.sidebar.markdown(
            f"""
            <a href="{auth_url}" target="_self" style="
                display: inline-block;
                width: 100%;
                padding: 0.5rem 1rem;
                background-color: #FF4B4B; 
                color: white;
                text-align: center;
                text-decoration: none;
                border-radius: 0.5rem;
                font-weight: bold;
                border: 1px solid transparent;
            ">
                Login with GitHub
            </a>
            """,
            unsafe_allow_html=True
        )

        st.info("Log in with GitHub to continue.")
        st.stop()

    # If logged in:
    gh_user = st.session_state["gh_user"]
    st.sidebar.success(f"Logged in as {gh_user.login}")

    # Logout option
    if st.sidebar.button("Logout"):
        # Safely remove keys
        for key in ["gh_access_token", "gh_user", "gh_state"]:
            st.session_state.pop(key, None)

        # Clear query params
        st.query_params.clear()
        
        # Rerun to show login screen
        st.rerun()

    # --- Repo selection ---
    access_token = st.session_state["gh_access_token"]
    repos = get_user_repos(access_token)
    repo_options = [r["full_name"] for r in repos]  # e.g. "owner/name"

    if not repo_options:
        st.warning("No repositories found for this GitHub user.")
        st.stop()

    selected_full_name = st.sidebar.selectbox("Select a repo", repo_options)
    owner, name = selected_full_name.split("/")
    repo_id = f"github::{selected_full_name}"

    # Branch for GitHub links (used in Code Q&A Sources)
    branch = st.sidebar.text_input(
        "Branch for GitHub links",
        value="main",
        key="branch_input",
        help="Used for building GitHub links in the Code Q&A sources section.",
    )

    # --- Indexing Block ---
    if st.sidebar.button("Fetch & Index selected repo"):
        try:
            with st.spinner("Cloning/updating & indexing repo..."):
                local_path, commit_hash = clone_or_update_repo(owner, name, access_token)
                files = list_code_files(local_path)
                chunks = chunk_repository(repo_id, files)

                # Normalize paths to be relative to repo root
                for ch in chunks:
                    try:
                        ch.file_path = str(Path(ch.file_path).relative_to(local_path))
                    except ValueError:
                        pass
                    ch.metadata["commit_hash"] = commit_hash

                build_index(repo_id, chunks)
                # Save index metadata
                save_index_metadata(
                    repo_id,
                    file_count=len(files),
                    chunk_count=len(chunks),
                    commit_hash=commit_hash,
                )
            st.success(f"Indexed {selected_full_name} @ {commit_hash[:7]} ✅")
        except Exception as e:
            st.error(f"Failed to fetch/index repo: {e}")

    # Ensure index exists for this repo (for Q&A + context)
    indexed = ensure_index_exists(repo_id)
    if not indexed:
        st.info("Index this repo first using the sidebar.")
        st.stop()

    # Load index metadata 
    # Show index status (Phase 4)
    meta = load_index_metadata(repo_id)
    if meta:
        commit = meta.get("commit_hash")
        commit_str = f"@ {commit[:7]}" if commit else ""
        st.caption(
            f"Repo: {selected_full_name} {commit_str} • "
            f"Files indexed: {meta.get('file_count', '?')} • "
            f"Chunks: {meta.get('chunk_count', '?')} • "
            f"Indexed at: {meta.get('indexed_at', '')}"
        )
    else:
        st.caption(f"Repo: {selected_full_name} • Index metadata unavailable")

    # TABS: Code Q&A, PR Review, Quality Dashboard
    # Initialize PR review graph once per session
    if "pr_review_graph" not in st.session_state:
        st.session_state.pr_review_graph = build_pr_review_graph()

    pr_review_graph = st.session_state.pr_review_graph
    
    tab_chat, tab_pr, tab_dashboard, tab_chat_graph = st.tabs(
    ["💬 Code Q&A", "🔍 PR Review", "📊 Quality Dashboard", "🧠 Chat (LangGraph)"]
    )


    # ------------------------
    # Tab 1: Code Q&A (your existing logic)
    # ------------------------
    with tab_chat:
        st.markdown("### Ask a question about this repo")
        # --- Session state for Q&A results ---
        if "qa_answer" not in st.session_state:
            st.session_state.qa_answer = None
        if "qa_sources" not in st.session_state:
            st.session_state.qa_sources = []
        
        # Semantic cache: list of dicts
        if "qa_semantic_cache" not in st.session_state:
            st.session_state.qa_semantic_cache = []

        question = st.text_input(
            "Question",
            placeholder="e.g. Where is user authentication implemented?",
            key="qna_question",
        )
        st.markdown(
            "_Example questions:_ "
            "`Where is the main entrypoint?`, "
            "`How does authentication work?`, "
            "`Where are database models defined?`"
        )

        if st.button("Ask", key="qna_ask") and question.strip():
            q_norm = question.strip()
            with st.spinner("Thinking..."):
                # --- Determine index_version so cache is tied to current index ---
                meta = load_index_metadata(repo_id)
                index_version = meta.get("indexed_at") if meta else "no-index-meta"

                # --- 1) Try semantic cache first ---
                cache = st.session_state.qa_semantic_cache

                # Embed current question once
                try:
                    q_emb = embed_query(q_norm)
                except Exception as e:
                    st.warning(f"Failed to embed question, falling back to normal RAG. ({e})")
                    q_emb = None

                best_entry = None
                best_sim = 0.0
                SEMANTIC_THRESHOLD = 0.90  # tweak as needed

                if q_emb is not None:
                    for entry in cache:
                        if entry["repo_id"] != repo_id:
                            continue
                        if entry["index_version"] != index_version:
                            continue
                        sim = cosine_sim(q_emb, entry["embedding"])
                        if sim > best_sim:
                            best_sim = sim
                            best_entry = entry
                if best_entry and best_sim >= SEMANTIC_THRESHOLD:
                    # Serve from semantic cache
                    st.info(f"Answer served from semantic cache (similarity {best_sim:.2f}).")
                    st.session_state.qa_answer = best_entry["answer"]
                    st.session_state.qa_sources = best_entry["sources"]
                else:
                    # --- 2) Fall back to normal RAG flow ---
                    retrieved = retrieve_chunks(repo_id, question, k=6)
                    if not retrieved:
                        st.warning("I couldn't find any relevant code snippets for that question.")
                    else:
                        # Simple guardrail: filter by distance/score if available
                        MAX_DISTANCE = 2
                        filtered: List[dict] = []
                        for r in retrieved:
                            score = r.get("score", 0.0)
                            if score < MAX_DISTANCE:
                                filtered.append(r)

                        if not filtered:
                            st.warning(
                                "I found some code, but none of it looked strongly related. "
                                "Try rephrasing your question or being more specific."
                            )
                            st.stop()

                        answer = answer_with_rag(question, filtered)
                        used_chunks = filtered

                        seen = set()
                        sources_meta = []

                        for r in used_chunks:
                            meta_r = r["metadata"]
                            key = (meta_r["file_path"], meta_r["start_line"], meta_r["end_line"])
                            if key in seen:
                                continue
                            seen.add(key)
                            sources_meta.append(meta_r)

                        st.session_state.qa_answer = answer
                        st.session_state.qa_sources = sources_meta
                        # 🔐 Add this QA pair to semantic cache
                        if q_emb is None:
                            # If embedding failed earlier, compute once now
                            try:
                                q_emb = embed_query(q_norm)
                            except Exception:
                                q_emb = None

                        if q_emb is not None:
                            st.session_state.qa_semantic_cache.append(
                                {
                                    "repo_id": repo_id,
                                    "index_version": index_version,
                                    "question": q_norm,
                                    "embedding": q_emb,
                                    "answer": answer,
                                    "sources": sources_meta,
                                }
                            )

        # ---- Render last answer + sources (SURVIVES reruns) ----
        if st.session_state.qa_answer:
            answer = st.session_state.qa_answer
            sources_meta = st.session_state.qa_sources        

            st.markdown("#### Answer")
            st.write(answer)

            # Build Sources list
            st.markdown("#### Sources")
            
            # GitHub links for sources
            github_base = f"https://github.com/{owner}/{name}/blob/{branch}"
            for meta_r in sources_meta:
                fp = meta_r["file_path"]
                start = meta_r["start_line"]
                end = meta_r["end_line"]
                url = f"{github_base}/{fp}#L{start}-L{end}"
                st.markdown(
                    f"- [{fp} (lines {start}-{end})]({url})",
                    unsafe_allow_html=False,
                )
            
            # Code viewer
            st.markdown("#### View source code")

            if sources_meta:
                options = [
                    f"{m['file_path']} (lines {m['start_line']}-{m['end_line']})"
                    for m in sources_meta
                ]
                selected_option = st.selectbox(
                    "Select a source to view",
                    options,
                    key="source_view_select",
                )

                selected_meta = sources_meta[options.index(selected_option)]

                local_repo_path = get_repo_local_path(owner, name)
                file_path = Path(local_repo_path) / selected_meta["file_path"]

                try:
                    code_text = file_path.read_text(encoding="utf-8", errors="ignore")
                except FileNotFoundError:
                    st.error(f"Could not read file: {file_path}")
                    code_text = ""

                st.code(
                    code_text,
                    language=selected_meta.get("language", "text"),
                )
            else:
                st.caption("No sources available to display.")
        else:
            st.info("Ask a question to see the answer, sources, and code viewer.")

    # ------------------------
    # Tab 2: PR Review
    # ------------------------
    with tab_pr:
        # ---- Phase 6 state ----
        if "pr_summary" not in st.session_state:
            st.session_state.pr_summary = None
        if "pr_comments" not in st.session_state:
            st.session_state.pr_comments = []
        if "pr_markdown" not in st.session_state:
            st.session_state.pr_markdown = None
        if "pr_number" not in st.session_state:
            st.session_state.pr_number = None

        st.markdown("### AI PR Review")
        
        

        # List open PRs
        try:
            prs = list_pull_requests(owner, name, access_token)
        except Exception as e:
            st.error(f"Failed to list pull requests: {e}")
            prs = []

        if not prs:
            st.info("No open pull requests found for this repo.")
        else:
            pr_labels = [f"#{pr.number} – {pr.title} (by {pr.author})" for pr in prs]
            selected_idx = st.selectbox("Select a PR to review", list(range(len(prs))), format_func=lambda i: pr_labels[i])
            selected_pr: PRInfo = prs[selected_idx]

            # Link to GitHub PR
            st.markdown(
                f"[Open PR in GitHub]({selected_pr.html_url})  \n"
                f"Base: `{selected_pr.base_branch}` → Head: `{selected_pr.head_branch}`"
            )
            # --- Run review: compute + STORE ONLY ---
            if st.button("Run AI Review", key="run_pr_review"):
                with st.spinner("Analyzing PR with LangGraph workflow..."):
                    try:
                        files_json = get_pull_request_files(owner, name, selected_pr.number, access_token)
                        diff_chunks = build_diff_chunks_from_github_files(selected_pr.repo_id, selected_pr.number, files_json)
                        
                        # --- LANGGRAPH INVOCATION START ---
                        initial_state = {
                            "repo_id": repo_id,
                            "owner": owner,
                            "name": name,
                            "pr": selected_pr,
                            "diff_chunks": diff_chunks,
                            "context_snippets": {},
                            "raw_model_output": None,
                            "summary_text": None,
                            "comments": [],
                            "error": None,
                        }

                        final_state = pr_review_graph.invoke(initial_state)

                        if final_state.get("error"):
                            raise RuntimeError(final_state["error"])

                        summary_text = final_state.get("summary_text") or ""
                        comments = final_state.get("comments") or []
                        # --- LANGGRAPH INVOCATION END ---

                        # Save to session state
                        st.session_state.pr_summary = summary_text
                        st.session_state.pr_comments = comments
                        st.session_state.pr_number = selected_pr.number
                        # Save metrics
                        save_review_run(repo_id, selected_pr.number, summary_text, comments)

                        # Build markdown and store it too
                        md_lines = []
                        md_lines.append(
                            f"AI Review for PR #{selected_pr.number} – {selected_pr.title}\n"
                        )
                        for c in comments:
                            md_lines.append(
                                f"- **{c.file_path}:{c.line}** "
                                f"[{c.severity.upper()}/{c.category}] – {c.body}"
                            )
                        st.session_state.pr_markdown = "\n".join(md_lines)

                    except Exception as e:
                        st.error(f"Failed to run PR review: {e}")

            # --- Render last review from session_state (survives reruns) ---
            if st.session_state.pr_summary is not None:
                summary_text = st.session_state.pr_summary
                comments = st.session_state.pr_comments
                full_review_md = st.session_state.pr_markdown

                st.subheader("AI Review Summary")
                st.write(summary_text)

                st.subheader("Review Comments")
                if not comments:
                    st.write("No significant issues found by the AI reviewer.")
                else:
                    # Group by file
                    comments_by_file = {}
                    for c in comments:
                        comments_by_file.setdefault(c.file_path, []).append(c)

                    for file_path, file_comments in comments_by_file.items():
                        st.markdown(f"**{file_path}**")
                        for c in file_comments:
                            st.markdown(
                                f"- Line {c.line} "
                                f"[{c.severity.upper()} / {c.category}] — {c.body}\n\n"
                                f"  _Why_: {c.rationale}"
                                + (f"\n\n  _Suggestion_: {c.suggestion}" if c.suggestion else "")
                            )

                    # Copy-friendly version
                    st.markdown("#### Copy all comments (Markdown)")
                    md_lines = []
                    md_lines.append(f"AI Review for PR #{selected_pr.number} – {selected_pr.title}\n")
                    for c in comments:
                        md_lines.append(
                            f"- **{c.file_path}:{c.line}** "
                            f"[{c.severity.upper()}/{c.category}] – {c.body}"
                        )
                    full_review_md = "\n".join(md_lines)
                    st.code(full_review_md, language="markdown")

                    # ---------- Phase 6: Post review back to GitHub ----------
                    st.subheader("Post review to GitHub")

                    st.markdown(
                        "This will post a single AI-generated review comment "
                        "to the PR conversation on GitHub. "
                    )

                    confirm_post = st.checkbox(
                        "I understand this will post a real comment to this PR on GitHub",
                        key="confirm_post_comment",
                    )

                    if st.button(
                        "Post AI Review Comment to GitHub", key="post_review_btn"
                    ):
                        if not confirm_post:
                            st.warning(
                                "Please check the confirmation box before posting."
                            )
                        else:
                            try:
                                resp = post_pr_issue_comment(
                                    owner=owner,
                                    repo=name,
                                    pr_number=st.session_state.pr_number,
                                    access_token=access_token,
                                    body=full_review_md,
                                )
                                comment_url = resp.get("html_url")
                                if comment_url:
                                    st.success(
                                        f"Posted review comment to GitHub: {comment_url}"
                                    )
                                else:
                                    st.success(
                                        "Posted review comment to GitHub."
                                    )
                            except Exception as e:
                                st.error(
                                    f"Failed to post review comment to GitHub: {e}"
                                )

                    

    # ------------------------
    # Tab 3: Quality Dashboard
    # ------------------------
    with tab_dashboard:
        st.markdown("### Code Quality Dashboard")

        from metrics.store import load_review_runs
        runs = load_review_runs(repo_id)

        if not runs:
            st.info("No PR reviews recorded yet. Run an AI review in the 'PR Review' tab first.")
        else:
            # Simple metrics
            

            df = pd.DataFrame(
                [
                    {
                        "created_at": r.created_at,
                        "pr_number": r.pr_number,
                        "comment_count": r.comment_count,
                        "critical": r.stats.get("by_severity", {}).get("critical", 0),
                        "warning": r.stats.get("by_severity", {}).get("warning", 0),
                        "info": r.stats.get("by_severity", {}).get("info", 0),
                        "security": r.stats.get("by_category", {}).get("security", 0),
                        "architecture": r.stats.get("by_category", {}).get("architecture", 0),
                    }
                    for r in runs
                ]
            ).sort_values("created_at")

            st.subheader("Overview")
            st.metric("Total PRs reviewed", len(df))
            st.metric("Avg comments per PR", round(df["comment_count"].mean(), 2))
            st.metric("Total critical issues", int(df["critical"].sum()))
            st.metric("Security issues (all time)", int(df["security"].sum()))

            st.subheader("Comments per PR over time")
            st.line_chart(df.set_index("created_at")[["comment_count"]])

            st.subheader("Critical issues over time")
            st.line_chart(df.set_index("created_at")[["critical"]])

            st.subheader("Security vs Architecture (total)")
            st.bar_chart(df[["security", "architecture"]].sum())

    # ------------------------
    # Tab 4: LangGraph Chat
    # ------------------------
    with tab_chat_graph:
        st.markdown("### 🧠 LangGraph Chat (multi-turn RAG over this repo)")

        # One chat history per repo in this session
        chat_state_key = f"lg_chat_history::{repo_id}"

        if chat_state_key not in st.session_state:
            st.session_state[chat_state_key] = []

        history: List[Dict[str, str]] = st.session_state[chat_state_key]

        # Build / cache a graph per repo
        graph_key = f"lg_chat_graph::{repo_id}"
        if graph_key not in st.session_state:
            st.session_state[graph_key] = build_chat_graph(repo_id)
        chat_graph = st.session_state[graph_key]

        # Input box for this tab
        user_message = st.text_input(
            "Ask anything about this repo (multi-turn, LangGraph-powered)",
            key="lg_chat_input",
        )

        if st.button("Send", key="lg_chat_send") and user_message.strip():
            user_message = user_message.strip()

            # 1) Build LangChain messages from history + new user turn
            messages: List[Any] = [
                SystemMessage(
                    content=(
                        "You are an AI code assistant helping with this GitHub repo. "
                        "Use the provided repository context to answer questions precisely. "
                        "If the codebase does not contain enough information to answer confidently, say so."
                    )
                )
            ]

            for turn in history:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content=turn["content"]))
                else:
                    messages.append(AIMessage(content=turn["content"]))

            messages.append(HumanMessage(content=user_message))

            # 2) Run the LangGraph workflow
            state = {"messages": messages}
            result = chat_graph.invoke(state)

            all_messages = result["messages"]

            # 3) Extract the last assistant message as the new answer
            last_assistant = None
            for m in reversed(all_messages):
                if isinstance(m, AIMessage):
                    last_assistant = m
                    break

            if last_assistant is None:
                answer_text = "I wasn't able to generate a response."
            else:
                answer_text = last_assistant.content

            # 4) Update session history
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": answer_text})
            st.session_state[chat_state_key] = history

        # ---- Render chat history ----
        if history:
            for turn in history:
                if turn["role"] == "user":
                    st.markdown(f"**You:** {turn['content']}")
                else:
                    st.markdown(f"**Assistant:** {turn['content']}")
        else:
            st.info("Start the conversation by asking a question about this repo.")


if __name__ == "__main__":
    main()