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

from auth.github_pr_client import list_pull_requests, get_pull_request_files
from pr.diff_ingestion import build_diff_chunks_from_github_files
from pr.review_service import run_pr_review
from metrics.store import save_review_run, load_review_runs
from pr.models import PRInfo, ReviewComment
import pandas as pd
from typing import Optional, Dict, Any, List

# ------------- Helpers -------------

def ensure_index_exists(repo_id: str) -> bool:
    try:
        load_index(repo_id)
        return True
    except FileNotFoundError:
        return False

# ------------- Main App -------------

def main():
    # Validate config - Error early if any API Key is missing
    try:
        validate_config()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    st.set_page_config(
        page_title="AI Code Review Assistant - GitHub RAG",
        layout="wide",
    )

    st.title("AI Code Review Assistant")
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
    tab_chat, tab_pr, tab_dashboard = st.tabs(["💬 Code Q&A", "🔍 PR Review", "📊 Quality Dashboard"])

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
            with st.spinner("Thinking..."):
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

            if st.button("Run AI Review", key="run_pr_review"):
                with st.spinner("Running AI code review..."):
                    try:
                        files_json = get_pull_request_files(owner, name, selected_pr.number, access_token)
                        diff_chunks = build_diff_chunks_from_github_files(selected_pr.repo_id, selected_pr.number, files_json)
                        summary_text, comments = run_pr_review(repo_id, owner, name, selected_pr, diff_chunks)

                        # Save metrics
                        save_review_run(repo_id, selected_pr.number, summary_text, comments)

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

                            # Optional: copy-friendly version
                            st.markdown("#### Copy all comments (Markdown)")
                            md_lines = []
                            md_lines.append(f"AI Review for PR #{selected_pr.number} – {selected_pr.title}\n")
                            for c in comments:
                                md_lines.append(
                                    f"- **{c.file_path}:{c.line}** "
                                    f"[{c.severity.upper()}/{c.category}] – {c.body}"
                                )
                            st.code("\n".join(md_lines), language="markdown")

                    except Exception as e:
                        st.error(f"Failed to run PR review: {e}")

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



if __name__ == "__main__":
    main()