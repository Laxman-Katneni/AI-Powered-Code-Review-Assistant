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



def ensure_index_exists(repo_id: str) -> bool:
    try:
        load_index(repo_id)
        return True
    except FileNotFoundError:
        return False

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

    st.title("AI Code Review Assistant (Phase 1: Local Repo RAG)")
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

    # --- Indexing Block ---
    if st.sidebar.button("Fetch & Index selected repo"):
        try:
            with st.spinner("Cloning/updating & indexing repo..."):
                local_path, commit_hash = clone_or_update_repo(owner, name, access_token)
                files = list_code_files(local_path)
                chunks = chunk_repository(repo_id, files)

                # Optionally normalize paths to be relative to repo root
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

    indexed = ensure_index_exists(repo_id)
    if not indexed:
        st.info("Index this repo first using the sidebar.")
        st.stop()

    # Show index status
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

    # --- Main chat UI ---
    st.markdown("### Ask a question about this repo")
    st.markdown(
        "_Example questions:_ "
        "`Where is the main entrypoint?`, "
        "`How does authentication work?`, "
        "`Where are database models defined?`"
    )
    question = st.text_input(
        "Question",
        placeholder="e.g. Where is user authentication implemented?",
    )

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            retrieved = retrieve_chunks(repo_id, question, k=6)

            if not retrieved:
                st.warning("I couldn't find any relevant code snippets for that question.")
                return

            # Simple distance-based filter (Chroma returns smaller = closer)
            MAX_DISTANCE = 2
            filtered = [r for r in retrieved if r["score"] < MAX_DISTANCE]

            if not filtered:
                st.warning(
                    "I found code, but none of it seemed strongly related. "
                    "Try rephrasing or being more specific."
                )
                return

            answer = answer_with_rag(question, filtered)
            used_chunks = filtered


        st.markdown("#### Answer")
        st.write(answer)

        st.markdown("#### Sources")

        seen = set()
        sources = []
        for r in used_chunks:
            meta = r["metadata"]
            key = (meta["file_path"], meta["start_line"], meta["end_line"])
            if key in seen:
                continue
            seen.add(key)
            sources.append(meta)

        # GitHub links (assuming main branch by default, Later - override in sidebar if you want)
        branch = st.sidebar.text_input("Branch to link (for Sources)", value="main", key="branch_input")

        # Base repo URL for GitHub
        github_base = f"https://github.com/{owner}/{name}/blob/{branch}"

        for meta in sources:
            file_path = meta["file_path"]
            start = meta["start_line"]
            end = meta["end_line"]
            url = f"{github_base}/{file_path}#L{start}-L{end}"
            st.markdown(
                f"- [{file_path} (lines {start}-{end})]({url})",
                unsafe_allow_html=False,
            )

                # ✅ Code viewer: let user pick a source to inspect
        st.markdown("#### View source code")

        if sources:
            options = [
                f"{m['file_path']} (lines {m['start_line']}-{m['end_line']})"
                for m in sources
            ]
            selected_option = st.selectbox(
                "Select a source to view",
                options,
            )

            # Map back to metadata
            selected_meta = sources[options.index(selected_option)]

            # Locate the file in the local cloned repo
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


if __name__ == "__main__":
    main()