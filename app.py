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
from ingestion.github_client import clone_or_update_repo


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
        page_title="AI Code Review Assistant",
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
            st.success(f"Indexed {selected_full_name} @ {commit_hash[:7]} ✅")
        except Exception as e:
            st.error(f"Failed to fetch/index repo: {e}")

    indexed = ensure_index_exists(repo_id)
    if not indexed:
        st.info("Index this repo first using the sidebar.")
        st.stop()

    # --- Main chat UI ---
    st.markdown("### Ask a question about this repo")
    question = st.text_input(
        "Question",
        placeholder="e.g. Where is user authentication implemented?",
    )
    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            retrieved = retrieve_chunks(repo_id, question, k=6)
            answer = answer_with_rag(question, retrieved)

        st.markdown("#### Answer")
        st.write(answer)

        st.markdown("#### Sources")

        seen = set()
        for r in retrieved:
            meta = r["metadata"]
            key = (meta["file_path"], meta["start_line"], meta["end_line"])
            if key in seen:
                continue
            seen.add(key)
            st.write(
                f"- `{meta['file_path']}` (lines {meta['start_line']}-{meta['end_line']})"
            )


if __name__ == "__main__":
    main()