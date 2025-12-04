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
        page_title = "AI Code Review Assistant",
        layout= "wide"
    )

    st.title("AI Code Review Assistant (Phase 1: Local Repo RAG)")
    st.write("Welcome! This is the starting point for your codebase assistant")

    # --- Session state setup ---
    if "gh_access_token" not in st.session_state:
        st.session_state.gh_access_token = None
    if "gh_user" not in st.session_state:
        st.session_state.gh_user = None
    if "gh_state" not in st.session_state:
        st.session_state.gh_state = None
    
    # --- Handle OAuth callback ---
    query_params = st.query_params

    if "code" in query_params and st.session_state.gh_access_token is None:
        code = query_params["code"]
        returned_state = query_params.get("state")

        if st.session_state.gh_state and returned_state != st.session_state.gh_state:
            st.error("State mismatch during GitHub OAuth. Please try logging in again.")
        else:
            try:
                access_token = exchange_code_for_token(code)
                user = get_user(access_token)
                st.session_state.gh_access_token = access_token
                st.session_state.gh_user = user

                # Clear query params so refresh URL looks clean
                st.query_params = {}

                st.success(f"Logged in as {user.login}")

            except Exception as e:
                st.error(f"GitHub OAuth failed: {e}")


    st.sidebar.header("GitHub")

    # --- If not logged in, show login button ---
    if not st.session_state.gh_access_token:
        st.sidebar.write("Connect your GitHub account to index your repos.")

        if st.sidebar.button("Login with GitHub"):
            state = generate_state()
            st.session_state.gh_state = state
            auth_url = get_authorize_url(state)
            st.markdown(
                f'<meta http-equiv="refresh" content="0; url={auth_url}"/>',
                unsafe_allow_html=True,
            )
        st.info("Log in with GitHub to continue.")
        st.stop()
    
    # If logged in:
    gh_user = st.session_state.gh_user
    st.sidebar.success(f"Logged in as {gh_user.login}")

    # Logout option
    if st.sidebar.button("Logout"):
        # st.session_state.gh_access_token = None
        # st.session_state.gh_user = None
        # st.session_state.gh_state = None

        # Completely remove GitHub-related keys from session_state
        for key in ["gh_access_token", "gh_user", "gh_state"]:
            if key in st.session_state:
                del st.session_state[key]
        # Clear query params in the URL
        st.query_params = {}
        # Rerun the app with a clean state
        st.rerun()

    # --- Repo selection ---
    access_token = st.session_state.gh_access_token
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
    question = st.text_input("Question", placeholder="e.g. Where is user authentication implemented?")
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