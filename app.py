import streamlit as st
from config import validate_config

from pathlib import Path
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

    st.sidebar.header("Repository Source")
    mode = st.sidebar.radio("Select source", ["GitHub", "Local"], index=0)

    # st.markdown("### Phase 0: Scaffolding")
    # st.write(
    #     "Right now, the app is just a skeleton. "
    #     "Next phases will add:\n"
    #     "- GitHub repo input\n"
    #     "- Code ingestion & chunking\n"
    #     "- Vector indexing\n"
    #     "- Q&A over the codebase"
    # )

    repo_id = None
    local_repo_root: Path | None = None

    if mode == "Local":
        # -------- Local mode (Phase 1) --------

        local_path_str = st.sidebar.text_input(
            "Local repo path",
            value=".",
            help="Absolute or relative path to a local Git repo.",
        )
        repo_id = f"local::{Path(local_path_str).resolve()}"

        index_col, _ = st.sidebar.columns([1, 1])
        with index_col:
            if st.button("Index / Reindex Repo"):
                repo_root = Path(local_path_str).resolve()
                if not repo_root.exists():
                    st.error(f"Path does not exist: {repo_root}")
                else:
                    with st.spinner("Indexing repo..."):
                        files = list_code_files(repo_root)
                        chunks = chunk_repository(repo_id, files)
                        
                        for ch in chunks:
                            try:
                                ch.file_path = str(Path(ch.file_path).relative_to(repo_root))
                            except ValueError:
                                # If for some reason it's not under repo_root, just keep as-is
                                pass
                        build_index(repo_id, chunks)
                    st.success("Index built successfully ✅")
        local_repo_root = Path(local_path_str).resolve()
    
    else:
        # -------- GitHub mode (Phase 2) --------
        owner = st.sidebar.text_input("GitHub owner", placeholder="e.g. luckykatneni")
        name = st.sidebar.text_input("GitHub repo", placeholder="e.g. sample-repo")
        repo_id = f"github::{owner}/{name}" if owner and name else None

        if st.sidebar.button("Fetch & Index from GitHub"):
            if not owner or not name:
                st.error("Please provide both owner and repo name.")
            else:
                try:
                    with st.spinner("Cloning/updating repo from GitHub..."):
                        local_path, commit_hash = clone_or_update_repo(owner, name)
                        # You could store commit_hash somewhere or pass into chunk metadata later
                        files = list_code_files(local_path)
                        chunks = chunk_repository(repo_id, files)
                        for ch in chunks:
                            try:
                                ch.file_path = str(Path(ch.file_path).relative_to(local_path))
                            except ValueError:
                                # If for some reason it's not under repo_root, just keep as-is
                                pass


                        # Attach commit hash into metadata for each chunk if you want:
                        for ch in chunks:
                            ch.metadata["commit_hash"] = commit_hash
                        build_index(repo_id, chunks)
                    st.success(f"GitHub repo indexed successfully ✅ (HEAD {commit_hash[:7]})")
                    local_repo_root = local_path
                except Exception as e:
                    st.error(f"Failed to fetch/index repo: {e}") 

    if not repo_id:
        st.info("Select a repo source and index a repo to begin.")
        st.stop()

    indexed = ensure_index_exists(repo_id)
    if not indexed:
        st.info("Index this repo first using the sidebar.")
        st.stop()

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