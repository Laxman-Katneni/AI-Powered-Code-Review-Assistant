import streamlit as st
from config import validate_config

from pathlib import Path
from ingestion.file_discovery import list_code_files
from ingestion.parser import chunk_repository
from indexing.vector_store import build_index, load_index
from retrieval.retriever import retrieve_chunks
from llm.chat_llm import answer_with_rag


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

    # st.markdown("### Phase 0: Scaffolding")
    # st.write(
    #     "Right now, the app is just a skeleton. "
    #     "Next phases will add:\n"
    #     "- GitHub repo input\n"
    #     "- Code ingestion & chunking\n"
    #     "- Vector indexing\n"
    #     "- Q&A over the codebase"
    # )

    st.sidebar.header("Repository")
    local_repo_path = st.sidebar.text_input(
        "Local repo path",
        value=".",
        help="Absolute or relative path to a local Git repo.",
    )
    repo_id = f"local::{Path(local_repo_path).resolve()}"

    index_col, _ = st.sidebar.columns([1, 1])
    with index_col:
        if st.button("Index / Reindex Repo"):
            repo_root = Path(local_repo_path).resolve()
            if not repo_root.exists():
                st.error(f"Path does not exist: {repo_root}")
            else:
                with st.spinner("Indexing repo..."):
                    files = list_code_files(repo_root)
                    chunks = chunk_repository(repo_id, files)
                    build_index(repo_id, chunks)
                st.success("Index built successfully ✅")

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
        for r in retrieved:
            meta = r["metadata"]
            st.write(
                f"- `{meta['file_path']}` (lines {meta['start_line']}-{meta['end_line']})"
            )

    

if __name__ == "__main__":
    main()