import streamlit as st
from config import validate_config

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

    st.title("AI Code Review Assistant")
    st.write("Welcome! This is the starting point for your codebase assistant")

    st.markdown("### Phase 0: Scaffolding")
    st.write(
        "Right now, the app is just a skeleton. "
        "Next phases will add:\n"
        "- GitHub repo input\n"
        "- Code ingestion & chunking\n"
        "- Vector indexing\n"
        "- Q&A over the codebase"
    )

if __name__ == "__main__":
    main()