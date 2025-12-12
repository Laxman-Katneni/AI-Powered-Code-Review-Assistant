# AI Code Review Assistant (Pre-Migration Version)

An AI-powered system that connects directly to GitHub repositories to help developers **understand, review, and improve codebases** using Retrieval-Augmented Generation (RAG) and large language models.

Does **PR(Pull Request) Review** on the selected PR and optionally posts the AI review back to GitHub as a PR comment

This version is implemented using **Python + Streamlit** and serves as the functional reference and prototype for a future full-stack migration (React + Spring Boot).

---

## 🚀 Features

### 🔐 GitHub Integration
- GitHub OAuth authentication
- Lists user repositories and pull requests
- Secure API access using GitHub tokens

### 📦 Repository Indexing (RAG)
- Clones or updates a selected GitHub repository
- Discovers source code files automatically
- Chunks code intelligently and embeds it using OpenAI embeddings
- Stores embeddings in a vector store for fast semantic retrieval
- Tracks index metadata (commit hash, file count, chunk count)

### 💬 Code Q&A (RAG-based)
- Ask natural language questions about a codebase
- Retrieves relevant code snippets using semantic search
- Generates contextual answers grounded in the actual repository code
- Displays:
  - Answer text
  - Source files with line ranges
  - Direct GitHub links
  - Local code viewer

### 🧠 Semantic Caching
- Uses embeddings to detect semantically similar questions
- Reuses previous answers when similarity exceeds a threshold
- Cache is scoped by:
  - Repository
  - Index version (commit hash / indexed_at)
- Prevents stale or incorrect answers after re-indexing

### 🔍 AI Pull Request Review
- Lists open pull requests for a repository
- Fetches PR diffs and changed files
- Builds contextual prompts using:
  - PR diff chunks
  - Relevant existing code retrieved via RAG
- Uses a structured LLM prompt to generate:
  - High-level PR summary
  - Structured review comments with:
    - File path
    - Line number
    - Severity (info / warning / critical)
    - Category (security, architecture, bug-risk, performance, etc.)
    - Rationale and suggestions
- Optionally posts the AI review back to GitHub as a PR comment

### 📊 Code Quality Dashboard
- Tracks historical PR review runs
- Aggregates metrics such as:
  - Total PRs reviewed
  - Average comments per PR
  - Critical issues over time
  - Security vs architecture issues
- Visualized using Streamlit charts

### 🔗 LangGraph Workflows
- PR Review implemented as a **LangGraph workflow**:
  - Context retrieval node
  - LLM review node
  - Structured parsing node
- Separate **LangGraph-based multi-turn chat tab**:
  - Maintains conversation history
  - Injects repository context per turn
  - Enables follow-up questions and deeper exploration

---

## 🧱 Architecture Overview (Pre-Migration)

**Frontend**
- Streamlit (single-page interactive UI)

**Backend / Logic**
- Python services organized by domain:
  - `auth/` – GitHub OAuth and API access
  - `ingestion/` – Repo cloning and file discovery
  - `indexing/` – Vector store and index metadata
  - `retrieval/` – Semantic retrieval over embeddings
  - `pr/` – PR diff ingestion and AI review logic
  - `metrics/` – Review run tracking and aggregation
  - `graphs/` – LangGraph workflows (PR review, chat)

**AI / LLM**
- OpenAI Chat API for reasoning and review generation
- OpenAI Embeddings for:
  - Code chunk indexing
  - Semantic caching
  - Question similarity detection

---

## 🛠️ Local Setup

### Prerequisites
- Python 3.10+
- Git
- OpenAI API key
- GitHub OAuth app credentials

### Install dependencies
```bash
pip install -r requirements.txt
```
### Environment Variables
```bash
OPENAI_API_KEY=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
```

### Run the App
```bash
streamlit run app.py
```

## 🔄 Webhook Integration (Planned – TODO)

> **Note:** Webhook-based automation is **intentionally not part of this pre-migration version**. It will be added as a core feature of the production-ready backend.

### Planned Webhook Features
* **Event Listener:** Listening for `pull_request` events.
* **Security:** Signature verification using `X-Hub-Signature-256`.
* **Idempotency:** Processing based on unique combinations of `(repo, pr_number, head_sha)`.
* **Automation Triggers:**
    * **PR Opened/Sync:** Auto-trigger repository indexing and AI review.
    * **Background Jobs:** Async execution of reviews to avoid timeouts.
* **Implementation Plan:** This will be implemented as a separate service (FastAPI or Spring Boot) utilizing a task queue for long-running review jobs.

---

## 🧭 Future Work (Post-Migration)

This project is currently in **Phase 1 (Prototype)**. The roadmap for **Phase 2 (Production)** includes:

* **Frontend Migration:** Move from Streamlit to **React + TypeScript** for a custom, highly responsive UI.
* **Backend Migration:** Port logic to **Java Spring Boot**.
* **Database:** Implement **PostgreSQL** for persistent storage of users, reviews, and metrics.
* **Vector Store:** Migrate to a scalable solution like **pgvector** or Pinecone.
* **Security:** Implement stateless JWT-based authentication.
* **Automation:** Full webhook-driven pipeline with retries and rate limiting.

---

## 🎯 Why This Project?

This project was built to explore and demonstrate:

* **RAG over Code:** Going beyond simple text RAG to handle code structure and syntax.
* **LLM Reasoning:** Using AI not just to write code, but to **review** architecture and logic.
* **LangGraph Pipelines:** Orchestrating complex, multi-step AI workflows.
* **System Design:** Planning for a scalable, modular architecture from day one.

