import os
from dotenv import load_dotenv
from pathlib import Path

# Loading .env file
BASE_DIR = Path(__file__).resolve().parent

# Reads and injects them into the environment
load_dotenv(BASE_DIR / ".env")

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OAuth
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# Optional - Only for internal development
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # TEMP: we'll move to OAuth later



# Paths
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = DATA_DIR / "repos"
INDEXES_DIR = DATA_DIR / "indexes"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-nano"

# Basic sanity check

def validate_config():
    missing = []
    # GITHUB_TOKEN is optional - only for local dev
    # if not GITHUB_TOKEN:
    #     missing.append("GITHUB_TOKEN (fine-grained PAT for Phase 2)")

    if not GITHUB_CLIENT_ID:
        missing.append("GITHUB_CLIENT_ID")
    if not GITHUB_CLIENT_SECRET:
        missing.append("GITHUB_CLIENT_SECRET")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    
    # To Ensure directories exist
    REPOS_DIR.mkdir(parents = True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)