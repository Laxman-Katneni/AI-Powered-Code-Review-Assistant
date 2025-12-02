import os
from dotenv import load_dotenv
from pathlib import Path

# Loading .env file
BASE_DIR = Path(__file__).resolve().parent

# Reads and injects them into the environment
load_dotenv(BASE_DIR / ".env")

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # TEMP: we'll move to OAuth later

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
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    # GITHUB_TOKEN is optional later when OAuth is in place

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    
    # To Ensure directories exist
    REPOS_DIR.mkdir(parents = True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)