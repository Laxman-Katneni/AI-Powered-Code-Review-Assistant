from __future__ import annotations

from pathlib import Path
from typing import Tuple

from git import Repo, GitCommandError

from config import REPOS_DIR

"""
Compute the local path where this repo will live.
e.g. data/repos/{owner}/{name}
"""
def get_repo_local_path(owner: str, name:str)-> Path:
    safe_owner = owner.strip()
    safe_name = name.strip()
    return (REPOS_DIR / safe_owner / safe_name).resolve()

"""
Clone the repo if it doesn't exist locally; otherwise git pull
Returns (local_path, current_commit_hash)

# Past in Phase 2 of the project- Uses a fine-grained PAT from env for HTTPS auth
Currently - Uses the user's GitHub OAuth access_token for HTTPS auth
"""
def clone_or_update_repo(owner:str, name:str, access_token: str) -> Tuple[Path, str]:
    # if not GITHUB_TOKEN:
    #     raise RuntimeError("GITHUB_TOKEN is not set. Add it to your .env for Phase 2.")
    
    if not access_token:
        raise RuntimeError("No GitHub access token provided")

    local_path = get_repo_local_path(owner, name)
    local_path.parent.mkdir(parents = True, exist_ok=True)

    repo_url = f"https://{access_token}:x-oauth-basic@github.com/{owner}/{name}.git"

    if local_path.exists() and (local_path / ".git").exists():
        # Repo already cloned: pull the latest version
        repo = Repo(local_path)
        try:
            origin = repo.remotes.origin
            origin.set_url(repo_url)  # ensure remote has tokenized URL
            origin.pull()
        except GitCommandError as e:
            # TODO: Improve logging here
            raise RuntimeError(f"Failed to pull repo: {e}") from e
    else:
        # Fresh Clone
        try:
            repo = Repo.clone_from(repo_url, local_path)
        except GitCommandError as e:
            raise RuntimeError(f"Failed to clone repo: {e}") from e

    # Determine current HEAD commit hash (for metadata)
    commit_hash = repo.head.commit.hexsha
    return local_path, commit_hash
        