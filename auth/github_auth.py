from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import List, Dict, Any

from urllib.parse import urlencode

import requests

from config import GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_BASE = "https://api.github.com"


@dataclass
class GitHubUser:
    login: str
    id: int
    avatar_url: str | None = None
    html_url: str | None = None

# Random CSRF state token
def generate_state()->str:
    return secrets.token_urlsafe(16)


"""
Build GitHub OAuth authorization URL
Scope 'repo' covers private repos; for only public, use 'read:user repo:status' etc
"""
def get_authorize_url(state: str) -> str:
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "scope": "repo", # adjust scope as needed
        "state": state
    }

    return f"{GITHUB_AUTH_URL}?{urlencode(params)}"


# Exchange temporary OAuth 'code' for an access token
def exchange_code_for_token(code: str)-> str:
    headers = {"Accept": "application/json"}
    data = {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "code": code,
    }
    resp = requests.post(GITHUB_TOKEN_URL, headers=headers, data=data, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    if "access_token" not in payload:
        raise RuntimeError(f"Failed to obtain access token: {payload}")
    return payload["access_token"]


def get_user(access_token: str) -> GitHubUser:
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"}
    resp = requests.get(f"{GITHUB_API_BASE}/user", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return GitHubUser(
        login=data["login"],
        id=data["id"],
        avatar_url=data.get("avatar_url"),
        html_url=data.get("html_url"),
    )


"""
Return list of repos for the authenticated user
Each item includes at least 'full_name' and 'clone_url'
"""
def get_user_repos(access_token: str) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"}
    repos: List[Dict[str, Any]] = []
    page = 1
    while True:
        resp = requests.get(
            f"{GITHUB_API_BASE}/user/repos",
            headers=headers,
            params={"per_page": 50, "page": page, "sort": "updated"},
            timeout=10,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        repos.extend(batch)
        page += 1
    return repos