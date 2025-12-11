# auth/github_pr_client.py
from __future__ import annotations

from typing import List, Dict, Any

import requests

from pr.models import PRInfo
from utils.retry import with_retry

GITHUB_API_BASE = "https://api.github.com"

# returns the auth info to put in request headerr
def _auth_headers(access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
    }




@with_retry()
def _get_with_retry(url: str, headers: dict, params: dict | None = None):
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp


@with_retry()
def _post_with_retry(url: str, headers: dict, json_data: dict):
    resp = requests.post(url, headers=headers, json=json_data, timeout=10)
    resp.raise_for_status()
    return resp

# List open pull requests for a repo, sorted by most recently updated
"""
list_pull_requests(owner, repo, access_token)
Calls: GET /repos/{owner}/{repo}/pulls?state=open
Returns List[PRInfo]
"""
def list_pull_requests(owner: str, repo: str, access_token: str) -> List[PRInfo]:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls"
    params = {"state": "open", "sort": "updated", "direction": "desc"}

    resp = requests.get(url, headers=_auth_headers(access_token), params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    pr_list: List[PRInfo] = []
    repo_id = f"{owner}/{repo}"

    for pr in data:
        pr_list.append(
            PRInfo(
                repo_id=repo_id,
                number=pr["number"],
                title=pr["title"],
                author=pr["user"]["login"],
                html_url=pr["html_url"],
                base_branch=pr["base"]["ref"],
                head_branch=pr["head"]["ref"],
                body=pr.get("body"),
            )
        )

    return pr_list


# Return the list of files in a PR, including diffs (patch)
"""
Calls: GET /repos/{owner}/{repo}/pulls/{number}/files
Each file JSON includes:
filename
status
patch (unified diff text)
This is the raw material for DiffChunk
"""
def get_pull_request_files(owner: str, repo: str, pr_number: int, access_token: str) -> List[Dict[str, Any]]:

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/files"
    files: List[Dict[str, Any]] = []
    page = 1
    while True:
        resp = requests.get(
            url,
            headers=_auth_headers(access_token),
            params={"per_page": 50, "page": page},
            timeout=10,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        files.extend(batch)
        page += 1
    return files


"""
Post a single Markdown comment to the PR's conversation (Issues API)
This is safer/simpler than inline diff comments for v1
"""
def post_pr_issue_comment(
    owner: str,
    repo: str,
    pr_number: int,
    access_token: str,
    body: str,
) -> Dict[str, Any]:

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{pr_number}/comments"
    resp = requests.post(
        url,
        headers=_auth_headers(access_token),
        json={"body": body},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
