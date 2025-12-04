from pathlib import Path
from typing import Iterable, List

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
}

DEFAULT_ALLOWED_EXTS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".rs",
    ".cpp",
    ".c",
    ".cs",
    ".php",
    ".scala",
    ".txt",
}

# Recursively listing code files under repo_root, applying basic filters
def list_code_files(
        repo_root: Path,
        allowed_exts: Iterable[str] = None,
        excluded_dirs: Iterable[str] = None
) -> List[Path]:
    
    repo_root = Path(repo_root).resolve()
    excluded = set(excluded_dirs or DEFAULT_EXCLUDED_DIRS)
    allowed = set(allowed_exts or DEFAULT_ALLOWED_EXTS)

    code_files: List[Path] = []

    for path in repo_root.rglob("*"):
        # Directories: skip excluded
        if path.is_dir():
            if path.name in excluded:
                # Skip this directory and its children
                # rglob handles skipping automatically if we don’t descend manually
                continue
            continue
        
        # Files: check extension
        if path.suffix in allowed:
            code_files.append(path)
    
    return code_files
    
