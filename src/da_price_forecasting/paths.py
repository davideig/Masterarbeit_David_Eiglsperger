from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward from *start*."""
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
        if (candidate / "pipeline").exists() and (candidate / "requirements").exists():
            return candidate

    raise FileNotFoundError("Could not determine repository root.")


def resolve_path(path: Path, repo_root: Path) -> Path:
    """Resolve a potentially relative path against the repository root."""
    return path if path.is_absolute() else (repo_root / path).resolve()

