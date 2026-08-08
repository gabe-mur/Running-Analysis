"""Small filesystem safeguards for athlete-owned local data."""

from __future__ import annotations

import os
from pathlib import Path


def private_directory(path: str | Path) -> Path:
    """Create a directory and restrict it to the current OS user on POSIX."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        target.chmod(0o700)
    return target


def private_file(path: str | Path) -> Path:
    """Restrict an existing file to the current OS user on POSIX."""
    target = Path(path)
    if target.exists() and os.name != "nt":
        target.chmod(0o600)
    return target


def private_tree(path: str | Path) -> Path:
    """Restrict an app-owned data tree without following symbolic links."""
    target = private_directory(path)
    if os.name == "nt":
        return target
    for current, directories, files in os.walk(target, followlinks=False):
        current_path = Path(current)
        current_path.chmod(0o700)
        directories[:] = [
            name for name in directories if not (current_path / name).is_symlink()
        ]
        for name in files:
            child = current_path / name
            if not child.is_symlink():
                child.chmod(0o600)
    return target
