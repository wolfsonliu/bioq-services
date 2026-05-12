"""File upload + zip extraction with path-traversal protection."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import IO, Iterable


def save_upload(stream: IO[bytes], dest: Path, *, chunk_size: int = 1024 * 1024) -> Path:
    """Stream a file-like object to disk in chunks (so multi-GB uploads don't OOM)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    return dest


def _validate_zip_members(names: Iterable[str]) -> None:
    """Reject path-traversal attempts in archive member names.

    A "safe" name is relative (no leading /) and contains no `..` segments.
    """
    for name in names:
        if name.startswith("/") or name.startswith("\\"):
            raise ValueError(f"absolute path in zip not allowed: {name!r}")
        # Normalize separators for the check.
        parts = name.replace("\\", "/").split("/")
        if any(p == ".." for p in parts):
            raise ValueError(f"parent traversal in zip not allowed: {name!r}")


def extract_dataset(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a zip into dest_dir/ and return the resolved dest path.

    - `dest_dir` is created if missing.
    - All members are pre-validated; any traversal attempt aborts before extraction.
    - Raises `zipfile.BadZipFile` or `ValueError` on invalid archives — caller maps
      these to HTTP 422.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        _validate_zip_members(zf.namelist())
        zf.extractall(dest_dir)
    return dest_dir.resolve()


__all__ = ["save_upload", "extract_dataset"]
