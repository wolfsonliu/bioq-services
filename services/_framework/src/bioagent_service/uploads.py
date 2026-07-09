"""File upload + zip extraction with path-traversal protection."""

from __future__ import annotations

import uuid
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


def safe_basename(filename: str | None) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips any directory components (so `../../etc/passwd` -> `passwd`) and
    falls back to `upload.bin` when the result is empty or a bare dot segment.
    """
    name = Path(filename or "").name
    if name in ("", ".", ".."):
        return "upload.bin"
    return name


def stage_upload(
    stream: IO[bytes],
    base_dir: Path,
    filename: str | None,
    *,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Stream an upload into a fresh UUID-keyed subdir under *base_dir*.

    Each call gets its own `<base_dir>/<uuid4-hex>/<safe_name>` path, so
    concurrent uploads of identically-named files never collide. Returns the
    absolute destination path; the caller turns it into a `file://` URI.
    """
    dest = base_dir / uuid.uuid4().hex / safe_basename(filename)
    save_upload(stream, dest, chunk_size=chunk_size)
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


__all__ = ["save_upload", "safe_basename", "stage_upload", "extract_dataset"]
