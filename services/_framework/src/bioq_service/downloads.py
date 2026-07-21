"""Output packaging + safe single-file resolution."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path


def archive_dir(src_dir: Path) -> io.BytesIO:
    """Zip the contents of `src_dir` (recursively) into an in-memory buffer.

    Member paths are relative to `src_dir` so the zip extracts to the user's
    chosen directory without a redundant top-level folder. The buffer's read
    pointer is rewound before return.
    """
    buf = io.BytesIO()
    if not src_dir.is_dir():
        # Return an empty zip rather than raising; routes layer handles "no files".
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED):
            pass
        buf.seek(0)
        return buf

    src = src_dir.resolve()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src))
    buf.seek(0)
    return buf


def list_files(src_dir: Path) -> list[str]:
    """Return file paths (relative to src_dir) for every file under it."""
    if not src_dir.is_dir():
        return []
    src = src_dir.resolve()
    return sorted(
        str(f.relative_to(src)) for f in src.rglob("*") if f.is_file()
    )


def safe_subpath(base: Path, requested: str) -> Path:
    """Resolve `base/requested`, asserting the result stays inside `base`.

    Raises `ValueError` on any traversal attempt. Callers map this to HTTP 400.
    """
    base_resolved = base.resolve()
    candidate = (base_resolved / requested).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as e:
        raise ValueError(f"path traversal rejected: {requested!r}") from e
    return candidate


__all__ = ["archive_dir", "list_files", "safe_subpath"]
