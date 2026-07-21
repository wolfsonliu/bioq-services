"""Input URI resolution for bindflow-server.

Two flavors of inputs:

* single-file (`protein`, `cofactor`, `membrane`): supports UploadFile or a
  URI (`oss://` / `job://` / `file://` / `http(s)://`).
* many-files (`ligands`): supports either N UploadFiles OR a single
  `ligands_zip_uri` pointing to a zip.  Zip is unpacked into
  `<input_dir>/ligands/` with flat layout (nested dirs get flattened).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile

from bioq_service.uris import resolve_uri

from .settings import BindFlowSettings


# ---------------------------------------------------------------------------
# Single-file inputs
# ---------------------------------------------------------------------------


def resolve_single_file(
    upload: Optional[UploadFile],
    uri: Optional[str],
    dest: Path,
    settings: BindFlowSettings,
    *,
    required: bool = True,
    field_name: str = "input",
) -> Optional[Path]:
    """Persist a single-file input to `dest` from either UploadFile or URI.

    Returns `dest` on success, or None if not required and neither source
    was provided.  Raises HTTPException(422) for validation errors.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if upload is not None and getattr(upload, "filename", None):
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        return dest

    if uri:
        return resolve_uri(uri, dest, settings)

    if required:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} is required: provide multipart file or {field_name}_uri.",
        )
    return None


# ---------------------------------------------------------------------------
# Multi-file (ligands)
# ---------------------------------------------------------------------------


def resolve_ligands(
    uploads: Optional[list[UploadFile]],
    zip_uri: Optional[str],
    dest_dir: Path,
    settings: BindFlowSettings,
) -> Path:
    """Materialize ligands under `dest_dir`.  Returns `dest_dir`."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Filter uploads that FastAPI hands us as empty when the field is
    # optional-but-not-supplied (filename == "").
    real_uploads = [u for u in (uploads or []) if getattr(u, "filename", None)]

    if real_uploads and zip_uri:
        raise HTTPException(
            status_code=422,
            detail="Provide either ligands (multipart) OR ligands_zip_uri, not both.",
        )

    if real_uploads:
        for up in real_uploads:
            name = _safe_filename(up.filename or "", field="ligands")
            with (dest_dir / name).open("wb") as f:
                shutil.copyfileobj(up.file, f)
        return dest_dir

    if zip_uri:
        zip_tmp = dest_dir.parent / "ligands.zip"
        resolve_uri(zip_uri, zip_tmp, settings)
        _unzip_flat(zip_tmp, dest_dir)
        zip_tmp.unlink(missing_ok=True)
        return dest_dir

    raise HTTPException(
        status_code=422,
        detail="ligands is required: provide multipart files OR ligands_zip_uri.",
    )


def resolve_dir_zip(
    upload: Optional[UploadFile],
    zip_uri: Optional[str],
    dest_dir: Path,
    settings: BindFlowSettings,
    *,
    field_name: str = "input",
) -> Optional[Path]:
    """Fetch + unzip a directory-shaped input (custom_ff, topology).

    Returns `dest_dir` when populated, None if neither source was given.
    """
    zip_path: Optional[Path] = None
    if upload is not None and getattr(upload, "filename", None):
        zip_path = dest_dir.parent / f"{field_name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zip_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
    elif zip_uri:
        zip_path = dest_dir.parent / f"{field_name}.zip"
        resolve_uri(zip_uri, zip_path, settings)

    if zip_path is None:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    _unzip_preserve(zip_path, dest_dir)
    zip_path.unlink(missing_ok=True)
    return dest_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_filename(name: str, *, field: str) -> str:
    """Reject filenames that contain path separators or are empty."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field} filename: {name!r}",
        )
    return name


def _unzip_flat(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip so that all files land directly in dest_dir (no nested dirs).

    Duplicate basenames across nested dirs are rejected — ligand names must be unique
    across the whole zip.
    """
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = Path(info.filename).name
            if not base:
                continue
            if base in seen:
                raise HTTPException(
                    status_code=422,
                    detail=f"Duplicate filename in ligands zip: {base}",
                )
            seen.add(base)
            target = dest_dir / base
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _unzip_preserve(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip while preserving directory structure, guarding against path traversal."""
    dest_dir_abs = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest_dir_abs / info.filename).resolve()
            if not str(target).startswith(str(dest_dir_abs) + "/") and target != dest_dir_abs:
                raise HTTPException(
                    status_code=422,
                    detail=f"Zip entry escapes destination: {info.filename}",
                )
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


__all__ = [
    "resolve_dir_zip",
    "resolve_ligands",
    "resolve_single_file",
]
