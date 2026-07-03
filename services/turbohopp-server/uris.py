"""Resolve a file reference (UploadFile or URI string) to a local path.

Cloned from ``services/rfantibody-server/uris.py`` — same schemes:

  * multipart upload → ``<job_dir>/input/<name>``
  * ``file:///abs/path`` (or bare ``/abs/path``) → NAS-local copy
  * ``job://<id>/<filename>`` → pull from a previous job's output dir
  * ``oss://<bucket>/<key>``  → Aliyun OSS
  * ``http(s)://...``         → generic HTTP download
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import httpx
from fastapi import HTTPException, UploadFile

from .settings import TurboHoppSettings

logger = logging.getLogger(__name__)


def save_upload(upload: UploadFile, dest: Path) -> Path:
    """Stream an UploadFile to ``dest`` (creating parent dirs as needed)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dest


def resolve_input(
    upload: Optional[UploadFile],
    input_uri: Optional[str],
    dest: Path,
    settings: TurboHoppSettings,
) -> Path:
    """Land a file at ``dest`` from either an upload or a URI.

    Raises HTTP 422 if neither is supplied.
    """
    if input_uri:
        return _resolve_uri(input_uri, dest, settings)
    if upload is not None:
        return save_upload(upload, dest)
    raise HTTPException(
        status_code=422,
        detail="Either a file upload or `<name>_uri` is required",
    )


def _resolve_uri(uri: str, dest: Path, settings: TurboHoppSettings) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if uri.startswith("job://"):
        parts = uri[len("job://"):].split("/", 1)
        if len(parts) != 2:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid job URI, expected job://<job_id>/<filename>: {uri}",
            )
        job_id, filename = parts
        src = settings.jobs_base_dir / job_id / "output" / filename
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {src}")
        shutil.copy2(src, dest)
        return dest

    if uri.startswith("file://"):
        uri = uri[len("file://"):]
    if uri.startswith("/"):
        src = Path(uri)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {src}")
        shutil.copy2(src, dest)
        return dest

    if uri.startswith("oss://"):
        return _download_from_oss(uri, dest, settings)

    if uri.startswith("http://") or uri.startswith("https://"):
        return _download_from_url(uri, dest)

    raise HTTPException(status_code=422, detail=f"Unsupported URI scheme: {uri}")


def _download_from_url(url: str, dest: Path) -> Path:
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download {url}: HTTP {e.response.status_code}",
        ) from e
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        raise HTTPException(status_code=502, detail=f"Failed to download {url}: {e}") from e
    return dest


def _download_from_oss(uri: str, dest: Path, settings: TurboHoppSettings) -> Path:
    try:
        import alibabacloud_oss_v2 as oss
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail="alibabacloud-oss-v2 not installed; OSS URIs unsupported",
        ) from e

    parts = uri[len("oss://"):].split("/", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid OSS URI, expected oss://<bucket>/<key>: {uri}",
        )
    bucket_name, key = parts

    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider
    cfg.region = settings.oss_region

    client = oss.Client(cfg)
    request = oss.models.GetObjectRequest(bucket=bucket_name, key=key)
    response = client.get_object(request)
    with open(dest, "wb") as f:
        for chunk in response.body.iter_bytes():
            f.write(chunk)
    return dest
