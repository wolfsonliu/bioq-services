"""Input URI resolution shared across all proteinmpnn-server endpoints.

Mirrors the scheme set used by ppiflow-server / rfantibody-server so an agent
learns one convention across services: upload / job:// / file:// / oss:// / http(s)://.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile

from .settings import ProteinMPNNSettings


def save_upload(upload: UploadFile, dest: Path) -> Path:
    """Stream an UploadFile to disk in 1 MB chunks."""
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
    settings: ProteinMPNNSettings,
) -> Path:
    """Materialize an input file from either a multipart upload or a URI string."""
    if input_uri:
        return _resolve_uri(input_uri, dest, settings)
    if upload is not None:
        return save_upload(upload, dest)
    raise HTTPException(status_code=422, detail="Either an upload or input_uri is required.")


def _resolve_uri(uri: str, dest: Path, settings: ProteinMPNNSettings) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if uri.startswith("job://"):
        # job://<job_id>/<filename> — pull from this service's own NAS root.
        body = uri[len("job://"):]
        try:
            job_id, filename = body.split("/", 1)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid job URI; expected job://<job_id>/<filename>: {uri}",
            ) from None
        src = settings.jobs_base_dir / job_id / "output" / filename
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"File not found in job: {src}")
        shutil.copy2(src, dest)
        return dest

    if uri.startswith("file://") or uri.startswith("/"):
        path = Path(uri[len("file://"):] if uri.startswith("file://") else uri)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        shutil.copy2(path, dest)
        return dest

    if uri.startswith("oss://"):
        return _download_oss(uri, dest, settings)

    if uri.startswith("http://") or uri.startswith("https://"):
        return _download_http(uri, dest)

    raise HTTPException(status_code=422, detail=f"Unsupported URI scheme: {uri}")


def _download_http(uri: str, dest: Path) -> Path:
    import httpx
    try:
        with httpx.stream("GET", uri, follow_redirects=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch {uri}: HTTP {e.response.status_code}",
        ) from None
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch {uri}: {e}") from None
    return dest


def _download_oss(uri: str, dest: Path, settings: ProteinMPNNSettings) -> Path:
    try:
        import alibabacloud_oss_v2 as oss
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="alibabacloud-oss-v2 not installed; cannot resolve oss:// URIs.",
        ) from None

    body = uri[len("oss://"):]
    try:
        bucket, key = body.split("/", 1)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid OSS URI; expected oss://<bucket>/<key>: {uri}",
        ) from None

    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider
    cfg.region = settings.oss_region

    client = oss.Client(cfg)
    request = oss.models.GetObjectRequest(bucket=bucket, key=key)
    response = client.get_object(request)
    with open(dest, "wb") as f:
        for chunk in response.body.iter_bytes():
            f.write(chunk)
    return dest
