"""Input URI resolution for chembounce-server.

ChemBounce takes SMILES as a string, not a file — so this URI resolver is
slim: it just supports an `input_smiles_uri` form field for chained
pipelines (e.g. a previous job's output SMILES).  Most calls will pass
`input_smiles` as a plain form field.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import HTTPException

from .settings import ChemBounceSettings


def resolve_smiles_uri(uri: str, settings: ChemBounceSettings) -> str:
    """Fetch a SMILES from a URI and return the string content (first non-empty line)."""

    if uri.startswith("job://"):
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
            raise HTTPException(
                status_code=404, detail=f"File not found in job: {src}",
            )
        return _read_first_smiles(src)

    if uri.startswith("file://") or uri.startswith("/"):
        path = Path(uri[len("file://"):] if uri.startswith("file://") else uri)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        return _read_first_smiles(path)

    if uri.startswith("oss://"):
        return _read_oss(uri, settings)

    if uri.startswith("http://") or uri.startswith("https://"):
        return _read_http(uri)

    raise HTTPException(status_code=422, detail=f"Unsupported URI scheme: {uri}")


def _read_first_smiles(path: Path) -> str:
    """Read the first non-empty line from a file; treat it as the SMILES."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise HTTPException(
        status_code=422,
        detail=f"No SMILES content in {path}; first non-empty line is required.",
    )


def _read_http(uri: str) -> str:
    import httpx
    try:
        with httpx.Client(timeout=60) as c:
            r = c.get(uri, follow_redirects=True)
            r.raise_for_status()
            text = r.text
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch {uri}: HTTP {e.response.status_code}",
        ) from None
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch {uri}: {e}",
        ) from None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise HTTPException(
        status_code=422, detail=f"No SMILES content at {uri}.",
    )


def _read_oss(uri: str, settings: ChemBounceSettings) -> str:
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
    tmp_path = Path("/tmp") / f"chembounce_smiles_{abs(hash(uri))}.txt"
    with open(tmp_path, "wb") as f:
        for chunk in response.body.iter_bytes():
            f.write(chunk)
    try:
        return _read_first_smiles(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# Kept for symmetry with other services that COPY raw files.
def save_text(content: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    return dest


# Shutdown noise — quiet unused import in a few code paths.
_ = shutil  # noqa: F841
