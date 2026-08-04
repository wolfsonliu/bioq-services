"""/v1 request/response models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class JobView(BaseModel):
    job_id: str
    account_id: str
    svc: str
    endpoint: str
    status: str
    output_prefix: Optional[str] = None
    detail: Optional[str] = None  # e.g. why a downstream status refresh failed


class PrepareUploadRequest(BaseModel):
    job_id: str
    filename: str
    sha256: Optional[str] = None


class UploadTarget(BaseModel):
    """Where/how to upload one input, minted by the active storage backend.

    Storage-agnostic: `put_url` is an OSS presigned PUT (direct-to-object) for
    the oss backend, or a gateway-relative /v1/files/<key> path for the file
    backend (client PUTs back through the gateway).
    """
    uri: str                 # oss://<bucket>/... or file:///... — injected as <field>_uri
    exists: bool             # True => already uploaded (sha256 dedup), skip PUT
    put_url: Optional[str] = None  # PUT target when exists is False
