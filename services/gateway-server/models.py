"""/v1 request/response models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class JobView(BaseModel):
    job_id: str
    principal: str
    svc: str
    endpoint: str
    status: str
    output_prefix: Optional[str] = None
    detail: Optional[str] = None  # e.g. why a downstream status refresh failed


class PresignRequest(BaseModel):
    job_id: str
    filename: str
    sha256: Optional[str] = None


class PresignResponse(BaseModel):
    uri: str                 # oss://bucket/users/<principal>/<job_id>/input/<name>
    exists: bool             # True => already uploaded, skip PUT
    url: Optional[str] = None  # presigned PUT URL when exists is False
