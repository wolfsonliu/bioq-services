"""Pluggable storage backends for input ingest + result retrieval.

The gateway abstracts *how* inputs get in and results come out so the same code
serves Alibaba OSS (presigned direct-to-object URLs) or a local shared
filesystem (gateway-proxied /v1/files IO over a volume both gateway and workers
mount). `make_storage(settings)` selects via GATEWAY_STORAGE_BACKEND.

Both backends share the job-centric key layout:
  input  key: users/<account_id>/<job_id>/input/<filename>
  output key: users/<account_id>/<job_id>/<filename>

The OSS backend is `presign.Presigner` (it already implements this interface).
The `file` backend writes/reads under a shared base dir; workers see the same
files by mounting that volume at their output mount, and read inputs via the
`file://` URIs this backend returns (framework uris.py resolves file://).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from fastapi import HTTPException

from .models import UploadTarget
from .presign import Presigner, build_oss_client


@runtime_checkable
class StorageBackend(Protocol):
    def prepare_upload(self, account_id: str, job_id: str, filename: str,
                       sha256: str | None = None) -> UploadTarget: ...

    def result_url_if_exists(self, account_id: str, job_id: str,
                             filename: str) -> str | None: ...


class FileStorage:
    """Shared-filesystem backend. Uploads/downloads are proxied through the
    gateway's /v1/files routes; `uri`/paths point into `base_dir`, a volume that
    workers mount at the same path so `file://` inputs resolve downstream.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir).resolve()

    def _input_key(self, account_id: str, job_id: str, filename: str) -> str:
        return f"users/{account_id}/{job_id}/input/{filename}"

    def _output_key(self, account_id: str, job_id: str, filename: str) -> str:
        return f"users/{account_id}/{job_id}/{filename}"

    def resolve(self, key: str) -> Path:
        """Map a storage key to an absolute path under base_dir, rejecting any
        traversal outside base_dir (`..`, absolute keys, symlink escapes)."""
        p = (self._base / key).resolve()
        if p != self._base and self._base not in p.parents:
            raise HTTPException(400, f"invalid file key: {key!r}")
        return p

    def prepare_upload(self, account_id: str, job_id: str, filename: str,
                       sha256: str | None = None) -> UploadTarget:
        key = self._input_key(account_id, job_id, filename)
        path = self.resolve(key)
        uri = f"file://{path}"
        if path.is_file():
            return UploadTarget(uri=uri, exists=True, put_url=None)
        # Gateway-relative PUT URL; the client PUTs through the gateway (same
        # origin, so auth carries over) rather than direct-to-object.
        return UploadTarget(uri=uri, exists=False, put_url=f"/v1/files/{key}")

    def result_url_if_exists(self, account_id: str, job_id: str,
                             filename: str) -> str | None:
        key = self._output_key(account_id, job_id, filename)
        if not self.resolve(key).is_file():
            return None
        return f"/v1/files/{key}"


def make_storage(settings) -> StorageBackend:
    if settings.storage_backend == "file":
        return FileStorage(settings.file_base_dir)
    if settings.storage_backend == "oss":
        return Presigner(client=build_oss_client(settings.oss_region),
                         bucket=settings.oss_bucket, region=settings.oss_region,
                         expiry_sec=settings.presign_expiry_sec)
    raise ValueError(f"unknown GATEWAY_STORAGE_BACKEND: {settings.storage_backend!r}")
