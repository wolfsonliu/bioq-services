"""Rewrite oss://<data-bucket>/<key> input URIs to the downstream's OSS mount path.

Services that mount the data-plane OSS bucket at /mnt/oss can read inputs
straight from the filesystem (their uris.py handles file://+absolute paths),
so no OSS SDK credentials are needed downstream. The gateway maps
`oss://<bucket>/<key>` -> `<mount>/<key>` in the run body for such services.
Values of other buckets/schemes are left untouched. Recurses into list/dict.
"""

from __future__ import annotations

from typing import Any


def _map(value: Any, prefix: str, mount: str) -> Any:
    if isinstance(value, str) and value.startswith(prefix):
        return f"{mount.rstrip('/')}/{value[len(prefix):]}"
    if isinstance(value, list):
        return [_map(v, prefix, mount) for v in value]
    if isinstance(value, dict):
        return {k: _map(v, prefix, mount) for k, v in value.items()}
    return value


def map_oss_inputs_to_mount(body: dict, *, bucket: str, mount: str) -> dict:
    prefix = f"oss://{bucket}/"
    return {k: _map(v, prefix, mount) for k, v in body.items()}
