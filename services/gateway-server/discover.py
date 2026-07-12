"""Fetch + cache downstream /api/manifest + /openapi.json for `describe`."""

from __future__ import annotations

import time
from typing import Any

import httpx


class Discovery:
    def __init__(
        self, *, client: httpx.Client | None = None, ttl_sec: int = 300,
        timeout_sec: float = 60.0,
    ) -> None:
        # Timeout is generous: the first describe of a scaled-to-zero service
        # triggers a cold start that can exceed 15s (heavy conda images).
        self._client = client or httpx.Client(timeout=timeout_sec)
        self._ttl = ttl_sec
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}

    def describe(self, svc: str, base_url: str) -> dict[str, Any]:
        now = time.time()
        if cached := self._cache.get(svc):
            info, exp = cached
            if exp > now:
                return info
        manifest = self._get_json(f"{base_url}/api/manifest")
        openapi = self._get_json(f"{base_url}/openapi.json")
        info = {"service": svc, "manifest": manifest, "openapi": openapi}
        # Only cache a fully-successful fetch. A partial result (e.g. a cold-start
        # timeout on one sub-call) would otherwise be stuck for the whole TTL;
        # leaving it uncached makes the next describe retry.
        if manifest and openapi:
            self._cache[svc] = (info, now + self._ttl)
        return info

    def _get_json(self, url: str) -> dict[str, Any]:
        try:
            r = self._client.get(url)
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001 — describe degrades gracefully
            return {}
