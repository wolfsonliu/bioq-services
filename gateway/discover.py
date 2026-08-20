"""Fetch + cache downstream /api/manifest + /openapi.json for `describe`.

Phase-1 hardening: bounded split timeouts, manifest-first short-circuit,
structured failure taxonomy, short-TTL negative caching, and per-service
single-flight coalescing. See
docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Literal

import httpx

FetchOutcome = Literal["ok", "warming", "no_manifest", "error"]


class Discovery:
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        ttl_sec: float = 300.0,
        negative_ttl_sec: float = 15.0,
        connect_timeout_sec: float = 5.0,
        read_timeout_sec: float = 8.0,
    ) -> None:
        if client is None:
            # httpx 0.28 requires a default or all four params; bound every
            # phase explicitly (write/pool ride the read budget).
            client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=connect_timeout_sec,
                    read=read_timeout_sec,
                    write=read_timeout_sec,
                    pool=read_timeout_sec,
                )
            )
        self._client = client
        self._ttl = ttl_sec
        self._negative_ttl = negative_ttl_sec
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def describe(self, svc: str, base_url: str) -> dict[str, Any]:
        if info := self._cached(svc):
            return info
        with self._lock_for(svc):
            if info := self._cached(svc):
                return info
            info = self._fetch(svc, base_url)
            ttl = self._cache_ttl(info["status"])
            if ttl > 0:
                self._cache[svc] = (info, time.time() + ttl)
            return info

    # ---- internal ----

    def _cached(self, svc: str) -> dict[str, Any] | None:
        entry = self._cache.get(svc)
        if entry is None:
            return None
        info, exp = entry
        return info if exp > time.time() else None

    def _lock_for(self, svc: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(svc)
            if lock is None:
                lock = threading.Lock()
                self._locks[svc] = lock
            return lock

    def _cache_ttl(self, status: str) -> float:
        if status == "ok":
            return self._ttl
        if status in ("warming", "error"):
            return self._negative_ttl
        return 0.0  # partial / no_manifest: never cached

    def _fetch(self, svc: str, base_url: str) -> dict[str, Any]:
        m_status, manifest = self._get_json(f"{base_url}/api/manifest")
        if m_status != "ok":
            # Short-circuit: never hit /openapi.json once the manifest already
            # failed (cold start / missing framework / hard error).
            return self._info(svc, {}, {}, m_status)
        o_status, openapi = self._get_json(f"{base_url}/openapi.json")
        status = "ok" if (openapi and o_status == "ok") else "partial"
        return self._info(svc, manifest, openapi, status)

    def _info(self, svc: str, manifest: dict, openapi: dict, status: str) -> dict[str, Any]:
        info = {
            "service": svc,
            "manifest": manifest,
            "openapi": openapi,
            "status": status,
            "source": "live",
        }
        if status != "ok":
            info["detail"] = _DETAIL[status]
        return info

    def _get_json(self, url: str) -> tuple[FetchOutcome, dict[str, Any]]:
        try:
            r = self._client.get(url)
            if r.status_code == 404:
                return "no_manifest", {}
            if r.status_code in (502, 504):
                return "warming", {}
            r.raise_for_status()
            return "ok", r.json()
        except httpx.TimeoutException:
            return "warming", {}
        except httpx.NetworkError:
            return "warming", {}
        except httpx.HTTPStatusError:
            return "error", {}
        except ValueError:  # 200-but-not-JSON
            return "error", {}
        except Exception:  # noqa: BLE001 — describe degrades gracefully
            return "error", {}


_DETAIL = {
    "warming": "downstream cold-start timed out; retry in ~15s",
    "no_manifest": "service has no /api/manifest (framework self-description not adopted)",
    "partial": "manifest ok; openapi unavailable (CLI/human path unaffected)",
    "error": "downstream returned an unexpected error",
}
