"""Downstream service registry: load services.yaml (svc -> record).

Thin wrapper over the framework's `load_services` (which parses the YAML
registry into `ServiceRecord`s). Kept as a small class so the app can hold it
on `app.state` and reload it if needed. Also serves the static describe
contracts committed under manifests/ (see
docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from bioq_service.service_registry import ServiceRecord, load_services


class ServiceRegistry:
    def __init__(self, registry_path: Path) -> None:
        self._path = Path(registry_path)
        self._manifests_dir = self._path.parent / "manifests"
        self._manifest_cache: dict[str, dict | None] = {}
        self._openapi_cache: dict[str, dict | None] = {}
        self._services: dict[str, ServiceRecord] = {}
        self.reload()

    def reload(self) -> None:
        self._services = load_services(self._path)

    def list(self) -> list[str]:
        return sorted(self._services)

    def record(self, svc: str) -> ServiceRecord:
        if svc not in self._services:
            raise KeyError(svc)
        return self._services[svc]

    def base_url(self, svc: str) -> str:
        return self.record(svc).url

    def manifest(self, svc: str) -> dict | None:
        if svc not in self._manifest_cache:
            self._manifest_cache[svc] = self._load_json(
                self._manifests_dir / f"{svc}.manifest.json"
            )
        return self._manifest_cache[svc]

    def openapi(self, svc: str) -> dict | None:
        if svc not in self._openapi_cache:
            self._openapi_cache[svc] = self._load_json(
                self._manifests_dir / f"{svc}.openapi.json"
            )
        return self._openapi_cache[svc]

    def _load_json(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
