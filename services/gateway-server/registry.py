"""Downstream service registry: load services/services.yaml (svc -> record).

Thin wrapper over the framework's `load_services` (which parses the YAML
registry into `ServiceRecord`s). Kept as a small class so the app can hold it
on `app.state` and reload it if needed.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.service_registry import ServiceRecord, load_services


class ServiceRegistry:
    def __init__(self, registry_path: Path) -> None:
        self._path = Path(registry_path)
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
