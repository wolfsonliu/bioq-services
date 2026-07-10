"""Downstream service registry: parse services/aliyun_fc_url.md (svc -> vpc url).

Reuses bioagent_service.fc_testing.parse_fc_urls so the file format stays in
one place: lines `<service-name>: https://...`, `#` comments ignored.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service.fc_testing import parse_fc_urls


class ServiceRegistry:
    def __init__(self, registry_path: Path) -> None:
        self._path = Path(registry_path)
        self._urls: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self._urls = parse_fc_urls(self._path)

    def list(self) -> list[str]:
        return sorted(self._urls)

    def base_url(self, svc: str) -> str:
        if svc not in self._urls:
            raise KeyError(svc)
        return self._urls[svc]
