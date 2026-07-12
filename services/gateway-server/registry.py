"""Downstream service registry: parse services/aliyun_fc_url.md (svc -> vpc url).

File format: lines `<service-name>: https://...`, `#` comments ignored, trailing
slashes stripped. The parser is inlined here (rather than imported from
`bioagent_service.fc_testing`) because that module imports `pytest` at import
time — a test-only dependency absent from the production image.
"""

from __future__ import annotations

import re
from pathlib import Path

_URL_LINE_RE = re.compile(r"^([A-Za-z0-9_-]+):\s*(https?://\S+)\s*$")


def parse_fc_urls(md_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in Path(md_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _URL_LINE_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2).rstrip("/")
    return out


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
