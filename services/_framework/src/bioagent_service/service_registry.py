"""Service registry helpers — parse `services/aliyun_fc_url.md` (svc -> url).

These are **general-purpose** utilities (production code may use them to resolve
downstream service URLs), so they live in their own module with no test-only
dependencies. `bioagent_service.fc_testing` re-exports them for backward
compatibility, but new production code should import from here directly.

File format: lines `<service-name>: https://...`; `#` comments and blank lines
skipped; trailing slashes stripped so callers can do `f"{url}/api/..."`.
"""

from __future__ import annotations

import re
from pathlib import Path

_URL_LINE_RE = re.compile(r"^([A-Za-z0-9_-]+):\s*(https?://\S+)\s*$")


def find_aliyun_fc_url_md(start: Path | None = None) -> Path:
    """Walk up from `start` (or cwd) until `services/aliyun_fc_url.md` is found.

    Raises FileNotFoundError if not found — typically means the caller is
    running from outside the bioagent repo.
    """
    base = (start or Path.cwd()).resolve()
    for candidate in [base, *base.parents]:
        target = candidate / "services" / "aliyun_fc_url.md"
        if target.is_file():
            return target
    raise FileNotFoundError(f"services/aliyun_fc_url.md not found above {base!r}")


def parse_fc_urls(md_path: Path) -> dict[str, str]:
    """Parse `services/aliyun_fc_url.md` into a `{service: url}` dict.

    Lines of the form `<service>: https://...` are extracted; comments and
    blank lines are skipped. Trailing slashes are stripped from URLs.
    """
    out: dict[str, str] = {}
    for raw in Path(md_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _URL_LINE_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2).rstrip("/")
    return out


def fc_url(service_name: str, *, start: Path | None = None) -> str:
    """Resolve the deployed FC URL for `service_name`.

    `start` controls where the upward walk for `aliyun_fc_url.md` begins —
    pass `Path(__file__)` from a test/conftest to anchor it at the service.
    """
    md = find_aliyun_fc_url_md(start=start)
    urls = parse_fc_urls(md)
    if service_name not in urls:
        raise KeyError(
            f"service {service_name!r} not listed in {md}; "
            f"known services: {sorted(urls)}"
        )
    return urls[service_name]


__all__ = ["fc_url", "find_aliyun_fc_url_md", "parse_fc_urls"]
