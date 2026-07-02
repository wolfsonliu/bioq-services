"""Shared helpers for tests that hit the real Function Compute deployments.

These tests are **opt-in**: by default they're skipped to avoid hitting prod
endpoints during normal `pytest` runs. To enable, either:

  * `pytest -m fc ...`     — selects only fc-marked tests
  * `RUN_FC_TESTS=1 pytest` — runs everything, including fc-marked tests

Each service's `tests/conftest.py` delegates marker registration + skip logic
to this module via:

    from bioagent_service.fc_testing import (
        register_fc_marker,
        skip_fc_tests_unless_enabled,
    )

    def pytest_configure(config):
        register_fc_marker(config)

    def pytest_collection_modifyitems(config, items):
        skip_fc_tests_unless_enabled(config, items)

The `fc_url(service_name)` helper resolves a service's deployed URL from
`services/aliyun_fc_url.md` (the single source of truth — keep that file in
sync with FC console). `poll_job(...)` polls a job to terminal status.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

_URL_LINE_RE = re.compile(r"^([A-Za-z0-9_-]+):\s*(https?://\S+)\s*$")


def find_aliyun_fc_url_md(start: Path | None = None) -> Path:
    """Walk up from `start` (or cwd) until `services/aliyun_fc_url.md` is found.

    Each test file lives inside `services/<svc>/tests/`, so the walk is short
    (3 levels up). Raises FileNotFoundError if not found — typically means the
    test is being run from outside the bioagent repo.
    """
    base = (start or Path.cwd()).resolve()
    for candidate in [base, *base.parents]:
        target = candidate / "services" / "aliyun_fc_url.md"
        if target.is_file():
            return target
    raise FileNotFoundError(
        f"services/aliyun_fc_url.md not found above {base!r}"
    )


def parse_fc_urls(md_path: Path) -> dict[str, str]:
    """Parse `services/aliyun_fc_url.md`.

    Lines of the form `<service>: https://...` are extracted; comments and
    blank lines are skipped. Trailing slashes are stripped from URLs so
    callers can do `f"{url}/api/..."` without worrying about doubled `//`.
    """
    out: dict[str, str] = {}
    for raw in md_path.read_text(encoding="utf-8").splitlines():
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


class Retry429Transport(httpx.HTTPTransport):
    """httpx transport that retries HTTP 429 responses with linear backoff.

    Alibaba Cloud FC returns 429 when the account-level GPU quota for a
    specific card class (e.g. ``fc.gpu.tesla.1``) is exhausted by any
    running function in the account.  This affects every service whose
    Dockerfile requests that card, not just the one under test — quota
    hits during unrelated inference runs will surface as 429 on smoke
    calls (``/healthz``, ``/api/manifest``, ``/openapi.json``).  Wrapping
    the client with this transport lets tests wait out short bursts
    instead of failing spuriously.

    Only 429 is retried; other status codes pass through unchanged.
    """

    def __init__(
        self,
        *args: Any,
        max_retries: int = 10,
        backoff_s: float = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries
        self._backoff_s = backoff_s

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        resp = super().handle_request(request)
        for _ in range(self._max_retries):
            if resp.status_code != 429:
                return resp
            resp.read()
            resp.close()
            time.sleep(self._backoff_s)
            resp = super().handle_request(request)
        return resp


def make_retrying_client(
    base_url: str,
    *,
    timeout: httpx.Timeout | float = 120.0,
    max_retries: int = 10,
    backoff_s: float = 20.0,
    **client_kwargs: Any,
) -> httpx.Client:
    """Return an ``httpx.Client`` whose transport auto-retries HTTP 429.

    Use this in FC integration tests where account-level GPU quota
    exhaustion can produce 429 on any request — including cheap smoke
    calls.  See :class:`Retry429Transport` for context.
    """
    transport = Retry429Transport(max_retries=max_retries, backoff_s=backoff_s)
    return httpx.Client(
        base_url=base_url,
        timeout=timeout,
        transport=transport,
        **client_kwargs,
    )


def poll_job(
    client: Any,
    base_url: str,
    job_id: str,
    *,
    timeout_s: int = 1800,
    interval_s: int = 15,
    max_transient_errors: int = 10,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Poll `GET /api/jobs/{job_id}` until status is terminal.

    `client` is any object with a `.get(url)` method whose response has
    `.raise_for_status()` and `.json()` — httpx.Client and requests.Session
    both fit. Returns the final JobInfo dict; raises TimeoutError on deadline.

    ``extra_headers`` are forwarded on every poll request — use this to pass
    the FC session affinity header so polls hit the correct instance.

    Transient network errors (connection refused, DNS failures, FC cold-start
    hiccups) are retried up to `max_transient_errors` consecutive times before
    being re-raised. A successful poll resets the counter.
    """
    deadline = time.monotonic() + timeout_s
    body: dict[str, Any] = {}
    consecutive_errors = 0
    hdrs = extra_headers or {}
    while time.monotonic() < deadline:
        try:
            resp = client.get(f"{base_url}/api/jobs/{job_id}", headers=hdrs)
            resp.raise_for_status()
            body = resp.json()
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors >= max_transient_errors:
                raise
            elapsed = int(time.monotonic() - (deadline - timeout_s))
            print(
                f"  [poll_job] transient error ({consecutive_errors}/"
                f"{max_transient_errors}): {exc!r} (elapsed {elapsed}s)"
            )
            time.sleep(interval_s)
            continue
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(interval_s)
    last = body.get("status", "<unknown>")
    raise TimeoutError(
        f"job {job_id!r} did not finish within {timeout_s}s; last status: {last}"
    )


# ---------------------------------------------------------------------------
# Pytest hooks — delegate from each service's conftest.py
# ---------------------------------------------------------------------------


def register_fc_marker(config: Any) -> None:
    """Register the `fc` marker so `pytest -m fc` is recognized."""
    config.addinivalue_line(
        "markers",
        "fc: opt-in test that hits a real Function Compute deployment "
        "(runs only with `pytest -m fc` or `RUN_FC_TESTS=1`)",
    )


def skip_fc_tests_unless_enabled(config: Any, items: list[Any]) -> None:
    """Add an auto-skip to every fc-marked test unless explicitly enabled."""
    if os.environ.get("RUN_FC_TESTS"):
        return
    selected = config.getoption("-m", default="") or ""
    if "fc" in selected.split():
        return
    skip = pytest.mark.skip(
        reason="FC tests are opt-in: run with `pytest -m fc` or `RUN_FC_TESTS=1`"
    )
    for item in items:
        if "fc" in item.keywords:
            item.add_marker(skip)


__all__ = [
    "Retry429Transport",
    "fc_url",
    "find_aliyun_fc_url_md",
    "make_retrying_client",
    "parse_fc_urls",
    "poll_job",
    "register_fc_marker",
    "skip_fc_tests_unless_enabled",
]
