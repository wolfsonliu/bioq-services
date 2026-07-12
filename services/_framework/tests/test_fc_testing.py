"""service_registry (via fc_testing re-exports) — YAML loader + helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioagent_service.fc_testing import (
    fc_url,
    find_services_yaml,
    load_services,
    poll_job,
)


def test_load_services_parses_records_and_strips_slash(tmp_path: Path) -> None:
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "services.yaml").write_text(
        "services:\n"
        "  ppiflow-server:\n"
        "    url: https://fc-ppiflow.example.com/\n"
        "    tier: hot\n"
        "  genie3-server:\n"
        "    url: https://fc-genie.example.com\n",
        encoding="utf-8",
    )
    svcs = load_services(tmp_path / "services" / "services.yaml")
    assert svcs["ppiflow-server"].url == "https://fc-ppiflow.example.com"
    assert svcs["ppiflow-server"].tier == "hot"
    assert svcs["genie3-server"].url == "https://fc-genie.example.com"
    assert svcs["genie3-server"].region == "cn-hangzhou"  # default
    assert svcs["genie3-server"].tier == "warm"           # default


def test_find_services_yaml_walks_up(tmp_path: Path) -> None:
    (tmp_path / "services").mkdir()
    target = tmp_path / "services" / "services.yaml"
    target.write_text("services:\n  foo: {url: https://example.com}\n")
    nested = tmp_path / "services" / "fooserver" / "tests"
    nested.mkdir(parents=True)
    assert find_services_yaml(start=nested) == target.resolve()


def test_find_raises_when_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_services_yaml(start=tmp_path)


def test_fc_url_resolves_and_raises_on_unknown(tmp_path: Path) -> None:
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "services.yaml").write_text(
        "services:\n  ppiflow-server: {url: https://example.com}\n"
    )
    assert fc_url("ppiflow-server", start=tmp_path) == "https://example.com"
    with pytest.raises(KeyError, match="bogus-server"):
        fc_url("bogus-server", start=tmp_path)


def test_poll_job_returns_on_completed() -> None:
    """`poll_job` should return immediately on the first terminal status."""

    class FakeResp:
        def __init__(self, payload: dict) -> None:
            self._p = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._p

    class FakeClient:
        def __init__(self, payloads: list[dict]) -> None:
            self._p = list(payloads)
            self.calls = 0

        def get(self, _url: str, **kwargs) -> FakeResp:
            self.calls += 1
            return FakeResp(self._p.pop(0))

    client = FakeClient(
        [
            {"status": "running"},
            {"status": "running"},
            {"status": "completed", "job_id": "abc"},
        ]
    )
    out = poll_job(client, "http://x", "abc", timeout_s=10, interval_s=0)
    assert out == {"status": "completed", "job_id": "abc"}
    assert client.calls == 3


def test_poll_job_times_out() -> None:
    class FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"status": "running"}

    class FakeClient:
        def get(self, _url: str, **kwargs) -> FakeResp:
            return FakeResp()

    with pytest.raises(TimeoutError, match="did not finish"):
        poll_job(FakeClient(), "http://x", "abc", timeout_s=0, interval_s=0)


def test_poll_job_retries_transient_errors() -> None:
    """Transient network errors are retried and don't break polling."""

    class FakeResp:
        def __init__(self, payload: dict) -> None:
            self._p = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._p

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, _url: str, **kwargs) -> FakeResp:
            self.calls += 1
            if self.calls == 2:
                raise ConnectionError("No route to host")
            if self.calls >= 3:
                return FakeResp({"status": "completed", "job_id": "abc"})
            return FakeResp({"status": "running"})

    client = FakeClient()
    out = poll_job(client, "http://x", "abc", timeout_s=10, interval_s=0)
    assert out == {"status": "completed", "job_id": "abc"}
    assert client.calls == 3


def test_poll_job_raises_after_max_transient_errors() -> None:
    """Consecutive transient errors beyond limit are re-raised."""

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, _url: str, **kwargs):
            self.calls += 1
            raise ConnectionError("No route to host")

    client = FakeClient()
    with pytest.raises(ConnectionError, match="No route to host"):
        poll_job(
            client, "http://x", "abc",
            timeout_s=10, interval_s=0, max_transient_errors=3,
        )
    assert client.calls == 3
