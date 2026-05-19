"""fc_testing — URL md parser + helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioagent_service.fc_testing import (
    fc_url,
    find_aliyun_fc_url_md,
    parse_fc_urls,
    poll_job,
)


def test_parse_fc_urls_strips_trailing_slash(tmp_path: Path) -> None:
    md = tmp_path / "aliyun_fc_url.md"
    md.write_text(
        "ppiflow-server: https://fc-ppiflow.example.com/\n"
        "genie3-server: https://fc-genie.example.com\n"
        "\n"
        "# a comment line\n"
        "noise without colon\n"
    )
    urls = parse_fc_urls(md)
    assert urls == {
        "ppiflow-server": "https://fc-ppiflow.example.com",
        "genie3-server": "https://fc-genie.example.com",
    }


def test_find_aliyun_fc_url_md_walks_up(tmp_path: Path) -> None:
    (tmp_path / "services").mkdir()
    target = tmp_path / "services" / "aliyun_fc_url.md"
    target.write_text("foo: https://example.com\n")
    nested = tmp_path / "services" / "fooserver" / "tests"
    nested.mkdir(parents=True)
    assert find_aliyun_fc_url_md(start=nested) == target.resolve()


def test_find_raises_when_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_aliyun_fc_url_md(start=tmp_path)


def test_fc_url_raises_on_unknown_service(tmp_path: Path) -> None:
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "aliyun_fc_url.md").write_text(
        "ppiflow-server: https://example.com\n"
    )
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

        def get(self, _url: str) -> FakeResp:
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
        def get(self, _url: str) -> FakeResp:
            return FakeResp()

    with pytest.raises(TimeoutError, match="did not finish"):
        poll_job(FakeClient(), "http://x", "abc", timeout_s=0, interval_s=0)
