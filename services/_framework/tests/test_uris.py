"""Shared URI-resolution helpers (bioagent_service.uris)."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic_settings import SettingsConfigDict

from bioagent_service.settings import ServiceSettings
from bioagent_service.uris import (
    maybe_resolve_input,
    resolve_input,
    resolve_uri,
    save_upload,
)


class _Settings(ServiceSettings):
    model_config = SettingsConfigDict(env_file=None, env_prefix="URIS_TEST_", extra="ignore")


@pytest.fixture
def settings(tmp_path: Path) -> _Settings:
    return _Settings(jobs_base_dir=tmp_path / "jobs")


def _fake_upload(data: bytes) -> SimpleNamespace:
    """Minimal UploadFile stand-in: only `.file.read` is used by save_upload."""
    return SimpleNamespace(file=io.BytesIO(data))


def test_save_upload_streams_bytes(tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "out.bin"
    save_upload(_fake_upload(b"hello"), dest)  # type: ignore[arg-type]
    assert dest.read_bytes() == b"hello"


def test_resolve_input_prefers_upload(settings: _Settings, tmp_path: Path) -> None:
    dest = tmp_path / "in.txt"
    resolve_input(_fake_upload(b"x"), None, dest, settings)  # type: ignore[arg-type]
    assert dest.read_bytes() == b"x"


def test_resolve_input_requires_something(settings: _Settings, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as ei:
        resolve_input(None, None, tmp_path / "d", settings)
    assert ei.value.status_code == 422


def test_maybe_resolve_input_returns_none(settings: _Settings, tmp_path: Path) -> None:
    assert maybe_resolve_input(None, None, tmp_path / "d", settings) is None


def test_resolve_input_field_name_in_422(settings: _Settings, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as ei:
        resolve_input(None, None, tmp_path / "d", settings, field_name="structure")
    assert ei.value.status_code == 422
    assert "structure" in ei.value.detail


def test_resolve_uri_job_scheme(settings: _Settings, tmp_path: Path) -> None:
    # Seed a prior job's output on the same NAS.
    src = settings.jobs_base_dir / "prev" / "output" / "result.pdb"
    src.parent.mkdir(parents=True)
    src.write_text("ATOM")
    dest = tmp_path / "in" / "result.pdb"
    resolve_uri("job://prev/result.pdb", dest, settings)
    assert dest.read_text() == "ATOM"


def test_resolve_uri_job_missing_file_404(settings: _Settings, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as ei:
        resolve_uri("job://prev/nope.pdb", tmp_path / "d", settings)
    assert ei.value.status_code == 404


def test_resolve_uri_job_malformed_422(settings: _Settings, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as ei:
        resolve_uri("job://noSlash", tmp_path / "d", settings)
    assert ei.value.status_code == 422


def test_resolve_uri_file_scheme_and_bare_path(settings: _Settings, tmp_path: Path) -> None:
    src = tmp_path / "local.txt"
    src.write_text("data")
    d1 = tmp_path / "out1.txt"
    resolve_uri(f"file://{src}", d1, settings)
    assert d1.read_text() == "data"
    d2 = tmp_path / "out2.txt"
    resolve_uri(str(src), d2, settings)  # bare absolute path
    assert d2.read_text() == "data"


def test_resolve_uri_file_missing_404(settings: _Settings, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as ei:
        resolve_uri(f"file://{tmp_path}/absent.txt", tmp_path / "d", settings)
    assert ei.value.status_code == 404


def test_resolve_uri_unsupported_scheme_422(settings: _Settings, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as ei:
        resolve_uri("ftp://host/file", tmp_path / "d", settings)
    assert ei.value.status_code == 422
