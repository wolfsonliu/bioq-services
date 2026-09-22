"""campaign 目录 URI 解析：零拷贝 + 目录穿越防护。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic_settings import SettingsConfigDict

from server.campaigns import resolve_campaign_dir
from server.settings import Bindcraft2Settings


class _Off(Bindcraft2Settings):
    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_TEST_", env_file=None, extra="ignore", case_sensitive=False
    )


@pytest.fixture
def settings(tmp_path: Path) -> _Off:
    return _Off(jobs_base_dir=tmp_path / "jobs")


def _make_campaign(settings, job_id: str) -> Path:
    out = settings.jobs_base_dir / job_id / "output"
    (out / "3_Ranked").mkdir(parents=True)
    (out / "3_Ranked" / "!_Ranked.csv").write_text("design,i_pDAE\n", encoding="utf-8")
    return out


def test_job_uri_resolves_to_output_dir(settings):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir("job://abc123", settings) == out


def test_job_uri_with_redundant_output_suffix(settings):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir("job://abc123/output", settings) == out


def test_job_uri_with_subdirectory(settings):
    out = _make_campaign(settings, "abc123")
    sub = out / "3_Ranked"
    assert resolve_campaign_dir("job://abc123/3_Ranked", settings) == sub


def test_job_uri_is_not_copied(settings):
    """零拷贝：返回的必须是 NAS 上的原路径，不是副本。"""
    out = _make_campaign(settings, "abc123")
    resolved = resolve_campaign_dir("job://abc123", settings)
    assert resolved == out
    assert resolved.is_dir()


def test_job_uri_accepts_redundant_output_suffix_with_subdir(settings):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir("job://abc123/output/3_Ranked", settings) == out / "3_Ranked"


def test_job_uri_rejects_dotdot_job_id(settings):
    """`job://..` 会让 root 落到 <jobs_base_dir>/../output——必须在拼接前拒掉。"""
    _make_campaign(settings, "abc123")
    for uri in ("job://..", "job://../secret", "job://."):
        with pytest.raises(HTTPException) as exc:
            resolve_campaign_dir(uri, settings)
        assert exc.value.status_code == 422, uri


def test_job_uri_does_not_strip_output_lookalike(settings):
    """只认整段 `output`；`output2` 不能被当作冗余前缀剥掉。"""
    _make_campaign(settings, "abc123")
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("job://abc123/output2", settings)
    assert exc.value.status_code == 404


def test_job_uri_rejects_symlinked_output_dir(settings):
    """`output/` 是指向树外的符号链接时必须 422。

    这条钉住的是 root 相对 jobs_base_dir 的包含性检查——少了它，`.resolve()`
    会跟随符号链接把 root 定到树外，`job://abc123` 就会直接返回外部目录。
    """
    outside = settings.jobs_base_dir.parent / "outside"
    outside.mkdir()
    job = settings.jobs_base_dir / "abc123"
    job.mkdir(parents=True)
    (job / "output").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("job://abc123", settings)
    assert exc.value.status_code == 422


def test_job_uri_rejects_traversal(settings):
    _make_campaign(settings, "abc123")
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("job://abc123/../../../etc", settings)
    assert exc.value.status_code == 422


def test_absolute_path_passthrough(settings, tmp_path: Path):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir(str(out), settings) == out


def test_file_uri_passthrough(settings):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir(f"file://{out}", settings) == out


def test_missing_directory_is_404(settings):
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("job://nope", settings)
    assert exc.value.status_code == 404


def test_unsupported_scheme_is_422(settings):
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("oss://bucket/key", settings)
    assert exc.value.status_code == 422


def test_empty_uri_is_422(settings):
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("   ", settings)
    assert exc.value.status_code == 422
