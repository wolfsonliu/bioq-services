"""Bindcraft2Adapter：输出判定、子进程环境、manifest extras。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from server.adapter import Bindcraft2Adapter
from server.settings import Bindcraft2Settings


class _Off(Bindcraft2Settings):
    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_TEST_", env_file=None, extra="ignore", case_sensitive=False
    )


@pytest.fixture
def adapter(tmp_path: Path) -> Bindcraft2Adapter:
    s = _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "root",
        alphafold_params_dir=tmp_path / "af",
        compile_cache_dir=tmp_path / "cache",
    )
    return Bindcraft2Adapter(settings=s)


def _job(tmp_path: Path) -> Path:
    d = tmp_path / "jobs" / "j1"
    (d / "output").mkdir(parents=True)
    return d


def test_name(adapter):
    assert adapter.name == "bindcraft2"


def test_subprocess_cwd_is_upstream_root(adapter, tmp_path):
    assert adapter.subprocess_cwd() == tmp_path / "root"


def test_subprocess_env_sets_required_vars(adapter, tmp_path):
    env = adapter.subprocess_env()
    # 必须显式设置，否则上游会尝试下载 5.3 GB 参数。
    assert env["BINDCRAFT_AF2_PARAMS"] == str(tmp_path / "af")
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["BINDCRAFT_WORKERS_PER_GPU"] == "1"
    assert env["BINDCRAFT_MAX_WORKERS_PER_GPU"] == "1"


def test_subprocess_env_creates_compile_cache_dir(adapter, tmp_path):
    adapter.subprocess_env()
    assert (tmp_path / "cache").is_dir()


def test_subprocess_env_survives_unwritable_cache_dir(adapter, tmp_path):
    (tmp_path / "af").mkdir(exist_ok=True)
    # 把 cache 路径指到一个已存在的普通文件 → mkdir 必然失败，但不能抛。
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    adapter.settings.compile_cache_dir = blocker / "cache"
    env = adapter.subprocess_env()
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(blocker / "cache")


def test_detect_outputs_false_when_empty(adapter, tmp_path):
    assert adapter.detect_outputs(_job(tmp_path)) is False


def test_detect_outputs_true_on_ranked_csv(adapter, tmp_path):
    d = _job(tmp_path)
    p = d / "output" / "3_Ranked" / "!_Ranked.csv"
    p.parent.mkdir(parents=True)
    p.write_text("design,i_pDAE\n", encoding="utf-8")
    assert adapter.detect_outputs(d) is True


def test_detect_outputs_true_on_summary_only(adapter, tmp_path):
    """accepted=0 的合法 campaign 只写 summary.csv——不能被判为失败。"""
    d = _job(tmp_path)
    (d / "output" / "summary.csv").write_text("scope,metric\ncampaign,final\n", encoding="utf-8")
    assert adapter.detect_outputs(d) is True


def test_detect_outputs_true_on_rank_output(adapter, tmp_path):
    d = _job(tmp_path)
    (d / "output" / "ranked_by_i_pTM.csv").write_text("design,rank\n", encoding="utf-8")
    assert adapter.detect_outputs(d) is True


def test_detect_outputs_true_on_filter_output(adapter, tmp_path):
    d = _job(tmp_path)
    (d / "output" / "filtered.csv").write_text("design,outcome\n", encoding="utf-8")
    assert adapter.detect_outputs(d) is True


def test_detect_outputs_ignores_empty_files(adapter, tmp_path):
    d = _job(tmp_path)
    (d / "output" / "summary.csv").write_text("", encoding="utf-8")
    assert adapter.detect_outputs(d) is False


def test_infer_job_from_dir_reports_accepted_count(adapter, tmp_path):
    d = _job(tmp_path)
    p = d / "output" / "3_Ranked" / "!_Ranked.csv"
    p.parent.mkdir(parents=True)
    p.write_text("design,i_pDAE\na,0.7\nb,0.6\n", encoding="utf-8")
    info = adapter.infer_job_from_dir(d)
    assert info.status.value == "completed"
    assert info.progress == "2 accepted"


def test_infer_job_from_dir_zero_accepted(adapter, tmp_path):
    d = _job(tmp_path)
    (d / "output" / "summary.csv").write_text("scope,metric\n", encoding="utf-8")
    info = adapter.infer_job_from_dir(d)
    assert info.status.value == "completed"
    assert info.progress == "finished"


def test_infer_job_from_dir_no_outputs(adapter, tmp_path):
    info = adapter.infer_job_from_dir(_job(tmp_path))
    assert info.status.value == "failed"


def test_manifest_extras_shape(adapter):
    extras = adapter.manifest_extras()
    assert extras["tool_outputs"]["ranked"].endswith("3_Ranked/!_Ranked.csv")
    assert extras["tool_outputs"]["summary"] == "summary.csv"
    assert "job://<job_id>[/<subdir>]" in extras["input_uri_schemes"]
    assert extras["weights"]["alphafold_params_dir"]
    assert len(extras["weights"]["expected_alphafold_models"]) == 7


def test_endpoint_examples_cover_all_six(adapter):
    examples = adapter.endpoint_examples()
    for path in (
        "/api/design",
        "/api/tasks/design",
        "/api/rank",
        "/api/tasks/rank",
        "/api/filter",
        "/api/tasks/filter",
    ):
        assert path in examples, f"missing examples for {path}"
        assert examples[path], f"empty examples for {path}"
        assert any(e.curl for e in examples[path]), f"no curl for {path}"
