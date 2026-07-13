from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from bioagent_service import JobAdapter, JobStatus, ServiceSettings
from bioagent_service.jobs import JobStore
from bioagent_service.task_endpoint import execute_task


class _Settings(ServiceSettings):
    model_config = SettingsConfigDict(env_prefix="TOSS_", extra="ignore")


class _Adapter(JobAdapter):
    name = "t"


class _Params(BaseModel):
    x: int = 1


def _request(tmp_path: Path, headers: dict | None = None):
    settings = _Settings(
        jobs_base_dir=tmp_path / "jobs",
        uploads_base_dir=tmp_path / "uploads",
        oss_output_mount=str(tmp_path / "mnt"),
    )
    (tmp_path / "mnt").mkdir()
    adapter = _Adapter(settings=settings)
    store = JobStore(persist_dir=settings.jobs_base_dir)
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        job_store=store, adapter=adapter, settings=settings)))
    req.headers = headers or {}
    return req


def _build_argv(_p, _job_id, job_dir):
    return ["sh", "-c", f"echo hi > {job_dir}/output/r.txt"]


def _build_argv_fail(_p, _job_id, job_dir):
    return ["sh", "-c", "echo boom >&2; exit 1"]  # no output + non-zero => FAILED


def test_execute_task_mirrors_to_oss_on_completion(tmp_path):
    req = _request(tmp_path)
    info = execute_task(
        req, job_id="j1", label="t", params=_Params(),
        build_argv=_build_argv, oss_prefix="users/alice/j1/",
    )
    assert info.status == JobStatus.COMPLETED
    d = tmp_path / "mnt" / "users" / "alice" / "j1"
    assert (d / "output" / "r.txt").exists()
    assert (d / "results.zip").exists()
    assert "r.txt" in zipfile.ZipFile(d / "results.zip").namelist()


def test_execute_task_no_mirror_without_prefix(tmp_path):
    req = _request(tmp_path)
    info = execute_task(
        req, job_id="j2", label="t", params=_Params(), build_argv=_build_argv,
    )
    assert info.status == JobStatus.COMPLETED
    assert not (tmp_path / "mnt" / "users").exists()  # no prefix => no mirror


def test_execute_task_mirrors_on_failure(tmp_path):
    req = _request(tmp_path)
    info = execute_task(
        req, job_id="j3", label="t", params=_Params(),
        build_argv=_build_argv_fail, oss_prefix="users/alice/j3/",
    )
    assert info.status == JobStatus.FAILED
    d = tmp_path / "mnt" / "users" / "alice" / "j3"
    assert (d / "logs" / "run.log").exists()   # logs mirrored even on failure
    assert not (d / "results.zip").exists()    # no output/ => no results.zip


def test_execute_task_mirrors_from_request_header_when_prefix_not_passed(tmp_path):
    # Bespoke per-service handlers call execute_task WITHOUT oss_prefix; the
    # prefix arrives as the X-Bioagent-Oss-Prefix header on the request.
    req = _request(tmp_path, headers={"X-Bioagent-Oss-Prefix": "users/alice/jh/"})
    info = execute_task(
        req, job_id="jh", label="t", params=_Params(), build_argv=_build_argv,
    )  # note: no oss_prefix kwarg
    assert info.status == JobStatus.COMPLETED
    d = tmp_path / "mnt" / "users" / "alice" / "jh"
    assert (d / "output" / "r.txt").exists()
    assert (d / "results.zip").exists()
