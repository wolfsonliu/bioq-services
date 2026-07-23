"""extract_error_summary + finalize_job."""

from __future__ import annotations

from pathlib import Path

from bioq_service import JobAdapter, ServiceSettings
from bioq_service.errors import extract_error_summary, finalize_job
from bioq_service.jobs import JobStore
from bioq_service.models import FailureKind, JobStatus


class _Adapter(JobAdapter):
    name = "test"


def _make(tmp_path: Path) -> tuple[JobStore, _Adapter]:
    settings = ServiceSettings(jobs_base_dir=tmp_path)
    adapter = _Adapter(settings=settings)
    store = JobStore()
    return store, adapter


def test_extract_returns_none_for_missing(tmp_path: Path) -> None:
    summary, tail = extract_error_summary(tmp_path / "missing.log")
    assert summary is None and tail is None


def test_extract_grabs_last_exception_line(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "Traceback (most recent call last):\n"
        '  File "foo.py", line 1, in <module>\n'
        "    raise ValueError('first')\n"
        "ValueError: first\n"
        "...later in the log, another error...\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: second\n"
    )
    summary, tail = extract_error_summary(log, tail_chars=200)
    assert summary == "RuntimeError: second"
    assert tail is not None and "RuntimeError: second" in tail


def test_extract_handles_dotted_exception_names(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("torch.cuda.OutOfMemoryError: CUDA out of memory at allocator\n")
    summary, _ = extract_error_summary(log)
    assert summary == "torch.cuda.OutOfMemoryError: CUDA out of memory at allocator"


def test_extract_falls_back_to_last_line_when_no_exception(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("just some output\nlast line wins\n")
    summary, _ = extract_error_summary(log)
    assert summary == "last line wins"


def test_finalize_completed(tmp_path: Path) -> None:
    store, adapter = _make(tmp_path)
    store.create("j1")
    out = adapter.output_dir(adapter.job_dir("j1"))
    out.mkdir(parents=True)
    (out / "result.txt").write_text("ok")

    finalize_job(store, adapter, "j1", rc=0, label="test")
    job = store.get("j1")
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.failure_kind is None
    assert job.error_summary is None
    assert job.output_count == 1
    assert job.output_total_bytes is not None and job.output_total_bytes > 0


def test_finalize_subprocess_error_attaches_summary(tmp_path: Path) -> None:
    store, adapter = _make(tmp_path)
    store.create("j2")
    log = adapter.log_path(adapter.job_dir("j2"))
    log.parent.mkdir(parents=True)
    log.write_text("ValueError: bad input\n")

    finalize_job(store, adapter, "j2", rc=1, label="test")
    job = store.get("j2")
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.failure_kind == FailureKind.SUBPROCESS_ERROR
    assert job.error_summary == "ValueError: bad input"
    assert job.error_tail is not None


def test_finalize_no_outputs_marks_distinct_failure_kind(tmp_path: Path) -> None:
    store, adapter = _make(tmp_path)
    store.create("j3")
    # rc=0 but output dir is empty
    adapter.output_dir(adapter.job_dir("j3")).mkdir(parents=True)

    finalize_job(store, adapter, "j3", rc=0, label="test")
    job = store.get("j3")
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.failure_kind == FailureKind.NO_OUTPUTS
    assert job.output_count is None
    assert job.output_total_bytes is None


def test_finalize_skips_when_job_missing(tmp_path: Path) -> None:
    store, adapter = _make(tmp_path)
    # Don't store anything; should not raise.
    finalize_job(store, adapter, "ghost", rc=0, label="test")
