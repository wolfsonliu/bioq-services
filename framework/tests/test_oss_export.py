from __future__ import annotations

import zipfile
from pathlib import Path

from bioq_service.oss_export import mirror_job_dir_to_oss


def _make_job(tmp_path: Path) -> Path:
    job = tmp_path / "jobs" / "job1"
    (job / "output").mkdir(parents=True)
    (job / "logs").mkdir()
    (job / "input").mkdir()
    (job / "intermediates").mkdir()
    (job / "output" / "seqs.fa").write_text(">a\nMKQ\n")
    (job / "logs" / "run.log").write_text("log")
    (job / "input" / "x.pdb").write_text("PDB")
    (job / "job.json").write_text("{}")
    return job


def test_mirror_copies_job_dir_and_zips_output(tmp_path):
    job = _make_job(tmp_path)
    mount = tmp_path / "mnt"
    mount.mkdir()
    dest = mirror_job_dir_to_oss(
        job_dir=job, output_dir=job / "output",
        oss_prefix="users/alice/job1/", mount=str(mount),
    )
    d = mount / "users" / "alice" / "job1"
    assert Path(dest) == d
    assert (d / "output" / "seqs.fa").exists()
    assert (d / "logs" / "run.log").exists()
    assert (d / "job.json").exists()
    assert not (d / "input").exists()          # input/ skipped (client already uploaded)
    assert (d / "results.zip").exists()
    assert "seqs.fa" in zipfile.ZipFile(d / "results.zip").namelist()


def test_mirror_noop_without_mount(tmp_path):
    job = _make_job(tmp_path)
    assert mirror_job_dir_to_oss(
        job_dir=job, output_dir=job / "output",
        oss_prefix="users/a/job1/", mount=str(tmp_path / "nope"),
    ) is None


def test_mirror_noop_without_prefix(tmp_path):
    job = _make_job(tmp_path)
    mount = tmp_path / "mnt"
    mount.mkdir()
    assert mirror_job_dir_to_oss(
        job_dir=job, output_dir=job / "output", oss_prefix="", mount=str(mount),
    ) is None
