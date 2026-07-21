"""`read_version_file` — sibling-VERSION lookup with `v` prefix stripping."""

from __future__ import annotations

from pathlib import Path

from bioq_service import read_version_file


def test_reads_sibling_version(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("v0.0.3\n")
    fake_app = tmp_path / "app.py"
    fake_app.write_text("")
    assert read_version_file(str(fake_app)) == "0.0.3"


def test_strips_only_leading_v(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("0.0.3\n")
    fake_app = tmp_path / "app.py"
    fake_app.write_text("")
    assert read_version_file(str(fake_app)) == "0.0.3"


def test_falls_back_when_missing(tmp_path: Path) -> None:
    fake_app = tmp_path / "app.py"
    fake_app.write_text("")
    assert read_version_file(str(fake_app), default="9.9.9") == "9.9.9"


def test_strips_surrounding_whitespace(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("  v1.2.3  \n")
    fake_app = tmp_path / "app.py"
    fake_app.write_text("")
    assert read_version_file(str(fake_app)) == "1.2.3"
