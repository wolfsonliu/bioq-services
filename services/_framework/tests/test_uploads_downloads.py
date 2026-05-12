"""Zip upload / extraction / packaging with path-traversal safeguards."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from bioagent_service.downloads import archive_dir, list_files, safe_subpath
from bioagent_service.uploads import extract_dataset, save_upload


def _make_zip(tmp_path: Path, name: str, members: dict[str, bytes]) -> Path:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        for member_name, data in members.items():
            zf.writestr(member_name, data)
    return p


def test_save_upload_streams_to_disk(tmp_path: Path) -> None:
    buf = io.BytesIO(b"abc" * 1000)
    dest = tmp_path / "out" / "f.bin"
    saved = save_upload(buf, dest)
    assert saved.read_bytes() == b"abc" * 1000


def test_extract_dataset_normal_zip(tmp_path: Path) -> None:
    zip_path = _make_zip(
        tmp_path, "ok.zip", {"data/a.txt": b"hi", "data/sub/b.txt": b"bye"}
    )
    out = extract_dataset(zip_path, tmp_path / "ext")
    assert (out / "data" / "a.txt").read_bytes() == b"hi"
    assert (out / "data" / "sub" / "b.txt").read_bytes() == b"bye"


def test_extract_dataset_rejects_absolute_path(tmp_path: Path) -> None:
    zip_path = _make_zip(tmp_path, "bad.zip", {"/etc/passwd": b"x"})
    with pytest.raises(ValueError, match="absolute path"):
        extract_dataset(zip_path, tmp_path / "ext")


def test_extract_dataset_rejects_parent_traversal(tmp_path: Path) -> None:
    zip_path = _make_zip(tmp_path, "bad.zip", {"../escape.txt": b"x"})
    with pytest.raises(ValueError, match="parent traversal"):
        extract_dataset(zip_path, tmp_path / "ext")


def test_extract_dataset_bad_zip(tmp_path: Path) -> None:
    p = tmp_path / "notazip.zip"
    p.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        extract_dataset(p, tmp_path / "ext")


def test_archive_dir_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"A")
    (src / "sub").mkdir()
    (src / "sub" / "b.txt").write_bytes(b"B")

    buf = archive_dir(src)
    with zipfile.ZipFile(buf) as zf:
        assert sorted(zf.namelist()) == ["a.txt", "sub/b.txt"]
        assert zf.read("sub/b.txt") == b"B"


def test_archive_dir_missing_returns_empty_zip(tmp_path: Path) -> None:
    buf = archive_dir(tmp_path / "absent")
    with zipfile.ZipFile(buf) as zf:
        assert zf.namelist() == []


def test_list_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")
    (src / "sub").mkdir()
    (src / "sub" / "b.txt").write_text("b")
    assert list_files(src) == ["a.txt", "sub/b.txt"]
    assert list_files(tmp_path / "absent") == []


def test_safe_subpath_allows_nested(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "child.txt").write_text("c")
    resolved = safe_subpath(base, "child.txt")
    assert resolved == (base / "child.txt").resolve()


def test_safe_subpath_blocks_traversal(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError, match="path traversal"):
        safe_subpath(base, "../outside")
