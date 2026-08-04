from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from server.presign import Presigner
from server.storage import FileStorage, StorageBackend, make_storage


def test_prepare_upload_missing_returns_gateway_url(tmp_path):
    fs = FileStorage(tmp_path)
    r = fs.prepare_upload("alice", "job1", "x.pdb")
    assert r.exists is False
    assert r.put_url == "/v1/files/users/alice/job1/input/x.pdb"
    assert r.uri == f"file://{(tmp_path / 'users/alice/job1/input/x.pdb').resolve()}"


def test_prepare_upload_existing_skips(tmp_path):
    p = tmp_path / "users/alice/job1/input/x.pdb"
    p.parent.mkdir(parents=True)
    p.write_text("data")
    r = FileStorage(tmp_path).prepare_upload("alice", "job1", "x.pdb")
    assert r.exists is True and r.put_url is None


def test_result_url_hit_and_miss(tmp_path):
    fs = FileStorage(tmp_path)
    assert fs.result_url_if_exists("alice", "job1", "results.zip") is None
    out = tmp_path / "users/alice/job1/results.zip"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"z")
    assert fs.result_url_if_exists("alice", "job1", "results.zip") == \
        "/v1/files/users/alice/job1/results.zip"


@pytest.mark.parametrize("bad", ["../evil", "/etc/passwd", "users/a/../../../etc/passwd"])
def test_resolve_rejects_traversal(tmp_path, bad):
    with pytest.raises(HTTPException):
        FileStorage(tmp_path).resolve(bad)


def test_resolve_allows_legit_key(tmp_path):
    p = FileStorage(tmp_path).resolve("users/alice/j1/input/x.pdb")
    assert str(p).startswith(str(tmp_path.resolve()))


def test_backends_satisfy_protocol(tmp_path):
    assert isinstance(FileStorage(tmp_path), StorageBackend)
    assert isinstance(
        Presigner(client=MagicMock(), bucket="b", region="r", expiry_sec=900),
        StorageBackend,
    )


def test_make_storage_selects_file_and_rejects_unknown(tmp_path):
    s_file = SimpleNamespace(storage_backend="file", file_base_dir=tmp_path)
    assert isinstance(make_storage(s_file), FileStorage)
    with pytest.raises(ValueError):
        make_storage(SimpleNamespace(storage_backend="nope"))
