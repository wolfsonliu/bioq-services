from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

from server.presign import Presigner


def test_input_key_is_job_centric():
    p = Presigner(client=MagicMock(), bucket="b", region="cn-hangzhou", expiry_sec=900)
    assert p.input_key("alice", "job1", "x.pdb") == "users/alice/job1/input/x.pdb"


def test_output_key_is_job_centric():
    p = Presigner(client=MagicMock(), bucket="b", region="cn-hangzhou", expiry_sec=900)
    assert p.output_key("alice", "job1", "results.zip") == "users/alice/job1/results.zip"


def test_presign_put_skips_when_exists():
    client = MagicMock()
    client.head_object.return_value = MagicMock()  # exists
    p = Presigner(client=client, bucket="b", region="cn-hangzhou", expiry_sec=900)
    resp = p.presign_put("alice", "job1", "x.pdb")
    assert resp.exists is True and resp.url is None
    assert resp.uri == "oss://b/users/alice/job1/input/x.pdb"
    client.presign.assert_not_called()


def test_presign_put_returns_url_when_missing():
    client = MagicMock()
    client.head_object.side_effect = Exception("NoSuchKey")
    client.presign.return_value = MagicMock(url="https://oss.put/signed")
    p = Presigner(client=client, bucket="b", region="cn-hangzhou", expiry_sec=900)
    resp = p.presign_put("alice", "job1", "x.pdb")
    assert resp.exists is False and resp.url == "https://oss.put/signed"
    assert resp.uri == "oss://b/users/alice/job1/input/x.pdb"
    assert isinstance(client.presign.call_args.kwargs.get("expires"), timedelta)


def test_presign_get_if_exists_hit_and_miss():
    client = MagicMock()
    client.presign.return_value = MagicMock(url="https://oss.get/signed")
    p = Presigner(client=client, bucket="b", region="cn-hangzhou", expiry_sec=900)
    client.head_object.side_effect = None
    client.head_object.return_value = MagicMock()
    assert p.presign_get_if_exists("alice", "job1", "results.zip") == "https://oss.get/signed"
    client.head_object.side_effect = Exception("NoSuchKey")
    assert p.presign_get_if_exists("alice", "job1", "results.zip") is None
