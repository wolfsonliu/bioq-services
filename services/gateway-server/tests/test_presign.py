from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from server.presign import Presigner


def _oss_error(status_code: int, code: str) -> Exception:
    """A minimal stand-in for an alibabacloud-oss-v2 ServiceError: `_exists`
    inspects `.status_code` / `.code` to tell an absent object (404/NoSuchKey)
    from a real OSS outage (auth/5xx/transport)."""
    exc = Exception(code)
    exc.status_code = status_code
    exc.code = code
    return exc


def _oss_not_found() -> Exception:
    return _oss_error(404, "NoSuchKey")


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
    client.head_object.side_effect = _oss_not_found()
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
    client.head_object.side_effect = _oss_not_found()
    assert p.presign_get_if_exists("alice", "job1", "results.zip") is None


def test_exists_propagates_oss_outage():
    # A non-404 error (auth/5xx) must NOT be mistaken for a missing object —
    # it propagates so an OSS outage can't silently masquerade as "not there".
    client = MagicMock()
    client.head_object.side_effect = _oss_error(403, "AccessDenied")
    p = Presigner(client=client, bucket="b", region="cn-hangzhou", expiry_sec=900)
    with pytest.raises(Exception, match="AccessDenied"):
        p.presign_put("alice", "job1", "x.pdb")
    client.presign.assert_not_called()
