from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

from server.presign import Presigner


def test_key_derivation_per_user():
    p = Presigner(client=MagicMock(), bucket="b", region="cn-hangzhou", expiry_sec=900)
    key = p.input_key("alice", "solvated.rst7", "abc123")
    assert key == "users/alice/inputs/abc123/solvated.rst7"


def test_presign_skips_when_object_exists():
    client = MagicMock()
    client.head_object.return_value = MagicMock()  # no exception => exists
    p = Presigner(client=client, bucket="b", region="cn-hangzhou", expiry_sec=900)
    resp = p.presign_put("alice", "f.dat", "sha1")
    assert resp.exists is True
    assert resp.url is None
    assert resp.uri == "oss://b/users/alice/inputs/sha1/f.dat"
    client.presign.assert_not_called()


def test_presign_returns_url_when_missing():
    client = MagicMock()
    client.head_object.side_effect = Exception("NoSuchKey")
    client.presign.return_value = MagicMock(url="https://oss.put/signed")
    p = Presigner(client=client, bucket="b", region="cn-hangzhou", expiry_sec=900)
    resp = p.presign_put("alice", "f.dat", "sha1")
    assert resp.exists is False
    assert resp.url == "https://oss.put/signed"
    assert resp.uri == "oss://b/users/alice/inputs/sha1/f.dat"
    # Must pass `expires` (a timedelta duration), NOT `expiration` (a datetime) —
    # the OSS v2 signer raises on a timedelta expiration.
    kwargs = client.presign.call_args.kwargs
    assert isinstance(kwargs.get("expires"), timedelta)
    assert "expiration" not in kwargs
