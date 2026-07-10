from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from server.auth.api_key import hash_secret
from server.auth.deps import require_auth
from server.settings import AuthSettings


def _req(*, auth=None, headers=None, db=None):
    r = MagicMock()
    r.app.state.settings.auth = auth or AuthSettings()
    r.app.state.db = db
    r.headers = headers or {}
    return r


def test_vpc_bypass():
    r = _req(headers={"host": "fc-gateway-x.cn-hangzhou-vpc.fcapp.run"})
    ident = require_auth(r)
    assert ident.method == "vpc_bypass"
    assert ident.principal == "internal_vpc"


def test_api_key_success():
    db = MagicMock()
    key_row = MagicMock()
    key_row.key_id = "gk_1"
    key_row.principal = "alice"
    db.find_api_key.return_value = key_row
    r = _req(auth=AuthSettings(bypass_vpc=False), db=db,
             headers={"x-api-key": "s3cr3t", "host": "public.example.com"})
    ident = require_auth(r)
    assert ident.method == "api_key"
    assert ident.principal == "alice"
    db.find_api_key.assert_called_once_with(hash_secret("s3cr3t"))


def test_no_creds_401():
    db = MagicMock()
    db.find_api_key.return_value = None
    r = _req(auth=AuthSettings(bypass_vpc=False), db=db,
             headers={"host": "public.example.com"})
    with pytest.raises(HTTPException) as e:
        require_auth(r)
    assert e.value.status_code == 401
