"""Auth dependency unit tests."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from server.auth.api_key import verify_api_key
from server.auth.deps import require_api_key
from server.settings import APIKeyConfig


def _allowlist(secret: str) -> list[APIKeyConfig]:
    return [APIKeyConfig(
        key_id="ek_test",
        secret_hash=hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        customer_id="cust1",
    )]


# ---------------------------------------------------------------------------
# verify_api_key
# ---------------------------------------------------------------------------

def test_verify_returns_entry_on_match():
    allowlist = _allowlist("right_secret")
    matched = verify_api_key("right_secret", allowlist)
    assert matched is not None
    assert matched.customer_id == "cust1"


def test_verify_returns_none_on_wrong_secret():
    allowlist = _allowlist("right_secret")
    assert verify_api_key("wrong", allowlist) is None


def test_verify_returns_none_on_empty_secret():
    allowlist = _allowlist("right_secret")
    assert verify_api_key("", allowlist) is None


def test_verify_returns_none_on_empty_allowlist():
    assert verify_api_key("any", []) is None


# ---------------------------------------------------------------------------
# require_api_key
# ---------------------------------------------------------------------------

def _fake_request(api_keys: list[APIKeyConfig]) -> MagicMock:
    """Build a fake FastAPI Request whose app.state has the allowlist."""
    request = MagicMock()
    request.app.state.settings.api_keys = api_keys
    return request


def test_require_api_key_returns_key_config_on_match():
    allowlist = _allowlist("good")
    request = _fake_request(allowlist)
    key = require_api_key(request, x_api_key="good")
    assert key.customer_id == "cust1"


def test_require_api_key_raises_401_on_wrong_secret():
    allowlist = _allowlist("good")
    request = _fake_request(allowlist)
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(request, x_api_key="bad")
    assert exc_info.value.status_code == 401


def test_require_api_key_raises_401_on_empty():
    allowlist = _allowlist("good")
    request = _fake_request(allowlist)
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(request, x_api_key="")
    assert exc_info.value.status_code == 401
