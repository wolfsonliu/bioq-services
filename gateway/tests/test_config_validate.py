"""Tests for config validation (server.config_validate) + app startup fail-fast."""
from __future__ import annotations

import os
from unittest import mock

import pytest
from server import config_gen as gen
from server import config_spec as spec
from server.config_validate import validate_settings
from server.settings import GatewaySettings


def _settings(**env):
    """Build a GatewaySettings from an isolated env mapping (no ambient leakage)."""
    with mock.patch.dict(os.environ, env, clear=True):
        return GatewaySettings()


def test_generated_profiles_are_clean(monkeypatch):
    """Every generated per-target file passes validation (with dummy secrets)."""
    for target in spec.TARGETS:
        for k, v in _parse(gen.render_target(target)).items():
            monkeypatch.setenv(k, v)
        # supply the secrets the profile needs so we test topology, not secrets
        monkeypatch.setenv("GATEWAY_AUTH__OIDC_CLIENT_SECRET", "x")
        s = GatewaySettings()
        fatals, _ = validate_settings(s)
        assert fatals == [], f"{target}: {fatals}"
        for k in _parse(gen.render_target(target)):
            monkeypatch.delenv(k, raising=False)


def _parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_fc_endpoint_with_equals_is_fatal():
    s = _settings(GATEWAY_FC_ENDPOINT="gateway_fc_endpoint=1713.cn-hangzhou-internal.fc.aliyuncs.com")
    fatals, _ = validate_settings(s)
    assert any("FC_ENDPOINT" in f for f in fatals)


def test_fc_endpoint_bare_host_ok():
    s = _settings(GATEWAY_FC_ENDPOINT="1713.cn-hangzhou-internal.fc.aliyuncs.com")
    fatals, _ = validate_settings(s)
    assert not any("FC_ENDPOINT" in f for f in fatals)


def test_fc_endpoint_with_scheme_is_fatal():
    s = _settings(GATEWAY_FC_ENDPOINT="https://1713.fc.aliyuncs.com")
    fatals, _ = validate_settings(s)
    assert any("FC_ENDPOINT" in f for f in fatals)


def test_oss_without_bucket_is_fatal():
    s = _settings(GATEWAY_STORAGE_BACKEND="oss", GATEWAY_OSS_BUCKET="")
    fatals, _ = validate_settings(s)
    assert any("OSS_BUCKET" in f for f in fatals)


def test_oss_placeholder_bucket_is_fatal():
    s = _settings(GATEWAY_STORAGE_BACKEND="oss", GATEWAY_OSS_BUCKET="<bucket-name>")
    fatals, _ = validate_settings(s)
    assert any("OSS_BUCKET" in f for f in fatals)


def test_oss_real_bucket_ok():
    s = _settings(GATEWAY_STORAGE_BACKEND="oss", GATEWAY_OSS_BUCKET="bio-gateway")
    fatals, _ = validate_settings(s)
    assert not any("OSS_BUCKET" in f for f in fatals)


def test_bypass_false_without_jwks_is_fatal():
    s = _settings(GATEWAY_AUTH__BYPASS_VPC="false", GATEWAY_AUTH__JWT_JWKS_URL="",
                  GATEWAY_STORAGE_BACKEND="file")
    fatals, _ = validate_settings(s)
    assert any("BYPASS_VPC" in f for f in fatals)


def test_openfaas_without_url_is_fatal():
    s = _settings(GATEWAY_DISPATCH_BACKEND="openfaas", GATEWAY_OPENFAAS_GATEWAY_URL="",
                  GATEWAY_STORAGE_BACKEND="file")
    fatals, _ = validate_settings(s)
    assert any("OPENFAAS_GATEWAY_URL" in f for f in fatals)


def test_bad_jwks_url_is_fatal():
    s = _settings(GATEWAY_AUTH__JWT_JWKS_URL="keycloak:8080/certs",
                  GATEWAY_STORAGE_BACKEND="file")
    fatals, _ = validate_settings(s)
    assert any("JWT_JWKS_URL" in f for f in fatals)


def test_bad_enum_is_fatal():
    s = _settings(GATEWAY_DISPATCH_BACKEND="lambda", GATEWAY_STORAGE_BACKEND="s3")
    fatals, _ = validate_settings(s)
    assert any("DISPATCH_BACKEND" in f for f in fatals)
    assert any("STORAGE_BACKEND" in f for f in fatals)


def test_fc_missing_creds_is_warning_not_fatal():
    s = _settings(GATEWAY_DISPATCH_BACKEND="fc", GATEWAY_STORAGE_BACKEND="file")
    fatals, warnings = validate_settings(s)
    assert not any("ALI_AK" in f for f in fatals)
    assert any("ALI_AK" in w for w in warnings)


def test_app_startup_raises_on_bad_config(monkeypatch, tmp_path):
    """Importing the app with fatal misconfig aborts (SystemExit)."""
    monkeypatch.setenv("GATEWAY_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("GATEWAY_UPLOADS_BASE_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("GATEWAY_DB_URL", f"sqlite:///{tmp_path/'gw.db'}")
    monkeypatch.setenv("GATEWAY_REGISTRY_PATH", str(tmp_path / "services.yaml"))
    (tmp_path / "services.yaml").write_text("services: {}\n", encoding="utf-8")
    import importlib

    import server.app as appmod
    importlib.reload(appmod)  # clean baseline (valid env)
    monkeypatch.setenv("GATEWAY_FC_ENDPOINT", "gateway_fc_endpoint=bad")
    with pytest.raises(SystemExit):
        importlib.reload(appmod)
    # reload once more with a clean endpoint so other tests get a valid module.
    monkeypatch.delenv("GATEWAY_FC_ENDPOINT", raising=False)
    importlib.reload(appmod)
