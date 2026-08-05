"""Tests for the config file generator (server.config_gen / config_spec)."""
from __future__ import annotations

import pytest
from server import config_gen as gen
from server import config_spec as spec
from server.settings import GatewaySettings


def _parse(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines, skipping comments/blanks (mimics env_file)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v
    return out


def test_curation_completeness():
    """Every schema field is either surfaced or explicitly omitted."""
    assert set(gen.surfaced_keys()) | spec.OMIT == gen.all_field_keys()


def test_docs_completeness():
    """Every surfaced field has a doc comment."""
    missing = [k for k in gen.surfaced_keys() if k not in spec.DOCS]
    assert not missing, f"surfaced fields without DOCS: {missing}"


def test_no_duplicate_surfaced_keys():
    keys = gen.surfaced_keys()
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("target", spec.TARGETS)
def test_no_secret_values_leak(target):
    """Secret fields appear only as commented placeholders, never as KEY=value."""
    parsed = _parse(gen.render_target(target))
    for key in spec.SECRETS:
        assert gen.env_name(key) not in parsed
    for name in spec.EXTERNAL_SECRETS.get(target, []):
        assert name not in parsed
    # db_url must never be emitted with an embedded password.
    if "GATEWAY_DB_URL" in parsed:
        assert "@" not in parsed["GATEWAY_DB_URL"]


@pytest.mark.parametrize("target", spec.TARGETS)
def test_roundtrip_matches_profile(target, monkeypatch):
    """Feeding the generated file back through env yields the profile values."""
    for k, v in _parse(gen.render_target(target)).items():
        monkeypatch.setenv(k, v)
    s = GatewaySettings()
    expected = dict(spec.COMMON)
    expected.update(spec.PROFILES[target])
    for key, want in expected.items():
        if key in spec.SECRETS:
            continue
        if key.startswith("auth."):
            got = getattr(s.auth, key[len("auth."):])
        else:
            got = getattr(s, key)
        # Compare through fmt() so Path/bool/int coerce like the env round-trip.
        assert gen.fmt(got) == gen.fmt(want), \
            f"{target}:{key} expected {want!r}, got {got!r}"


def test_ecs_bucket_is_bio_gateway():
    parsed = _parse(gen.render_target("ecs"))
    assert parsed["GATEWAY_OSS_BUCKET"] == "bio-gateway"


def test_ali_keys_use_bare_alias():
    """ali_access_key_* map to bare ALI_AK/ALI_SK (validation_alias), not GATEWAY_*."""
    assert gen.env_name("ali_access_key_id") == "ALI_AK"
    assert gen.env_name("ali_access_key_secret") == "ALI_SK"


@pytest.mark.parametrize("target", spec.TARGETS)
def test_render_is_deterministic(target):
    assert gen.render_target(target) == gen.render_target(target)


@pytest.mark.parametrize("target", spec.TARGETS)
def test_committed_file_matches_generator(target):
    """CI drift gate: the checked-in file must equal the generator output."""
    path = gen.target_path(target)
    assert path.read_text(encoding="utf-8") == gen.render_target(target), \
        f"{path} is stale — run `make gen-config`"
