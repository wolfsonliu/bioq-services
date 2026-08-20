"""Tests for bioq_service.fields.default_semantics."""

from bioq_service import default_semantics


def test_default_semantics_returns_bioq_default_marker() -> None:
    marker = default_semantics("auto", "auto-select CUDA if available")
    assert marker == {"bioq_default": {"kind": "auto", "note": "auto-select CUDA if available"}}


def test_default_semantics_unset() -> None:
    assert default_semantics("unset", "only used when explicitly provided") == {
        "bioq_default": {"kind": "unset", "note": "only used when explicitly provided"}
    }
