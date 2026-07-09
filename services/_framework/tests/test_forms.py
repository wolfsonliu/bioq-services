"""Tests for ``bioagent_service.forms.model_form_depends``.

Verifies the workaround for FastAPI's ``Annotated[Model, Form()]`` limitation
when the route also accepts ``UploadFile`` parameters: with the helper, every
pydantic field becomes a flat form field, the OpenAPI schema reflects that,
``model_validator`` cross-field checks still run, and complex (dict/list)
fields are accepted as JSON strings.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, model_validator

from bioagent_service import model_form_depends


# ---------------------------------------------------------------------------
# Models exercised across the suite
# ---------------------------------------------------------------------------


class SimpleModel(BaseModel):
    name: str = Field(default="run")
    temp: float = Field(default=0.1, ge=0.0, le=1.0)
    flag: bool = False


class WithValidator(BaseModel):
    variant: Literal["a", "b"] = "a"
    label: str = "x"

    @model_validator(mode="after")
    def _check(self) -> "WithValidator":
        if self.variant == "b" and self.label == "x":
            raise ValueError("label must differ from default when variant=b")
        return self


class WithComplex(BaseModel):
    name: str = "run"
    bias: Optional[dict[str, float]] = None
    lengths: Optional[list[int]] = None


class RequiredFields(BaseModel):
    target_chain: str = Field(..., min_length=1)
    motif_names: list[str] = Field(..., min_length=1)


class WithBareContainers(BaseModel):
    """Bare, unparameterized ``dict``/``list`` annotations (no ``get_origin()``)."""

    name: str = "run"
    bias: Optional[dict] = None
    lengths: Optional[list] = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_app(model_cls) -> TestClient:
    """Build a tiny app that mirrors real services: form model + optional file."""
    app = FastAPI()

    @app.post("/x")
    def handler(
        params: model_cls = Depends(model_form_depends(model_cls)),
        pdb: Optional[UploadFile] = File(None),
    ):
        return {"params": params.model_dump(), "pdb": pdb.filename if pdb else None}

    return TestClient(app)


def test_openapi_schema_is_flat():
    """OpenAPI body should expose model fields directly, NOT a $ref to the model."""
    client = _make_app(SimpleModel)
    spec = client.app.openapi()
    body = next(iter(spec["paths"]["/x"]["post"]["requestBody"]["content"].values()))
    body_schema_ref = body["schema"]["$ref"]
    body_schema = spec["components"]["schemas"][body_schema_ref.rsplit("/", 1)[-1]]
    props = body_schema["properties"]
    # All model fields must be at the top level of the body schema.
    assert "name" in props
    assert "temp" in props
    assert "flag" in props
    # The wrapper-only `params` name must NOT appear.
    assert "params" not in props


def test_defaults_applied_when_omitted():
    client = _make_app(SimpleModel)
    r = client.post("/x", data={})
    assert r.status_code == 200, r.text
    assert r.json()["params"] == {"name": "run", "temp": 0.1, "flag": False}


def test_form_values_override_defaults():
    client = _make_app(SimpleModel)
    r = client.post("/x", data={"name": "abc", "temp": "0.5", "flag": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["params"] == {"name": "abc", "temp": 0.5, "flag": True}


def test_type_coercion_failure_returns_422():
    client = _make_app(SimpleModel)
    r = client.post("/x", data={"temp": "not-a-number"})
    assert r.status_code == 422


def test_constraint_violation_returns_422():
    client = _make_app(SimpleModel)
    r = client.post("/x", data={"temp": "5"})  # ge/le bound
    assert r.status_code == 422


def test_model_validator_fires():
    """Cross-field `model_validator(mode='after')` must still run on the assembled object."""
    client = _make_app(WithValidator)
    r = client.post("/x", data={"variant": "b", "label": "x"})
    assert r.status_code == 422
    # Sanity: legal combination still passes.
    r2 = client.post("/x", data={"variant": "b", "label": "y"})
    assert r2.status_code == 200


def test_complex_dict_field_accepts_json_string():
    client = _make_app(WithComplex)
    r = client.post("/x", data={"bias": '{"D": 1.5, "E": 1.5}'})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["bias"] == {"D": 1.5, "E": 1.5}


def test_complex_list_field_accepts_json_string():
    client = _make_app(WithComplex)
    r = client.post("/x", data={"lengths": "[80, 100, 120]"})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["lengths"] == [80, 100, 120]


def test_bare_dict_field_accepts_json_string():
    """Bare ``dict`` (no type args) must be treated as complex → decoded from JSON."""
    client = _make_app(WithBareContainers)
    r = client.post("/x", data={"bias": '{"D": 1.5, "E": 1.5}'})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["bias"] == {"D": 1.5, "E": 1.5}


def test_bare_list_field_accepts_json_string():
    """Bare ``list`` (no type args) must be treated as complex → decoded from JSON."""
    client = _make_app(WithBareContainers)
    r = client.post("/x", data={"lengths": "[80, 100, 120]"})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["lengths"] == [80, 100, 120]


def test_complex_field_omitted_uses_model_default():
    client = _make_app(WithComplex)
    r = client.post("/x", data={"name": "z"})
    assert r.status_code == 200, r.text
    assert r.json()["params"] == {"name": "z", "bias": None, "lengths": None}


def test_complex_field_invalid_json_returns_422():
    client = _make_app(WithComplex)
    r = client.post("/x", data={"bias": "{not-json"})
    assert r.status_code == 422
    assert "Invalid JSON" in r.text


def test_required_complex_field_missing_returns_422():
    client = _make_app(RequiredFields)
    r = client.post("/x", data={"target_chain": "A"})
    assert r.status_code == 422
    # The missing field's name must surface in the error.
    assert "motif_names" in r.text


def test_required_scalar_field_missing_returns_422():
    client = _make_app(RequiredFields)
    r = client.post("/x", data={"motif_names": '["a"]'})
    assert r.status_code == 422


def test_works_alongside_file_upload():
    client = _make_app(SimpleModel)
    r = client.post(
        "/x",
        data={"name": "abc", "temp": "0.2"},
        files={"pdb": ("input.pdb", b"ATOM 1 N MET A 1")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pdb"] == "input.pdb"
    assert body["params"]["name"] == "abc"
