"""Form-data Pydantic flattening helper for endpoints that also accept files.

FastAPI's ``Annotated[Model, Form()]`` only flattens model fields into individual
form parameters when the route has *no other* file/form parameters. As soon as
``UploadFile = File(...)`` is added alongside, FastAPI falls back to treating
``params`` as a single nested object: clients sending ``-F model_variant=vanilla``
hit 422 ``params: Field required`` (or, if the model param has a default, every
field is silently dropped and defaults take over). This is a known FastAPI
limitation, not a bug in our request models.

``model_form_depends(Model)`` builds a ``Depends()``-compatible callable whose
signature exposes one ``Form(...)`` parameter per Pydantic field. FastAPI then
emits a flat OpenAPI schema, accepts ``multipart/form-data`` requests with
individual fields, and runs FastAPI-level type coercion. The wrapper finally
constructs ``Model(**kwargs)`` so all ``model_validator(mode="after")``
cross-field checks still fire.

Complex field types (``dict``, ``list``, nested ``BaseModel``) are received as
JSON strings — that's the only way to encode structured values in a form field
anyway. The wrapper ``json.loads`` them before calling the model constructor.

Usage::

    from fastapi import Depends, File, UploadFile
    from bioagent_service.forms import model_form_depends

    @app.post("/api/design")
    def post_design(
        params: DesignRequest = Depends(model_form_depends(DesignRequest)),
        pdb: Optional[UploadFile] = File(None),
    ):
        ...
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Optional, Type, TypeVar, Union, get_args, get_origin

from fastapi import Form, HTTPException
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

M = TypeVar("M", bound=BaseModel)


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional) for `Optional[T]` / `Union[T, None]`."""
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _is_complex_type(annotation: Any) -> bool:
    """Form fields can only carry scalars. dict/list/BaseModel must arrive as JSON."""
    inner, _ = _strip_optional(annotation)
    origin = get_origin(inner)
    if origin in (dict, list, set, tuple, frozenset):
        return True
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return True
    return False


def model_form_depends(model_cls: Type[M]) -> Callable[..., M]:
    """Build a FastAPI dependency that parses ``model_cls`` from flat form fields.

    Each pydantic field becomes one ``Form()`` parameter on the returned
    function's synthetic signature. Complex-typed fields (dict/list/nested
    BaseModel) are received as JSON strings and decoded before model
    construction; pydantic's own validators still run on the assembled object.
    """
    parameters: list[inspect.Parameter] = []
    complex_field_names: set[str] = set()

    for field_name, field in model_cls.model_fields.items():
        annotation = field.annotation
        is_complex = _is_complex_type(annotation)
        is_required = field.is_required()

        if is_complex:
            complex_field_names.add(field_name)
            param_annotation: Any = str if is_required else Optional[str]
            if is_required:
                form_default = Form(..., description=field.description)
            else:
                form_default = Form(None, description=field.description)
        else:
            param_annotation = annotation
            if is_required:
                form_default = Form(..., description=field.description)
            else:
                default_value = (
                    field.default if field.default is not PydanticUndefined else None
                )
                form_default = Form(default_value, description=field.description)

        parameters.append(
            inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=param_annotation,
                default=form_default,
            )
        )

    def factory(**kwargs: Any) -> M:
        for name in complex_field_names:
            value = kwargs.get(name)
            if value is None or value == "":
                if model_cls.model_fields[name].is_required():
                    raise HTTPException(
                        status_code=422,
                        detail=[
                            {
                                "type": "missing",
                                "loc": ["body", name],
                                "msg": "Field required (JSON-encoded).",
                            }
                        ],
                    )
                kwargs.pop(name, None)
                continue
            if isinstance(value, str):
                try:
                    kwargs[name] = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=[
                            {
                                "type": "json_invalid",
                                "loc": ["body", name],
                                "msg": f"Invalid JSON for {name!r}: {exc.msg}",
                            }
                        ],
                    ) from exc
        try:
            return model_cls(**kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    factory.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=parameters, return_annotation=model_cls
    )
    factory.__name__ = f"_form_{model_cls.__name__}"
    factory.__doc__ = (
        f"Auto-generated multipart/form-data dependency for {model_cls.__name__}. "
        "Each model field becomes one Form() parameter; complex types (dict/list/"
        "nested BaseModel) are accepted as JSON strings."
    )
    return factory


__all__ = ["model_form_depends"]
