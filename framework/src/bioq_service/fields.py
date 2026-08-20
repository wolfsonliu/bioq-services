"""Field-annotation helpers for Optional request fields.

An ``Optional[X] = None`` field surfaces in the manifest with ``default: null``
(FastAPI emits no ``default`` key for a None default), which the E12 contract
audit reads as "undeclared".  When a field actually means "omission lets the
service/tool supply a default" or "omission leaves the feature off", the service
annotates it with ``default_semantics(...)`` so the manifest carries a non-null,
machine-readable omission semantics token instead of a bare null.
"""

from typing import Any, Literal

DefaultKind = Literal["auto", "unset"]


def default_semantics(kind: DefaultKind, note: str) -> dict[str, Any]:
    """Marker dict for ``Field(json_schema_extra=...)`` on an Optional field.

    kind  "auto"  -> omitting lets the service/tool supply a default value
                      (selected at runtime or a fixed upstream default).
          "unset" -> omitting leaves the parameter inactive / unused.
    note  one-line prose describing the omission semantics.
    """
    return {"bioq_default": {"kind": kind, "note": note}}


__all__ = ["default_semantics"]