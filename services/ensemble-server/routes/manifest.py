"""Healthz / method discovery / manifest routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..task_kind import TaskKind

router = APIRouter()


@router.get("/v1/healthz")
def healthz(request: Request) -> dict:
    return {
        "status": "ok",
        "service": "ensemble",
        "version": request.app.version,
    }


@router.get("/v1/methods")
def list_methods(request: Request, task_kind: str = "folding") -> dict:
    """List registered methods for a given TaskKind, with their options schema."""
    try:
        tk = TaskKind(task_kind)
    except ValueError:
        raise HTTPException(422, f"unknown task_kind: {task_kind!r}")

    registry = request.app.state.registry
    return {
        "task_kind": task_kind,
        "methods": [
            {
                "name": name,
                "options_schema": registry.get(tk, name).method_options_schema.model_json_schema(),
                "estimated_runtime_seconds_default": registry.get(tk, name).estimate_runtime_seconds(None),
            }
            for name in registry.list_methods(tk)
        ],
    }


@router.get("/v1/manifest")
def manifest(request: Request) -> dict:
    """Per-task-kind list of registered methods."""
    registry = request.app.state.registry
    return {
        "service": "ensemble",
        "version": request.app.version,
        "task_kinds": ["folding"],
        "methods": {
            "folding": registry.list_methods(TaskKind.FOLDING),
        },
    }
