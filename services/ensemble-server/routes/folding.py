"""Folding ensemble routes.

`POST /v1/folding/ensemble` — submit one or more folding methods on the same
sequence input.  Returns 202 + task_id.  Client polls GET /v1/jobs/{task_id}
for status.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..auth.deps import require_api_key
from ..folding.schemas import FoldingInput
from ..task_kind import TaskKind

router = APIRouter()


class FoldingEnsembleRequest(BaseModel):
    input: FoldingInput
    methods: list[str] | None = None       # default: all registered folding methods
    method_options: dict[str, dict] = Field(default_factory=dict)


@router.post("/v1/folding/ensemble", status_code=202)
async def submit_folding_ensemble(
    request: Request,
    body: FoldingEnsembleRequest,
    api_key=Depends(require_api_key),
) -> dict:
    """Submit a folding ensemble job.  Returns 202 with the new task_id."""
    orchestrator = request.app.state.orchestrator
    registry = request.app.state.registry

    methods = body.methods or registry.list_methods(TaskKind.FOLDING)
    if not methods:
        raise HTTPException(503, "no folding methods registered")

    for m in methods:
        try:
            registry.get(TaskKind.FOLDING, m)
        except KeyError:
            raise HTTPException(422, f"unknown method: {m!r}")

    job = await orchestrator.submit(
        task_kind=TaskKind.FOLDING,
        input=body.input,
        methods=methods,
        method_options=body.method_options,
        customer_id=api_key.customer_id,
    )
    return {
        "task_id": job.task_id,
        "status": "accepted",
        "requested_methods": job.requested_methods,
    }
