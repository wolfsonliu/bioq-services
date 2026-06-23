"""Pydantic response schemas for the ColabFold-compatible HTTP protocol.

These are the wire shapes returned by the four `/ticket/*` + `/result/*`
endpoints. Form-data inputs (`q`, `mode`) are parsed at the route layer via
`Form(...)`, not via a Pydantic body model, because the ColabFold protocol uses
``multipart/form-data`` rather than JSON.

Status strings are kept *exactly* as ColabFold's reference server emits them
(uppercase, no spaces, no aliases). The ColabFold client compares them
literally — case matters.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# Status vocabulary expected by the ColabFold client.
#
# - PENDING / RUNNING : transient; the client keeps polling.
# - COMPLETE          : terminal success; the client GETs /result/download/<id>.
# - ERROR             : terminal failure; the client surfaces the message.
# - RATELIMIT         : transient; the client backs off (we don't emit this in
#                       v0.0.1 but it's part of the protocol for forward compat).
# - MAINTENANCE       : transient; same idea, reserved.
ColabFoldStatus = Literal[
    "PENDING",
    "RUNNING",
    "COMPLETE",
    "ERROR",
    "RATELIMIT",
    "MAINTENANCE",
]


class TicketSubmitResponse(BaseModel):
    """Response body for POST /ticket/msa and POST /ticket/pair."""

    id: str
    status: ColabFoldStatus


class TicketStatusResponse(BaseModel):
    """Response body for GET /ticket/<id>.

    `error` is populated only when `status == "ERROR"` (truncated upstream by
    the framework). The ColabFold client ignores unknown fields, so adding this
    one is safe.
    """

    id: str
    status: ColabFoldStatus
    error: Optional[str] = None


__all__ = [
    "ColabFoldStatus",
    "TicketStatusResponse",
    "TicketSubmitResponse",
]
