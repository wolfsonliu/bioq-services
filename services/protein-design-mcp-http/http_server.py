"""HTTP wrapper for protein-design-mcp.

Bridges the upstream stdio MCP server (protein_design_mcp.server) to HTTP so it
can be deployed on Alibaba Cloud Function Compute / any HTTP host. Upstream
code is imported as-is — no edits to opensource/protein-design-mcp/.

Endpoints:
  GET  /healthz           — plaintext health check
  GET  /sse               — MCP 2024 SSE transport (event stream)
  POST /messages/?session_id=...   — SSE counterpart for client→server JSON-RPC
  ANY  /mcp               — MCP 2025-03-26 Streamable HTTP transport

Env:
  PORT       (default 9000)
  LOG_LEVEL  (default info)
"""

from __future__ import annotations

import contextlib
import logging
import os

from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Mount, Route

from protein_design_mcp.server import server  # Server instance with all tools registered

logger = logging.getLogger("protein-design-mcp.http")

# --- SSE transport (MCP 2024) -----------------------------------------------
sse_transport = SseServerTransport("/messages/")


async def handle_sse(request: Request) -> Response:
    # request._send is the documented MCP-SDK pattern for SSE bridging.
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
    return Response()


# --- Streamable HTTP transport (MCP 2025-03-26) -----------------------------
# stateless=True: each request stands alone — required for horizontally scaled
# FC / load-balanced deployments where successive requests may land on
# different container instances.
session_manager = StreamableHTTPSessionManager(
    app=server,
    event_store=None,
    json_response=False,
    stateless=True,
)


async def handle_streamable_http(scope, receive, send) -> None:
    await session_manager.handle_request(scope, receive, send)


# --- Health -----------------------------------------------------------------
async def healthz(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@contextlib.asynccontextmanager
async def lifespan(_: Starlette):
    async with session_manager.run():
        logger.info("Streamable HTTP session manager started")
        yield


app = Starlette(
    debug=False,
    lifespan=lifespan,
    routes=[
        Route("/healthz", endpoint=healthz),
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse_transport.handle_post_message),
        Mount("/mcp", app=handle_streamable_http),
    ],
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "9000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=log_level)
