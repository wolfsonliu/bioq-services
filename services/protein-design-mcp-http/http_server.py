"""HTTP wrapper for protein-design-mcp.

Bridges the upstream stdio MCP server (protein_design_mcp.server) to HTTP so it
can be deployed on Alibaba Cloud Function Compute / any HTTP host. Upstream
code is imported as-is — no edits to opensource/protein-design-mcp/.

This wrapper also augments the upstream tool surface with two MCP tools that
close the file-IO gap for external clients:

  * `stage_file` — upload PDB / FASTA text → returns server-local path
  * `fetch_file` — download a server-local file (e.g. a designed PDB produced
                   by `design_binder`) as text or base64

Every other upstream tool consumes / produces file-system paths
(`target_pdb`, `backbone_pdb`, `pdb_path`, ...), which are unreachable for
clients that don't share storage with the server — these two tools close that
gap without leaving the MCP protocol.

Endpoints:
  GET  /healthz           — plaintext health check
  GET  /sse               — MCP 2024 SSE transport (event stream)
  POST /messages/?session_id=...   — SSE counterpart for client→server JSON-RPC
  ANY  /mcp               — MCP 2025-03-26 Streamable HTTP transport

Env:
  PORT                 (default 9000)
  LOG_LEVEL            (default info)
  STAGE_DIR            (default /data/protein_design_staged) — uploads root.
                       `/data` is the NAS-mapped path in the FC deployment;
                       persists across cold starts and is shared across
                       horizontally scaled instances.
  STAGE_MAX_BYTES      (default 20 * 1024 * 1024)            — upload size cap
  FETCH_ALLOWED_ROOTS  (default <STAGE_DIR>:/data:/tmp)       — colon-separated
                       absolute prefixes `fetch_file` may read from
  FETCH_MAX_BYTES      (default 20 * 1024 * 1024)            — fetch size cap
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Mount, Route

from protein_design_mcp.server import (
    call_tool as _upstream_call_tool,
    list_tools as _upstream_list_tools,
    server,  # Server instance with all upstream tools already registered
)

logger = logging.getLogger("protein-design-mcp.http")

# --- File staging tool ------------------------------------------------------
# Every upstream tool that takes structures/sequences expects a server-local
# path; external MCP clients have no way to inject content into those paths.
# `stage_file` writes the content under STAGE_DIR/<uuid>/<filename> and hands
# the absolute path back, which the agent then passes as `target_pdb` etc.

# Default to /data (NAS-mapped in the FC deployment) so staged files survive
# cold starts and are visible to every horizontally scaled instance.
_STAGE_DIR = Path(os.environ.get("STAGE_DIR", "/data/protein_design_staged"))
_STAGE_MAX_BYTES = int(os.environ.get("STAGE_MAX_BYTES", str(20 * 1024 * 1024)))
_ALLOWED_KINDS = ("pdb", "fasta")

_STAGE_FILE_TOOL = Tool(
    name="stage_file",
    description=(
        "Stage a PDB or FASTA file on the server by uploading its text content. "
        "Returns a server-local absolute path you can pass to any other tool's "
        "file-path argument (e.g. `target_pdb`, `backbone_pdb`, `pdb_path`, "
        "`complex_pdb`, `expected_structure`). Use this BEFORE calling tools "
        f"that consume structures or sequences. Max content size: {_STAGE_MAX_BYTES} bytes."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Full text contents of the PDB or FASTA file.",
            },
            "kind": {
                "type": "string",
                "enum": list(_ALLOWED_KINDS),
                "description": "File kind — used as the file extension.",
            },
            "filename": {
                "type": "string",
                "description": (
                    "Optional base filename (e.g. 'target.pdb'). Path components are "
                    "stripped. Defaults to 'input.<kind>'."
                ),
            },
        },
        "required": ["content", "kind"],
    },
)


def _stage_file_sync(content: str, kind: str, filename: str | None) -> dict[str, Any]:
    if kind not in _ALLOWED_KINDS:
        return {"error": f"unsupported kind: {kind!r}", "allowed": list(_ALLOWED_KINDS)}
    encoded = content.encode("utf-8")
    if len(encoded) > _STAGE_MAX_BYTES:
        return {
            "error": "content exceeds size limit",
            "size_bytes": len(encoded),
            "limit_bytes": _STAGE_MAX_BYTES,
        }
    job_dir = _STAGE_DIR / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    # Strip any path components from a caller-supplied filename to avoid
    # writing outside the staging dir.
    base = Path(filename).name if filename else f"input.{kind}"
    if not base:
        base = f"input.{kind}"
    target = job_dir / base
    target.write_bytes(encoded)
    return {
        "path": str(target),
        "kind": kind,
        "bytes": len(encoded),
    }


async def handle_stage_file(arguments: dict[str, Any]) -> dict[str, Any]:
    content = arguments.get("content")
    kind = arguments.get("kind")
    if not isinstance(content, str) or not isinstance(kind, str):
        return {"error": "`content` and `kind` are required strings"}
    return _stage_file_sync(content, kind, arguments.get("filename"))


# --- File fetch tool --------------------------------------------------------
# Symmetric to `stage_file`: lets clients pull back any server-local file an
# upstream tool produced (designed PDBs, FASTA outputs, score JSON, etc.).
# Reads are restricted to a configurable set of root prefixes to avoid
# turning the MCP endpoint into an arbitrary file-read primitive.

# `/data` covers NAS-mapped staging + any upstream tool that writes there;
# `/tmp` is kept so per-instance scratch outputs (RFdiffusion, ProteinMPNN, ...)
# remain fetchable within a single warm instance.
_FETCH_ALLOWED_ROOTS: list[Path] = [
    Path(p).resolve()
    for p in os.environ.get(
        "FETCH_ALLOWED_ROOTS",
        f"{_STAGE_DIR}:/data:/tmp",
    ).split(":")
    if p
]
_FETCH_MAX_BYTES = int(os.environ.get("FETCH_MAX_BYTES", str(20 * 1024 * 1024)))

_FETCH_FILE_TOOL = Tool(
    name="fetch_file",
    description=(
        "Download a server-local file (e.g. a designed PDB / FASTA / scores "
        "JSON / quiver) produced by another tool. Pass the absolute `path` "
        "returned in any upstream tool's response. Reads are restricted to: "
        f"{[str(p) for p in _FETCH_ALLOWED_ROOTS]}. Max file size: "
        f"{_FETCH_MAX_BYTES} bytes. Use encoding='base64' for binary outputs "
        "(quiver, pickled tensors); the default 'text' returns UTF-8 strings "
        "suitable for PDB / FASTA / JSON."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute server-local path to fetch.",
            },
            "encoding": {
                "type": "string",
                "enum": ["text", "base64"],
                "default": "text",
                "description": (
                    "How to encode the response body. 'text' decodes as UTF-8 "
                    "and errors out on binary; 'base64' is safe for any bytes."
                ),
            },
        },
        "required": ["path"],
    },
)


def _is_under_allowed_root(p: Path) -> bool:
    resolved = p.resolve()
    return any(
        resolved == root or root in resolved.parents for root in _FETCH_ALLOWED_ROOTS
    )


async def handle_fetch_file(arguments: dict[str, Any]) -> dict[str, Any]:
    path_str = arguments.get("path")
    if not isinstance(path_str, str):
        return {"error": "`path` is required string"}
    encoding = arguments.get("encoding", "text")
    if encoding not in ("text", "base64"):
        return {"error": f"unsupported encoding: {encoding!r}"}

    path = Path(path_str)
    if not path.is_absolute():
        return {"error": "path must be absolute", "path": path_str}
    if not _is_under_allowed_root(path):
        return {
            "error": "path not under allowed roots",
            "path": path_str,
            "allowed_roots": [str(r) for r in _FETCH_ALLOWED_ROOTS],
        }
    if not path.exists():
        return {"error": "file not found", "path": str(path)}
    if not path.is_file():
        return {"error": "not a regular file", "path": str(path)}

    size = path.stat().st_size
    if size > _FETCH_MAX_BYTES:
        return {
            "error": "file exceeds size limit",
            "path": str(path),
            "size_bytes": size,
            "limit_bytes": _FETCH_MAX_BYTES,
        }

    raw = path.read_bytes()
    if encoding == "text":
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "error": "file is not UTF-8; retry with encoding='base64'",
                "path": str(path),
                "size_bytes": size,
            }
        return {
            "path": str(path),
            "encoding": "text",
            "bytes": size,
            "content": content,
        }
    return {
        "path": str(path),
        "encoding": "base64",
        "bytes": size,
        "content": base64.b64encode(raw).decode("ascii"),
    }


# Re-register list_tools / call_tool on the upstream Server instance. The MCP
# decorator pattern installs request handlers into `server.request_handlers`
# (keyed by request type), so the most-recent registration wins. We delegate
# to the upstream functions for every name we don't own.


@server.list_tools()
async def list_tools_with_file_io() -> list[Tool]:
    base = await _upstream_list_tools()
    return [*base, _STAGE_FILE_TOOL, _FETCH_FILE_TOOL]


@server.call_tool()
async def call_tool_with_file_io(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "stage_file":
        result = await handle_stage_file(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    if name == "fetch_file":
        result = await handle_fetch_file(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    return await _upstream_call_tool(name, arguments)


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
    _STAGE_DIR.mkdir(parents=True, exist_ok=True)
    async with session_manager.run():
        logger.info("Streamable HTTP session manager started; stage_dir=%s", _STAGE_DIR)
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
