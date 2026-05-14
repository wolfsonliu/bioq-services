"""`bioagent-service-mcp-stdio` CLI — run a service's MCP server over stdio.

Usage::

    bioagent-service-mcp-stdio --app proteinmpnn_server.app:app
    bioagent-service-mcp-stdio --app rfantibody_server.app:app

The argument is a Python module path (`module:attr`) that imports to the
service's FastAPI `app` object. The CLI builds an MCP server mirroring that
app and runs it over stdio (JSON-RPC on stdin/stdout) — the transport that
Claude Desktop, Cursor, and Codex IDE plugins use.

This is intended for local development and IDE integration. For Alibaba
Cloud FC, the service's existing HTTP entrypoint already mounts the same
MCP server at `/mcp` via `attach_mcp` — no stdio needed there.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from typing import Any


def _load_app(spec: str) -> Any:
    """`module.path:attr` -> the FastAPI app object."""
    if ":" not in spec:
        raise SystemExit(f"--app must be 'module.path:attr' (got {spec!r})")
    mod_path, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_path)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise SystemExit(f"module {mod_path!r} has no attribute {attr!r}") from e


def main() -> None:
    parser = argparse.ArgumentParser(prog="bioagent-service-mcp-stdio")
    parser.add_argument(
        "--app",
        required=True,
        help="Import path of the FastAPI app, e.g. 'proteinmpnn_server.app:app'.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for the framework. stdio MCP requires stdout to "
        "be reserved for JSON-RPC, so all logs go to stderr.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    app = _load_app(args.app)

    # Reuse the server already attached to the app if attach_mcp was called;
    # otherwise build one on the fly with the same auto-discovery logic.
    mcp = getattr(app.state, "mcp", None)
    if mcp is None:
        from bioagent_service.mcp_server import make_mcp_server

        mcp = make_mcp_server(app, app.state.adapter, app.state.settings)
        app.state.mcp = mcp

    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":  # pragma: no cover
    main()
