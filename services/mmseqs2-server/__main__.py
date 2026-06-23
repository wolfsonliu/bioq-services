"""Run mmseqs2-server with: python -m server.

CLI batch mode (`python -m server <endpoint> ...`) is intentionally not wired
up in v0.0.1 — the ColabFold protocol is HTTP-only by design, and Slurm users
should invoke ``python -m server.orchestrator ...`` directly. See
``services/mmseqs2-server/orchestrator.py`` for the orchestrator CLI surface.
"""

from __future__ import annotations

import os

import uvicorn

from .app import app


def main() -> None:
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
