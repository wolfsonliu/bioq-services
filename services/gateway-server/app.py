"""gateway-server FastAPI app.

Built on bioagent_service.create_app for /healthz + settings + MCP. The
gateway's own API lives under /v1/* (added in later tasks). app.state gets
`db`, `registry`, `discover`, `dispatch` wired here.
"""

from __future__ import annotations

from bioagent_service import attach_mcp, create_app, read_version_file

from .adapter import GatewayAdapter
from .settings import GatewaySettings

settings = GatewaySettings()
adapter = GatewayAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="Gateway Server",
    version=read_version_file(__file__, default="0.0.1"),
)

# /v1/* routers are registered in later tasks (auth/dispatch/presign).

attach_mcp(app)
