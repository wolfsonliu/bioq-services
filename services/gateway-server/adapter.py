"""Minimal JobAdapter — the gateway dispatches, it runs no local subprocess.

create_app requires an adapter for its generic router; a name is all the
gateway needs.
"""

from __future__ import annotations

from bioagent_service import JobAdapter


class GatewayAdapter(JobAdapter):
    name = "gateway"
