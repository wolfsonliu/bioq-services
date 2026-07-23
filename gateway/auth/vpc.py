"""VPC URL detection for auth bypass. Host header is the only needed signal —
VPC URLs are not routable from the public internet."""

from __future__ import annotations


def is_vpc_host(host: str | None) -> bool:
    if not host:
        return False
    host_lc = host.lower().split(":")[0]
    return "-vpc.fcapp.run" in host_lc or host_lc in ("127.0.0.1", "localhost")
