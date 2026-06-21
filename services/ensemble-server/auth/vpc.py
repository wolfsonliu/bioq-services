"""VPC URL detection for auth bypass.

Requests reaching the function via a VPC URL (`*-vpc.fcapp.run`) are
considered already on the trusted internal network and bypass auth.  The
Host header is the only signal needed because VPC URLs aren't routable
from the public internet — an attacker outside the VPC can't reach the
function via that hostname at all.

Local-dev hosts (`localhost`, `127.0.0.1`) are also treated as VPC for
the same reason (you're running on the same machine as the server).
"""

from __future__ import annotations


def is_vpc_host(host: str | None) -> bool:
    """Return True when `host` indicates VPC URL or local-dev access.

    Strips port if present.  Case-insensitive.

    Examples:
        is_vpc_host("fc-ensemble-XXX.cn-hangzhou-vpc.fcapp.run") → True
        is_vpc_host("fc-ensemble-XXX.cn-hangzhou.fcapp.run")     → False
        is_vpc_host("localhost:9000")                            → True
        is_vpc_host("127.0.0.1")                                 → True
        is_vpc_host("")                                          → False
        is_vpc_host(None)                                        → False
    """
    if not host:
        return False
    host_lc = host.lower().split(":")[0]
    return (
        "-vpc.fcapp.run" in host_lc
        or host_lc in ("127.0.0.1", "localhost")
    )
