from __future__ import annotations

from server.settings import GatewaySettings


def test_discovery_timeout_defaults():
    s = GatewaySettings()
    assert s.discovery_ttl_sec == 300.0
    assert s.discovery_negative_ttl_sec == 15.0
    assert s.discovery_read_timeout_sec == 8.0
    assert s.discovery_connect_timeout_sec == 5.0
