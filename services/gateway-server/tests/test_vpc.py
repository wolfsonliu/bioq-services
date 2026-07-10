from __future__ import annotations

from server.auth.vpc import is_vpc_host


def test_vpc_host_matches_vpc_fcapp_run():
    assert is_vpc_host("fc-gateway-abc.cn-hangzhou-vpc.fcapp.run") is True


def test_vpc_host_rejects_public():
    assert is_vpc_host("fc-gateway-abc.cn-hangzhou.fcapp.run") is False


def test_vpc_host_localhost_and_none():
    assert is_vpc_host("localhost:9000") is True
    assert is_vpc_host("127.0.0.1") is True
    assert is_vpc_host(None) is False
    assert is_vpc_host("") is False
