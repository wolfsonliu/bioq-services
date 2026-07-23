from __future__ import annotations

import pytest

from server.registry import ServiceRegistry


def _yaml(tmp_path):
    p = tmp_path / "services.yaml"
    p.write_text(
        "services:\n"
        "  openbpmd-server:\n"
        "    url: https://fc-openbpmd-x.cn-hangzhou-vpc.fcapp.run\n"
        "  dockq-server:\n"
        "    url: https://fc-dockq-y.cn-hangzhou-vpc.fcapp.run/\n"
        "    tier: cold\n",
        encoding="utf-8",
    )
    return p


def test_list_and_get(tmp_path):
    reg = ServiceRegistry(_yaml(tmp_path))
    assert reg.list() == ["dockq-server", "openbpmd-server"]
    assert reg.base_url("dockq-server") == "https://fc-dockq-y.cn-hangzhou-vpc.fcapp.run"
    assert reg.record("dockq-server").tier == "cold"


def test_unknown_service(tmp_path):
    reg = ServiceRegistry(_yaml(tmp_path))
    with pytest.raises(KeyError):
        reg.base_url("nope")
