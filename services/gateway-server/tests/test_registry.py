from __future__ import annotations

import pytest

from server.registry import ServiceRegistry


def _md(tmp_path):
    p = tmp_path / "aliyun_fc_url.md"
    p.write_text(
        "# registry\n"
        "openbpmd-server: https://fc-openbpmd-x.cn-hangzhou-vpc.fcapp.run\n"
        "dockq-server: https://fc-dockq-y.cn-hangzhou-vpc.fcapp.run/\n",
        encoding="utf-8",
    )
    return p


def test_list_and_get(tmp_path):
    reg = ServiceRegistry(_md(tmp_path))
    assert set(reg.list()) == {"openbpmd-server", "dockq-server"}
    assert reg.base_url("dockq-server") == "https://fc-dockq-y.cn-hangzhou-vpc.fcapp.run"


def test_unknown_service(tmp_path):
    reg = ServiceRegistry(_md(tmp_path))
    with pytest.raises(KeyError):
        reg.base_url("nope")
