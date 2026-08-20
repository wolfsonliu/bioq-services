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


def _manifest_tree(tmp_path):
    p = _yaml(tmp_path)
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    (mdir / "openbpmd-server.manifest.json").write_text(
        '{"service": "openbpmd", "endpoints": [{"path": "/api/score"}]}',
        encoding="utf-8",
    )
    (mdir / "openbpmd-server.openapi.json").write_text(
        '{"paths": {"/api/score": {}}}', encoding="utf-8",
    )
    return p


def test_manifest_reads_manifests_dir(tmp_path):
    reg = ServiceRegistry(_manifest_tree(tmp_path))
    assert reg.manifest("openbpmd-server")["service"] == "openbpmd"
    assert reg.openapi("openbpmd-server")["paths"] == {"/api/score": {}}


def test_manifest_missing_returns_none(tmp_path):
    reg = ServiceRegistry(_yaml(tmp_path))  # no manifests/ dir
    assert reg.manifest("openbpmd-server") is None
    assert reg.openapi("openbpmd-server") is None
