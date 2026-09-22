"""settings 的默认值与 env 覆盖行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

from server.settings import Bindcraft2Settings


def test_defaults_match_container_layout():
    s = Bindcraft2Settings(_env_file=None)
    assert s.jobs_base_dir == Path("/data/bindcraft2_jobs")
    assert s.root == Path("/opt/bindcraft")
    assert s.shipped_weights_dir == Path("/opt/bindcraft")
    assert s.python == "/opt/venv/bin/python"
    assert s.module == "bindcraft.cli"
    assert s.alphafold_params_dir == Path("/data/models/bindcraft2/alphafold")
    assert s.compile_cache_dir == Path("/data/models/bindcraft2/xla_cache")
    assert s.max_concurrent_jobs == 1
    assert s.workers_per_gpu == 1
    assert s.max_workers_per_gpu == 1
    assert s.default_max_trajectories == 500
    assert s.gpu_probe_ttl_seconds == 300


def test_env_prefix_is_bindcraft2(monkeypatch):
    monkeypatch.setenv("BINDCRAFT2_MAX_CONCURRENT_JOBS", "3")
    monkeypatch.setenv("BINDCRAFT2_DEFAULT_MAX_TRAJECTORIES", "7")
    s = Bindcraft2Settings(_env_file=None)
    assert s.max_concurrent_jobs == 3
    assert s.default_max_trajectories == 7


def test_unknown_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("BINDCRAFT2_NOT_A_FIELD", "1")
    assert Bindcraft2Settings(_env_file=None).module == "bindcraft.cli"


def test_bounds_are_enforced():
    with pytest.raises(ValueError):
        Bindcraft2Settings(_env_file=None, workers_per_gpu=0)
    with pytest.raises(ValueError):
        Bindcraft2Settings(_env_file=None, max_concurrent_jobs=99)
