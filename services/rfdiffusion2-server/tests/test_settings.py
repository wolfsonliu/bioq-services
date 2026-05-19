"""Default-path snapshot tests for RFdiffusion2Settings.

These pin down the new /opt/rfdiffusion2-server/* layout introduced by the
vendor refactor (see engineering/decisions/2026-05-19-rfdiffusion2-server-vendor.md).
"""
from __future__ import annotations

from pathlib import Path

from server.settings import RFdiffusion2Settings


def test_default_root():
    s = RFdiffusion2Settings()
    assert s.root == Path("/opt/rfdiffusion2-server")


def test_default_pythonpath_points_at_upstream():
    s = RFdiffusion2Settings()
    assert s.pythonpath == Path("/opt/rfdiffusion2-server/upstream")


def test_default_inference_script_under_upstream():
    s = RFdiffusion2Settings()
    assert s.inference_script == Path(
        "/opt/rfdiffusion2-server/upstream/rf_diffusion/run_inference.py"
    )


def test_default_models_dir_under_upstream():
    s = RFdiffusion2Settings()
    assert s.models_dir == Path(
        "/opt/rfdiffusion2-server/upstream/rf_diffusion/model_weights"
    )


def test_python_interpreter_unchanged():
    """The conda env path is independent of the source layout."""
    s = RFdiffusion2Settings()
    assert s.python == Path("/opt/conda/envs/rfd2/bin/python")
