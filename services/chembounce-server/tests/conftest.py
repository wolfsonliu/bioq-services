"""Register `server` package alias + `fc` marker."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parent.parent

if "server" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "server",
        SERVICE_DIR / "__init__.py",
        submodule_search_locations=[str(SERVICE_DIR)],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["server"] = module
        spec.loader.exec_module(module)

from bioagent_service.fc_testing import (  # noqa: E402
    register_fc_marker,
    skip_fc_tests_unless_enabled,
)


def pytest_configure(config):
    register_fc_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_fc_tests_unless_enabled(config, items)


@pytest.fixture(scope="session")
def local_output_dir() -> Path:
    base = Path(__file__).resolve().parent / "fc_outputs"
    run_dir = base / f"run-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
