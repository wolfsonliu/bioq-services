"""Test setup: import the service modules without installing the package.

The Dockerfile copies `services/rfantibody-server/` into `/opt/rfantibody/server/`
and imports it as `server.app:app`. For local pytest, we wire up the same alias
by inserting the parent dir on sys.path under that name.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent

# Make the directory importable under the name `server` so `from server.app import app`
# works the same way it does inside the Docker image.
if "server" not in sys.modules:
    # Trick: load the directory as a namespace package called "server".
    parent = SERVICE_DIR.parent  # services/
    # Map a temporary alias dir into sys.modules so `server` resolves to our service.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "server", SERVICE_DIR / "__init__.py",
        submodule_search_locations=[str(SERVICE_DIR)],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["server"] = module
        spec.loader.exec_module(module)


# ---------------------------------------------------------------------------
# `fc` marker — opt-in tests that hit the deployed Function Compute URL.
# ---------------------------------------------------------------------------
from bioagent_service.fc_testing import (  # noqa: E402
    register_fc_marker,
    skip_fc_tests_unless_enabled,
)


def pytest_configure(config):
    register_fc_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_fc_tests_unless_enabled(config, items)
