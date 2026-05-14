"""Test setup: import the service modules without installing the package.

The Dockerfile copies `services/proteinmpnn-server/` into `/opt/proteinmpnn/server/`
and imports it as `server.app:app`. For local pytest, we wire up the same
alias by registering this directory as the namespace package `server`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
