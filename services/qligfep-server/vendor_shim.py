"""Inject upstream qligfep repo into sys.path for wrapper subprocesses."""
from __future__ import annotations

import os
import sys
from pathlib import Path

QLIGFEP_UPSTREAM_DIR = Path(
    os.environ.get("QLIGFEP_UPSTREAM_DIR", "/opt/qligfep-server/upstream/qligfep")
)


def inject() -> Path:
    """Prepend QLIGFEP_UPSTREAM_DIR to sys.path, return it."""
    d = str(QLIGFEP_UPSTREAM_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    return QLIGFEP_UPSTREAM_DIR


def script(name: str) -> Path:
    """Return absolute path to any upstream script (e.g. 'protprep.py', 'scripts/COG.py')."""
    return QLIGFEP_UPSTREAM_DIR / name
