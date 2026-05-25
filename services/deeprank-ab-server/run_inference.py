"""Wrapper for upstream DeepRank-Ab inference.py.

Loads the upstream script as a module, applies runtime patches, then
delegates to main(). Replaces sed-based Dockerfile patches with a proper
Python shim so the upstream source stays unmodified.

Patches applied:

1. fetch_weights() -- upstream returns a bare filename
   ("esm2_t33_650M_UR50D.pt") which fails when CWD != the weight
   directory. Patched to return the absolute path from WEIGHT_PATH.

2. NUM_WORKERS -- upstream hardcodes 96, excessive for FC GPU instances
   (~8 vCPU). Overridden via DEEPRANK_AB_NUM_WORKERS env var (default 8).
"""

from __future__ import annotations

import importlib.util
import os
import sys


def _load_upstream():
    root = os.environ.get("DEEPRANK_AB_ROOT", "/opt/deeprank-ab")
    script = os.path.join(root, "DeepRank-Ab", "scripts", "inference.py")
    spec = importlib.util.spec_from_file_location("inference", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inference"] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch(mod):
    _original_fetch = mod.fetch_weights

    def _fetch_weights() -> str:
        _original_fetch()
        return os.getenv("WEIGHT_PATH") or f"{mod.ESM_MODEL}.pt"

    mod.fetch_weights = _fetch_weights
    mod.NUM_WORKERS = int(os.getenv("DEEPRANK_AB_NUM_WORKERS", "8"))


if __name__ == "__main__":
    inference = _load_upstream()
    _patch(inference)
    inference.main()
