"""Drop-in replacement for ``from torch.cuda.nvtx import range as nvtx_range``.

PyTorch builds without NVTX support (common in conda envs) define
``torch.cuda.nvtx.range`` as a pure-Python function that only fails at
*call time* — the import itself succeeds. A naive try/except around the
import therefore never triggers.

This module probes NVTX at import time by actually entering the context
manager. If the probe raises, we fall back to ``contextlib.nullcontext``
which is a no-op ``with``-block — zero overhead, zero profiling markers.

Usage (after vendor.sh patches the SE3Transformer sources)::

    from nvtx_compat import nvtx_range   # instead of torch.cuda.nvtx
"""

from __future__ import annotations

from contextlib import nullcontext

try:
    from torch.cuda.nvtx import range as _torch_nvtx_range

    # Probe: actually enter+exit the context manager.
    _ctx = _torch_nvtx_range("probe")
    _ctx.__enter__()
    _ctx.__exit__(None, None, None)

    nvtx_range = _torch_nvtx_range
except Exception:
    nvtx_range = nullcontext  # type: ignore[assignment]
