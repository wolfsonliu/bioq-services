"""Canonical structure file layout shared by every folding adapter.

The orchestrator unzips a downstream service's output into
``<jobs_base>/<task_id>/outputs/<method>/`` with whatever directory tree
the upstream service produced (alphafold's ``input/ranked_<N>.pdb``,
boltz's ``boltz_results_input/predictions/input/input_model_<N>.cif``,
promera's ``cofold/cofold_seed<s>_samp<j>.cif``, ...).  Those tails leak
upstream implementation choices into our API contract — a Boltz minor
version bump that renames ``boltz_results_input/`` would break every URL
already issued to customers.

To stabilize the public surface, every adapter routes each ranked
structure through :func:`publish_canonical`: the raw file is copied
(NOT moved — originals stay for debugging / audit) to a flat,
upstream-agnostic name and the StructureFile uses that name in its URL.

Canonical scheme:
    <method>/rank_<i>.<ext>           e.g.  alphafold/rank_0.pdb
where ``i`` is the cross-method rank (0 = best for this method) and
``ext`` is ``cif`` or ``pdb``.

If the upstream layout changes, the canonical URL stays the same.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal


def publish_canonical(
    *,
    src: Path,
    downloaded_dir: Path,
    ensemble_task_id: str,
    method: str,
    rank: int,
    format: Literal["cif", "pdb"],
) -> tuple[str, str]:
    """Copy ``src`` into the canonical slot and return ``(url, original_filename)``.

    ``downloaded_dir`` is the per-method outputs root that ensemble-server's
    download route resolves URLs against
    (``<jobs_base>/<task_id>/outputs/<method>/``).  The canonical file is
    placed at the root of that dir so customer-visible URLs are short:
    ``/v1/jobs/<task_id>/structures/<method>/rank_<i>.<ext>``.

    Copies are idempotent — re-running normalize_output (e.g. after a
    retried download) safely overwrites with the same content.
    """
    dest = downloaded_dir / f"rank_{rank}.{format}"
    shutil.copyfile(src, dest)
    url = f"/v1/jobs/{ensemble_task_id}/structures/{method}/{dest.name}"
    return url, src.name


__all__ = ["publish_canonical"]
