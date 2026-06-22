"""Promera folding adapter.

Maps normalized FoldingInput → promera-server's ``/api/tasks/cofold``
multipart request (chain-keyed JSON schema upload + diffusion options).
Normalizes outputs: ``cofold_seed<i>_samp<j>.cif`` structures plus their
sibling ``*_conf.json`` files (which carry per-sample plddt / ptm / iptm).

promera-server forces the uploaded schema's on-disk filename to
``cofold.json`` (see services/promera-server/app.py:60), so the output
directory is always ``output/cofold/`` and structure filenames have the
``cofold_`` prefix regardless of what the client uploads.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ...folding.schemas import FoldingInput, FoldingMethodResult, StructureFile
from ...task_kind import TaskKind
from ..base import MethodAdapter


class PromeraOptions(BaseModel):
    """Subset of promera-server CofoldRequest exposed to ensemble clients.

    The three ``save_*`` toggles (trajectory, full_confidence, distogram)
    are intentionally omitted — they only produce auxiliary files that the
    ensemble layer doesn't consume, and they materially inflate runtime.
    """

    num_seeds: int = Field(default=1, ge=1, le=10)
    diffusion_samples: int = Field(default=5, ge=1, le=25)
    diffusion_steps: int = Field(default=200, ge=10, le=1000)
    recycling_steps: int = Field(default=4, ge=1, le=20)


class PromeraFoldingAdapter(MethodAdapter[FoldingInput, FoldingMethodResult]):
    name = "promera"
    task_kind = TaskKind.FOLDING
    method_options_schema = PromeraOptions

    def build_request(self, input, options):
        # promera-server expects a JSON file keyed by chain_id:
        #   { "A": {"type": "protein", "sequence": "..."}, "B": {...}, ... }
        # — see services/promera-server/tests/data/test_target.json.  We
        # write it to a tempfile that the HTTPDispatcher reads at submit time;
        # delete=False avoids closing the handle before submit reads bytes.
        schema = {
            s.id: {"type": s.type, "sequence": s.sequence}
            for s in input.sequences
        }
        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="promera_schema_",
            delete=False, encoding="utf-8",
        )
        json.dump(schema, fd)
        fd.close()

        payload = {
            "num_seeds": options.num_seeds,
            "diffusion_samples": options.diffusion_samples,
            "diffusion_steps": options.diffusion_steps,
            "recycling_steps": options.recycling_steps,
        }
        files = {"input_schema": Path(fd.name)}
        return "/api/tasks/cofold", payload, files

    def normalize_output(self, sub_task_id, downloaded_dir):
        ensemble_task_id = sub_task_id.split("__")[0]

        # promera writes per-structure confidence as ``<stem>_conf.json``
        # alongside ``<stem>.cif``.  Keys we care about: plddt, ptm, iptm.
        # See services/promera-server/adapter.py for the documented schema.
        structures: list[StructureFile] = []
        cif_files = sorted(
            cif for cif in downloaded_dir.rglob("*.cif")
            # Skip trajectory CIFs (multi-frame, not the final prediction).
            if not cif.name.endswith("_traj.cif")
        )
        for i, cif in enumerate(cif_files):
            rel = cif.relative_to(downloaded_dir)
            scores = _read_conf_json(cif)
            structures.append(StructureFile(
                rank=i,
                format="cif",
                url=f"/v1/jobs/{ensemble_task_id}/structures/{self.name}/{rel.as_posix()}",
                plddt=scores.get("plddt"),
                size_bytes=cif.stat().st_size,
            ))

        # Sort by plddt desc when available, else preserve discovery order.
        structures.sort(
            key=lambda s: (s.plddt if s.plddt is not None else -1.0),
            reverse=True,
        )
        # Re-assign ranks after sort so rank-0 is the best-scoring structure.
        for idx, s in enumerate(structures):
            s.rank = idx

        confidence: dict[str, float] = {}
        if structures and cif_files:
            top_scores = _read_conf_json(cif_files[0])
            for k, v in top_scores.items():
                confidence[k] = v

        return FoldingMethodResult(
            method=self.name,
            status="completed",
            runtime_seconds=None,
            fc_job_id=sub_task_id,
            structures=structures,
            confidence=confidence,
            metadata={},
        )

    def estimate_runtime_seconds(self, input):
        # ~3-4 min for a 75aa monomer at default sampling; scale with length.
        total = sum(len(s.sequence) for s in input.sequences) if input else 75
        return max(180, int(total * 3))


def _read_conf_json(cif: Path) -> dict[str, float]:
    """Read ``<stem>_conf.json`` next to a structure CIF; return float scores only."""
    conf = cif.with_name(f"{cif.stem}_conf.json")
    if not conf.is_file():
        return {}
    try:
        data = json.loads(conf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("plddt", "ptm", "iptm"):
        val = data.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out
