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

        cif_files = sorted(
            cif for cif in downloaded_dir.rglob("*.cif")
            # Skip trajectory CIFs (multi-frame diagnostic, not the final prediction).
            if not cif.name.endswith("_traj.cif")
        )

        # Build a (cif, scalar_scores, chain_plddt) triple for each prediction.
        per_cif: list[tuple[Path, dict[str, float], dict[str, float]]] = []
        for cif in cif_files:
            raw = _read_conf_raw(cif)
            per_cif.append((cif, _extract_scalar_scores(raw), _extract_chain_plddt(raw)))

        # Sort by complex_plddt desc; missing-plDDT structures land at the end.
        per_cif.sort(
            key=lambda t: (t[1].get("plddt", -1.0)),
            reverse=True,
        )

        structures: list[StructureFile] = []
        for idx, (cif, scores, _) in enumerate(per_cif):
            rel = cif.relative_to(downloaded_dir)
            structures.append(StructureFile(
                rank=idx,
                format="cif",
                url=f"/v1/jobs/{ensemble_task_id}/structures/{self.name}/{rel.as_posix()}",
                plddt=scores.get("plddt"),
                size_bytes=cif.stat().st_size,
            ))

        confidence: dict[str, float] = {}
        metadata: dict[str, Any] = {}
        if per_cif:
            _, top_scores, top_chain_plddt = per_cif[0]
            confidence.update(top_scores)
            if top_chain_plddt:
                metadata["chain_plddt"] = top_chain_plddt

        return FoldingMethodResult(
            method=self.name,
            status="completed",
            runtime_seconds=None,
            fc_job_id=sub_task_id,
            structures=structures,
            confidence=confidence,
            metadata=metadata,
        )

    def estimate_runtime_seconds(self, input):
        # ~3-4 min for a 75aa monomer at default sampling; scale with length.
        total = sum(len(s.sequence) for s in input.sequences) if input else 75
        return max(180, int(total * 3))


def _read_conf_raw(cif: Path) -> dict[str, Any]:
    """Read the raw ``<stem>_conf.json`` next to a structure CIF; ``{}`` if missing."""
    conf = cif.with_name(f"{cif.stem}_conf.json")
    if not conf.is_file():
        return {}
    try:
        data = json.loads(conf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _extract_scalar_scores(raw: dict[str, Any]) -> dict[str, float]:
    """Pull scalar confidence fields from a promera conf JSON.

    Actual promera schema (verified against a v0.0.8 run, NOT the
    single-`plddt` form that promera-server's adapter docstring suggests):

      {
        "complex_plddt": float,          # top-level mean plddt
        "complex_ptm":   float,          # top-level mean ptm
        "chain_plddt":   {chain: float}, # per-chain plddt  (returned separately)
        "ptm":           {chain: float}, # per-chain ptm   (DICT, not scalar)
        "iCS":           {...}
      }

    Mapping into our flat scalar dict:
      - ``plddt``  := complex_plddt   (this is what StructureFile.plddt uses)
      - ``ptm``    := complex_ptm
      - ``iptm``   := iptm if present as scalar
    """
    out: dict[str, float] = {}
    if isinstance(raw.get("complex_plddt"), (int, float)):
        out["plddt"] = float(raw["complex_plddt"])
    if isinstance(raw.get("complex_ptm"), (int, float)):
        out["ptm"] = float(raw["complex_ptm"])
    if isinstance(raw.get("iptm"), (int, float)):
        out["iptm"] = float(raw["iptm"])
    return out


def _extract_chain_plddt(raw: dict[str, Any]) -> dict[str, float]:
    """Pull per-chain plddt from a promera conf JSON (empty if absent)."""
    chain_plddt = raw.get("chain_plddt")
    if not isinstance(chain_plddt, dict):
        return {}
    return {
        str(k): float(v) for k, v in chain_plddt.items()
        if isinstance(v, (int, float))
    }
