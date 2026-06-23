"""Boltz folding adapter.

Maps normalized FoldingInput → boltz-server's /api/tasks/predict_structure
form request (sequences JSON, with msa_mode + boltz-specific knobs).
Phase-1 MVP does not pass MSA files or templates; those are TODO.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ...folding.schemas import FoldingInput, FoldingMethodResult, StructureFile
from ...task_kind import TaskKind
from .._canonical import publish_canonical
from ..base import MethodAdapter


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _find_confidence_json(cif: Path) -> Optional[Path]:
    """Locate ``confidence_<stem>_model_<N>.json`` sibling for a given CIF.

    Boltz writes it next to the structure file; fall back to any
    ``confidence_*.json`` in the same directory when the strict pairing
    doesn't match (defensive against minor boltz version differences).
    """
    sibling = cif.with_name(f"confidence_{cif.stem}.json")
    if sibling.is_file():
        return sibling
    for candidate in cif.parent.glob("confidence_*.json"):
        return candidate
    return None


def _read_complex_plddt(cif: Path) -> Optional[float]:
    """Extract ``complex_plddt`` from the per-structure confidence JSON."""
    conf_json = _find_confidence_json(cif)
    if conf_json is None:
        return None
    val = _safe_load_json(conf_json).get("complex_plddt")
    return float(val) if isinstance(val, (int, float)) else None


class BoltzOptions(BaseModel):
    """Subset of boltz-server PredictStructureRequest exposed to ensemble clients."""

    recycling_steps: int = Field(default=3, ge=1, le=20)
    sampling_steps: int = Field(default=200, ge=10, le=1000)
    diffusion_samples: int = Field(default=1, ge=1, le=100)
    seed: Optional[int] = None


# Match `predictions/<stem>/<stem>_model_<N>.cif` (the only well-defined output
# filename in boltz-server — see services/boltz-server/tools.py:7).  Capturing
# the model index lets us pair each CIF with its sibling `confidence_<stem>_model_<N>.json`.
_BOLTZ_MODEL_RE = re.compile(r"^(?P<stem>.+)_model_(?P<idx>\d+)\.cif$")


class BoltzFoldingAdapter(MethodAdapter[FoldingInput, FoldingMethodResult]):
    name = "boltz"
    task_kind = TaskKind.FOLDING
    method_options_schema = BoltzOptions

    def build_request(self, input, options):
        # boltz-server expects sequences as JSON in a form field, with msa_mode.
        # Each sequence entry needs a msa_uri="empty" when msa_mode is "empty"
        # to skip the per-chain MSA lookup.
        sequences = []
        for s in input.sequences:
            entry = {"type": s.type, "id": s.id, "sequence": s.sequence}
            if input.msa_mode == "empty":
                entry["msa_uri"] = "empty"
            sequences.append(entry)

        payload = {
            "msa_mode": input.msa_mode,
            "sequences": json.dumps(sequences),
            "recycling_steps": options.recycling_steps,
            "sampling_steps": options.sampling_steps,
            "diffusion_samples": options.diffusion_samples,
        }
        if options.seed is not None:
            payload["seed"] = options.seed
        return "/api/tasks/predict_structure", payload, {}

    def normalize_output(self, sub_task_id, downloaded_dir):
        ensemble_task_id = sub_task_id.split("__")[0]

        # Boltz output: predictions/<stem>/<stem>_model_<N>.cif plus
        # confidence_<stem>_model_<N>.json alongside.  Pair them, sort by
        # model index, then republish each as <method>/rank_<i>.cif.
        raw_cifs = sorted(downloaded_dir.rglob("*.cif"))

        def _model_idx(p: Path) -> int:
            m = _BOLTZ_MODEL_RE.match(p.name)
            return int(m.group("idx")) if m else 0

        raw_cifs.sort(key=_model_idx)

        structures: list[StructureFile] = []
        for i, cif in enumerate(raw_cifs):
            url, original_filename = publish_canonical(
                src=cif, downloaded_dir=downloaded_dir,
                ensemble_task_id=ensemble_task_id, method=self.name,
                rank=i, format="cif",
            )
            structures.append(StructureFile(
                rank=i,
                format="cif",
                url=url,
                plddt=_read_complex_plddt(cif),
                size_bytes=cif.stat().st_size,
                original_filename=original_filename,
            ))

        # Top-level confidence (rank-0 model's scores), exposed for clients
        # that want to inspect ptm/iptm/confidence_score in addition to plddt.
        confidence: dict[str, float] = {}
        if raw_cifs:
            conf_json = _find_confidence_json(raw_cifs[0])
            if conf_json is not None:
                data = _safe_load_json(conf_json)
                for key in ("complex_plddt", "ptm", "iptm", "confidence_score"):
                    if isinstance(data.get(key), (int, float)):
                        confidence[key] = float(data[key])

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
        # 56aa peptide complex ~100s; scales roughly linearly.
        total = sum(len(s.sequence) for s in input.sequences) if input else 100
        return max(120, int(total * 2))
