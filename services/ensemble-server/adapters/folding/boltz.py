"""Boltz folding adapter.

Maps normalized FoldingInput → boltz-server's /api/tasks/predict_structure
form request (sequences JSON, with msa_mode + boltz-specific knobs).
Phase-1 MVP does not pass MSA files or templates; those are TODO.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ...folding.schemas import FoldingInput, FoldingMethodResult, StructureFile
from ...task_kind import TaskKind
from ..base import MethodAdapter


class BoltzOptions(BaseModel):
    """Subset of boltz-server PredictStructureRequest exposed to ensemble clients."""

    recycling_steps: int = Field(default=3, ge=1, le=20)
    sampling_steps: int = Field(default=200, ge=10, le=1000)
    diffusion_samples: int = Field(default=1, ge=1, le=100)
    seed: Optional[int] = None
    name: str = "ensemble"


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
            "name": options.name,
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

        # Boltz output structure: predictions/<name>/<seed>_model_*.cif + confidence.json
        structures: list[StructureFile] = []
        cif_files = sorted(downloaded_dir.rglob("*.cif"))
        for i, cif in enumerate(cif_files):
            structures.append(StructureFile(
                rank=i,
                format="cif",
                url=f"/v1/jobs/{ensemble_task_id}/structures/{self.name}/{cif.name}",
                plddt=None,
                size_bytes=cif.stat().st_size,
            ))

        # Try to extract confidence from any *confidence*.json file.
        confidence: dict[str, float] = {}
        for cj in downloaded_dir.rglob("*confidence*.json"):
            try:
                data = json.loads(cj.read_text())
                if isinstance(data, dict):
                    for key in ("plddt", "ptm", "iptm", "confidence_score"):
                        if key in data and isinstance(data[key], (int, float)):
                            confidence[key] = float(data[key])
                    if "plddt" in confidence and structures:
                        structures[0].plddt = confidence["plddt"]
                    break
            except (json.JSONDecodeError, ValueError):
                continue

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
