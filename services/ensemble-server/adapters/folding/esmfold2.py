"""ESMFold2 folding adapter.

Maps normalized FoldingInput → esmfold2-server's /api/tasks/fold form
request (sequences JSON string, no file upload).  Normalizes
`prediction_0.cif` + `metrics.json` outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ...folding.schemas import FoldingInput, FoldingMethodResult, StructureFile
from ...task_kind import TaskKind
from ..base import MethodAdapter


class ESMFold2Options(BaseModel):
    """Subset of esmfold2-server FoldRequest exposed to ensemble clients."""

    num_loops: int = Field(default=3, ge=1, le=20)
    num_sampling_steps: int = Field(default=50, ge=1, le=1000)
    num_diffusion_samples: int = Field(default=1, ge=1, le=50)
    seed: Optional[int] = None


class ESMFold2FoldingAdapter(MethodAdapter[FoldingInput, FoldingMethodResult]):
    name = "esmfold2"
    task_kind = TaskKind.FOLDING
    method_options_schema = ESMFold2Options

    def build_request(self, input, options):
        # esmfold2-server accepts sequences as a JSON string in form data.
        sequences = [
            {"type": s.type, "id": s.id, "sequence": s.sequence}
            for s in input.sequences
        ]
        payload = {
            "sequences": json.dumps(sequences),
            "num_loops": options.num_loops,
            "num_sampling_steps": options.num_sampling_steps,
            "num_diffusion_samples": options.num_diffusion_samples,
        }
        if options.seed is not None:
            payload["seed"] = options.seed
        return "/api/tasks/fold", payload, {}

    def normalize_output(self, sub_task_id, downloaded_dir):
        ensemble_task_id = sub_task_id.split("__")[0]
        structures: list[StructureFile] = []
        for cif in sorted(downloaded_dir.rglob("prediction_*.cif")):
            structures.append(StructureFile(
                rank=int(cif.stem.split("_")[-1]),
                format="cif",
                url=f"/v1/jobs/{ensemble_task_id}/structures/{self.name}/{cif.name}",
                plddt=None,
                size_bytes=cif.stat().st_size,
            ))
        structures.sort(key=lambda s: s.rank)

        # Try to extract plddt from metrics.json if available.
        confidence: dict[str, float] = {}
        metrics_files = list(downloaded_dir.rglob("metrics.json"))
        if metrics_files:
            try:
                metrics = json.loads(metrics_files[0].read_text())
                if isinstance(metrics, dict):
                    if "mean_plddt" in metrics:
                        confidence["mean_plddt"] = float(metrics["mean_plddt"])
                        if structures:
                            structures[0].plddt = confidence["mean_plddt"]
                    if "ptm" in metrics:
                        confidence["ptm"] = float(metrics["ptm"])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

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
        # 76aa ubiquitin ~80s; scales roughly linearly with length.
        total = sum(len(s.sequence) for s in input.sequences) if input else 100
        return max(60, int(total * 1.2))
