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
from .._canonical import publish_canonical
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

        # esmfold2-server writes metrics.json with shape
        # {"samples": [{"sample_index": i, "output_file": "prediction_i.cif",
        #               "plddt_mean": ..., "ptm": ..., "iptm": ...}, ...]}
        # — see services/esmfold2-server/inference.py.
        per_file_scores: dict[str, dict[str, float]] = {}
        for mj in downloaded_dir.rglob("metrics.json"):
            try:
                metrics = json.loads(mj.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if not isinstance(metrics, dict):
                continue
            for sample in metrics.get("samples", []):
                if not isinstance(sample, dict):
                    continue
                output_file = sample.get("output_file")
                if not output_file:
                    continue
                scores: dict[str, float] = {}
                for key in ("plddt_mean", "ptm", "iptm"):
                    val = sample.get(key)
                    if isinstance(val, (int, float)):
                        scores[key] = float(val)
                if scores:
                    per_file_scores[output_file] = scores
            break  # first metrics.json wins

        # Sort by the prediction index first so rank assignment is stable, then
        # publish each to <method>/rank_<i>.cif before constructing StructureFile.
        ranked_cifs = sorted(
            downloaded_dir.rglob("prediction_*.cif"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        structures: list[StructureFile] = []
        for i, cif in enumerate(ranked_cifs):
            scores = per_file_scores.get(cif.name, {})
            url, original_filename = publish_canonical(
                src=cif, downloaded_dir=downloaded_dir,
                ensemble_task_id=ensemble_task_id, method=self.name,
                rank=i, format="cif",
            )
            structures.append(StructureFile(
                rank=i,
                format="cif",
                url=url,
                plddt=scores.get("plddt_mean"),
                size_bytes=cif.stat().st_size,
                original_filename=original_filename,
            ))

        # Top-level confidence = rank-0 sample's scores (looked up by the
        # raw upstream filename, since URL paths are now canonicalized).
        confidence: dict[str, float] = {}
        if ranked_cifs:
            for k, v in per_file_scores.get(ranked_cifs[0].name, {}).items():
                confidence[k] = v
            if "plddt_mean" in confidence:
                # Convenience alias preferred by aggregator + clients.
                confidence.setdefault("mean_plddt", confidence["plddt_mean"])

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
