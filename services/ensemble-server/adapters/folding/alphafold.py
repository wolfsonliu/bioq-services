"""AlphaFold v2.3.2 folding adapter.

Maps normalized FoldingInput → alphafold-server's /api/tasks/fold multipart
request (FASTA upload + form options).  Normalizes outputs: `ranked_*.pdb`
files become StructureFile list ordered by AlphaFold's own ranking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ...folding.schemas import FoldingInput, FoldingMethodResult, StructureFile
from ...task_kind import TaskKind
from ..base import MethodAdapter


class AlphaFoldOptions(BaseModel):
    """Subset of alphafold-server FoldRequest exposed to ensemble clients."""

    model_preset: Literal["monomer", "monomer_casp14", "monomer_ptm", "multimer"] = "monomer_ptm"
    db_preset: Literal["reduced_dbs", "full_dbs"] = "reduced_dbs"
    models_to_relax: Literal["all", "best", "none"] = "best"


class AlphaFoldFoldingAdapter(MethodAdapter[FoldingInput, FoldingMethodResult]):
    name = "alphafold"
    task_kind = TaskKind.FOLDING
    method_options_schema = AlphaFoldOptions

    def build_request(self, input, options):
        # alphafold-server takes a FASTA UploadFile (input_fasta) + form fields.
        fasta = "".join(
            f">{s.id}\n{s.sequence}\n" for s in input.sequences
        )
        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False, encoding="utf-8",
        )
        fd.write(fasta)
        fd.close()
        payload = {
            "model_preset": options.model_preset,
            "db_preset": options.db_preset,
            "models_to_relax": options.models_to_relax,
        }
        files = {"input_fasta": Path(fd.name)}
        return "/api/tasks/fold", payload, files

    def normalize_output(self, sub_task_id, downloaded_dir):
        # alphafold output: ranked_0.pdb..ranked_4.pdb (+ ranking_debug.json with plddt).
        # task_id pattern: <ensemble_task_id>__<method>.  Use the ensemble task_id
        # to build URLs that route through ensemble-server's download endpoint.
        ensemble_task_id = sub_task_id.split("__")[0]
        structures: list[StructureFile] = []
        ranked = sorted(downloaded_dir.rglob("ranked_*.pdb"))
        for i, pdb in enumerate(ranked):
            structures.append(StructureFile(
                rank=i,
                format="pdb",
                url=f"/v1/jobs/{ensemble_task_id}/structures/{self.name}/{pdb.name}",
                plddt=None,  # Phase-1 simplification; Phase-2 parse from ranking_debug.json
                size_bytes=pdb.stat().st_size,
            ))
        return FoldingMethodResult(
            method=self.name,
            status="completed",
            runtime_seconds=None,  # filled by orchestrator
            fc_job_id=sub_task_id,
            structures=structures,
            confidence={},
            metadata={"model_preset": "monomer_ptm"},
        )

    def estimate_runtime_seconds(self, input):
        # MSA + 5 models, dominated by reduced_dbs database query.
        return 2000
