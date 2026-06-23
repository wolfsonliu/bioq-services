"""AlphaFold v2.3.2 folding adapter.

Maps normalized FoldingInput → alphafold-server's /api/tasks/fold multipart
request (FASTA upload + form options).  Normalizes outputs: `ranked_*.pdb`
files become StructureFile list ordered by AlphaFold's own ranking.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ...folding.schemas import FoldingInput, FoldingMethodResult, StructureFile
from ...task_kind import TaskKind
from .._canonical import publish_canonical
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
        # alphafold-server's pipeline writes outputs into ``output/input/``
        # (the dir name is the FASTA stem, which is fixed to ``input`` by the
        # service; see services/alphafold-server/app.py:62).  The orchestrator
        # unzips into our downloaded_dir, stripping the ``output/`` root, so
        # files end up at ``<downloaded_dir>/input/ranked_<N>.pdb``.  We
        # republish each via publish_canonical so the public URL is
        # ``<method>/rank_<i>.pdb`` regardless of upstream layout.
        ensemble_task_id = sub_task_id.split("__")[0]

        # Parse ranking_debug.json once so we can attach per-model plDDT to
        # the right ranked_<N>.pdb.  Schema:
        #   {"order": [model_name, ...],          # sorted best→worst
        #    "plddts": {model_name: float, ...}}
        #
        # AlphaFold reports plDDT on the **0-100 scale** (e.g. 75.62) while
        # esmfold2 / boltz / promera all report **0-1** (e.g. 0.76).  The
        # ensemble aggregator ranks by raw score, so without normalization
        # alphafold would always dominate purely by units.  We rescale to
        # 0-1 here so cross-method ranking is meaningful.
        per_rank_plddt: dict[int, float] = {}
        for rd in downloaded_dir.rglob("ranking_debug.json"):
            try:
                data = json.loads(rd.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            order = data.get("order") or []
            plddts = data.get("plddts") or {}
            for rank_idx, model_name in enumerate(order):
                val = plddts.get(model_name)
                if isinstance(val, (int, float)):
                    per_rank_plddt[rank_idx] = float(val) / 100.0
            break

        structures: list[StructureFile] = []
        ranked = sorted(downloaded_dir.rglob("ranked_*.pdb"))
        for i, pdb in enumerate(ranked):
            url, original_filename = publish_canonical(
                src=pdb, downloaded_dir=downloaded_dir,
                ensemble_task_id=ensemble_task_id, method=self.name,
                rank=i, format="pdb",
            )
            structures.append(StructureFile(
                rank=i,
                format="pdb",
                url=url,
                plddt=per_rank_plddt.get(i),
                size_bytes=pdb.stat().st_size,
                original_filename=original_filename,
            ))

        confidence: dict[str, float] = {}
        if structures and structures[0].plddt is not None:
            confidence["plddt"] = structures[0].plddt

        return FoldingMethodResult(
            method=self.name,
            status="completed",
            runtime_seconds=None,  # filled by orchestrator
            fc_job_id=sub_task_id,
            structures=structures,
            confidence=confidence,
            metadata={"model_preset": "monomer_ptm"},
        )

    def estimate_runtime_seconds(self, input):
        # MSA + 5 models, dominated by reduced_dbs database query.
        return 2000
