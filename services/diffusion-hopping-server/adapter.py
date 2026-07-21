"""Service-wide policy for diffusion-hopping-server."""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import DiffusionHoppingSettings


class DiffusionHoppingAdapter(JobAdapter):
    name = "diffusion-hopping"

    settings: DiffusionHoppingSettings  # narrow for IDEs

    def __init__(self, settings: DiffusionHoppingSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Job is `completed` only if at least one valid .sdf was written."""
        out_dir = self.output_dir(job_dir)
        if not out_dir.exists():
            return False
        for p in out_dir.glob("output_*.sdf"):
            if p.is_file() and p.stat().st_size > 0:
                return True
        return False

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        # DiffHopp's ObabelTransform / ReduceTransform write temp files to
        # `self.tmpdir` (resolved from CWD), so pin CWD to a writable dir.
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "DiffHopp",
                "paper": "arXiv:2308.07416",
                "task": "scaffold hopping (de novo molecule design via graph diffusion)",
                "variants": [
                    "gvp_conditional",
                    "gvp_unconditional",
                    "egnn_conditional",
                    "egnn_unconditional",
                ],
                "output_format": "SDF (one molecule per file: output_<i>.sdf)",
            },
            "tool_outputs": {
                "generate": "output/output_<i>.sdf — one SDF per generated scaffold (i in [0, num_samples)). "
                "Files where the generated graph couldn't be assembled into a valid Mol are omitted.",
            },
            "config_tips": {
                "num_samples": "10 is a reasonable default; sampling is batched, so 50 takes ~4× a single sample, not 50×.",
                "model_variant": "Start with gvp_conditional (DiffHopp main, paper's primary result). "
                "Switch to *_unconditional for inpainting scenarios where the reference ligand's functional group should NOT bias generation.",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data — protein (.pdb) and reference ligand (.sdf/.mol2/.pdb)",
                "oss://<bucket>/<key>": "fetched at submit-time via configured OSS region",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/generate": [
                EndpointExample(
                    title="basic scaffold hopping (10 candidates, GVP-conditional)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F protein=@1abc_pocket.pdb "
                        "-F reference_ligand=@ref.sdf "
                        "-F num_samples=10 "
                        "-F model_variant=gvp_conditional"
                    ),
                    notes="Smallest useful call. Returns a JobInfo; poll /api/jobs/<id> until completed, then GET /api/jobs/<id>/files.",
                ),
                EndpointExample(
                    title="EGNN variant with inpainting (no functional-group conditioning)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F protein=@pocket.pdb "
                        "-F reference_ligand=@ref.mol2 "
                        "-F num_samples=20 "
                        "-F model_variant=egnn_unconditional"
                    ),
                    notes="Use *_unconditional variants when you want the model to ignore the reference ligand's functional groups.",
                ),
            ],
            "/api/tasks/generate": [
                EndpointExample(
                    title="FC async task mode (single 202 + blocking compute)",
                    curl=(
                        "curl -X POST $URL/api/tasks/generate "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: my-job-001' "
                        "-F protein=@pocket.pdb "
                        "-F reference_ligand=@ref.sdf "
                        "-F num_samples=10"
                    ),
                    notes="Returns 202 immediately; FC keeps the instance alive until the diffusion completes. Use this for production calls behind the FC async-task console toggle.",
                ),
            ],
        }
