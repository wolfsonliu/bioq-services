"""Service-wide policy for semlaflow-server."""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import SemlaFlowSettings


class SemlaFlowAdapter(JobAdapter):
    name = "semlaflow"

    settings: SemlaFlowSettings  # narrow for IDEs

    def __init__(self, settings: SemlaFlowSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Job is `completed` only if the SDF is non-empty (>= 1 mol).

        Upstream `save_rdkit_sdf` writes `<save_dir>/<save_file>.sdf` where
        save_file defaults to `predictions.smol`, so the file is
        `predictions.smol.sdf`.
        """
        sdf = self.output_dir(job_dir) / "predictions.smol.sdf"
        return sdf.exists() and sdf.stat().st_size > 0

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "SemlaFlow",
                "paper": "Irwin et al. 2024 (Semla + equivariant OT flow matching)",
                "task": "unconditional 3D de novo small-molecule generation",
                "models": list(self._model_names()),
                "output_format": (
                    "predictions.smol.sdf (RDKit SDF, N molecules) + "
                    "metrics.json (validity/uniqueness/novelty/energy/...)"
                ),
                "training_data": "QM9 / GEOM-Drugs",
            },
            "tool_outputs": {
                "generate": (
                    "output/predictions.smol.sdf — up to n_molecules valid 3D "
                    "molecules (SDF). Actual count ≤ n_molecules (unbuildable "
                    "graphs skipped). output/metrics.json reports generative "
                    "metrics; output/generation_stats.json reports n_valid vs "
                    "n_requested."
                ),
            },
            "config_tips": {
                "model_name": "qm9 is fast; geom-drugs is drug-like but its "
                "novelty metric is slow (builds ~300k RDKit reference mols).",
                "n_molecules": "100 is a good default.",
                "integration_steps": "100 is the upstream default; drop to 50 "
                "for quick smoke checks.",
            },
            "input_uri_schemes": {
                "none": "unconditional — no receptor/ligand/SMILES inputs",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/generate": [
                EndpointExample(
                    title="basic generation (100 mols, qm9)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F model_name=qm9 -F n_molecules=100"
                    ),
                    notes="Returns a JobInfo; poll /api/jobs/<id> until "
                    "completed, then GET /api/jobs/<id>/file/predictions.smol.sdf.",
                ),
                EndpointExample(
                    title="fast smoke (10 mols, 50 steps)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F model_name=qm9 -F n_molecules=10 "
                        "-F integration_steps=50"
                    ),
                    notes="Fastest useful call for CI / warm-up.",
                ),
                EndpointExample(
                    title="drug-like generation (geom-drugs)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F model_name=geom-drugs -F n_molecules=100 -F seed=42"
                    ),
                    notes="Slower — geom-drugs novelty builds a large RDKit "
                    "reference set. Prefer the async task endpoint.",
                ),
            ],
            "/api/tasks/generate": [
                EndpointExample(
                    title="FC async task mode (single 202 + blocking compute)",
                    curl=(
                        "curl -X POST $URL/api/tasks/generate "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: my-job-001' "
                        "-F model_name=qm9 -F n_molecules=100"
                    ),
                    notes="Returns 202 immediately; FC keeps the instance "
                    "alive until sampling completes. Preferred production entry.",
                ),
            ],
        }

    # ---- helpers ----

    def _model_names(self) -> list[str]:
        return [m.name for m in self.settings.list_models()]
