"""Service-wide policy for megalodon-server."""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import MegalodonSettings


class MegalodonAdapter(JobAdapter):
    name = "megalodon"

    settings: MegalodonSettings  # narrow for IDEs

    def __init__(self, settings: MegalodonSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Completed only if the SDF exists and is non-empty (>= 1 mol)."""
        sdf = self.output_dir(job_dir) / "generated_molecules.sdf"
        return sdf.exists() and sdf.stat().st_size > 0

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "Megalodon",
                "paper": "Reidenbach et al., arXiv:2505.18392 (2025) — "
                "EGNN-Transformer co-design, diffusion + flow matching",
                "task": "unconditional 3D de novo small-molecule generation",
                "models": list(MODEL_REGISTRY_NAMES),
                "output_format": (
                    "generated_molecules.sdf (RDKit SDF, N molecules) + "
                    "metrics.json (2D/3D/train-data) + generation_stats.json"
                ),
                "training_data": "QM9 / GEOM-Drugs",
                "license": "Apache-2.0 (code) / NVIDIA Open Model License (weights)",
            },
            "tool_outputs": {
                "generate": (
                    "output/generated_molecules.sdf — up to n_molecules valid "
                    "3D molecules (SDF). Actual count <= n_molecules "
                    "(unbuildable graphs skipped). output/metrics.json reports "
                    "generative metrics; output/generation_stats.json reports "
                    "n_valid vs n_requested + timing."
                ),
            },
            "config_tips": {
                "model_name": "drugs_* = drug-like (GEOM-Drugs); qm9_* = small "
                "molecules. *_diffusion = highest quality; *_fm = flow matching "
                "(fast at ~100 steps); *_quick = lighter/faster architecture.",
                "n_atoms_per_mol": "null samples atom counts from the training "
                "distribution (recommended); set an int to fix molecule size.",
                "timesteps": "500 for diffusion; ~100 for fm variants.",
            },
            "input_uri_schemes": {
                "none": "unconditional — no receptor/ligand/SMILES inputs",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/generate": [
                EndpointExample(
                    title="basic generation (100 mols, drugs diffusion)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F model_name=drugs_diffusion -F n_molecules=100"
                    ),
                    notes="Returns a JobInfo; poll /api/jobs/<id> until "
                    "completed, then GET /api/jobs/<id>/file/generated_molecules.sdf.",
                ),
                EndpointExample(
                    title="fast smoke (10 mols, fm variant, 100 steps)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F model_name=drugs_fm -F n_molecules=10 -F timesteps=100"
                    ),
                    notes="Fastest useful call for CI / warm-up.",
                ),
                EndpointExample(
                    title="fixed molecule size (25 atoms each)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F model_name=qm9_diffusion -F n_molecules=50 "
                        "-F n_atoms_per_mol=25 -F seed=42"
                    ),
                    notes="n_atoms_per_mol pins every molecule to a fixed size.",
                ),
            ],
            "/api/tasks/generate": [
                EndpointExample(
                    title="FC async task mode (single 202 + blocking compute)",
                    curl=(
                        "curl -X POST $URL/api/tasks/generate "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: my-job-001' "
                        "-F model_name=drugs_diffusion -F n_molecules=100"
                    ),
                    notes="Returns 202 immediately; FC keeps the instance "
                    "alive until sampling completes. Preferred production entry.",
                ),
            ],
        }


# Imported lazily to avoid a hard import cycle at module import time.
from .models import MODEL_REGISTRY as _REG  # noqa: E402

MODEL_REGISTRY_NAMES = tuple(_REG.keys())

__all__ = ["MegalodonAdapter"]
