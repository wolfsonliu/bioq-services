"""Service-wide policy for flowmol-server."""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import FlowMolSettings


# Variants pre-staged to NAS by default fetch_weights.sh; other 18 ablations
# are loadable if manually staged. The /healthz/detail probe verifies these.
PRIMARY_VARIANTS = ("flowmol3", "fm3_nodistort", "fm3_none", "fm3_ahigh")


class FlowMolAdapter(JobAdapter):
    name = "flowmol"

    settings: FlowMolSettings  # narrow for IDEs

    def __init__(self, settings: FlowMolSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Job is `completed` only if molecules.sdf is non-empty (>= 1 mol)."""
        sdf = self.output_dir(job_dir) / "molecules.sdf"
        return sdf.exists() and sdf.stat().st_size > 0

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "FlowMol3",
                "paper": "arXiv:2508.12629",
                "task": "unconditional 3D de novo small-molecule generation (flow matching)",
                "primary_variants": list(PRIMARY_VARIANTS),
                "output_format": "single SDF with N molecules (kekulize=False)",
                "training_data": "GEOM-Drugs (243k drug-like molecules)",
                "params": "6M",
            },
            "tool_outputs": {
                "generate": (
                    "output/molecules.sdf — up to n_mols valid 3D molecules "
                    "(SDF, kekulize=False). Actual count ≤ n_mols since "
                    "unbuildable graphs are skipped. "
                    "output/sampling_stats.json reports n_written vs n_requested."
                ),
            },
            "config_tips": {
                "n_mols": "100 is a good sweet spot; batched sampling scales "
                "sublinearly.",
                "n_timesteps": "250 is paper SOTA; drop to 100 for quick "
                "smoke checks (small %PB-Valid loss).",
                "model_variant": "Start with flowmol3 (paper SOTA). "
                "Ablation variants are for research; pre-stage them on NAS "
                "first.",
            },
            "input_uri_schemes": {
                "none": "unconditional — no receptor/ligand inputs",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/generate": [
                EndpointExample(
                    title="basic generation (100 mols, default flowmol3 variant)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F n_mols=100 -F n_timesteps=250"
                    ),
                    notes="Smallest useful call. Returns a JobInfo; poll "
                    "/api/jobs/<id> until completed, then GET "
                    "/api/jobs/<id>/file/molecules.sdf.",
                ),
                EndpointExample(
                    title="fast smoke (10 mols, 100 steps)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F n_mols=10 -F n_timesteps=100"
                    ),
                    notes="Roughly 5-10 s on T4; useful for CI / warm-up.",
                ),
                EndpointExample(
                    title="fixed molecule size (25 atoms each)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F n_mols=50 -F n_atoms_per_mol=25 -F seed=42"
                    ),
                    notes="Pins every generated molecule to exactly N atoms.",
                ),
                EndpointExample(
                    title="ablation variant (no geometry distortion)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F n_mols=100 -F model_variant=fm3_nodistort"
                    ),
                    notes="Requires fm3_nodistort pre-staged to NAS "
                    "(check /healthz/detail.staged_variants).",
                ),
            ],
            "/api/tasks/generate": [
                EndpointExample(
                    title="FC async task mode (single 202 + blocking compute)",
                    curl=(
                        "curl -X POST $URL/api/tasks/generate "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: my-job-001' "
                        "-F n_mols=100 -F n_timesteps=250"
                    ),
                    notes="Returns 202 immediately; FC keeps the instance "
                    "alive until sampling completes. Preferred production entry.",
                ),
            ],
        }
