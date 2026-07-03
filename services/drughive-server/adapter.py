"""Service-wide policy: name + output detection + manifest_extras + endpoint_examples."""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import DrughiveSettings


class DrughiveAdapter(JobAdapter):
    name = "drughive"

    settings: DrughiveSettings

    def __init__(self, settings: DrughiveSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """True if any of the three modes' primary output SDF exists."""
        out = self.output_dir(job_dir)
        if not out.exists():
            return False
        for pattern in ("mols_gen_*.sdf", "mols_pred_*.sdf", "mols_initial_*.sdf"):
            for p in out.glob(pattern):
                if p.is_file() and p.stat().st_size > 0:
                    return True
        return False

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "id": self.settings.model_id,
                "checkpoint_filename": self.settings.checkpoint_filename,
                "paper": "Weller & Rohs, JCIM 2024 (10.1021/acs.jcim.4c01193)",
                "license": "USC-RL v2.0 (non-commercial academic research only)",
            },
            "tool_outputs": {
                "generate": "output/mols_gen_<pdb_id>.sdf",
                "generate_spatial": "output/mols_pred_<pdb_id>.sdf",
                "optimize": "output/mols_initial_<pdb_id>.sdf + mols_opt_<i>_<pdb_id>.sdf per cycle",
            },
            "input_uri_schemes": {
                "target": "multipart / oss:// / file:// / job:// / http(s)://",
                "ligand": "multipart / oss:// / file:// / job:// / http(s)://",
                "substruct_modify": "multipart / oss:// / file:// / job:// / http(s)://",
                "target_pdbqt": "multipart / oss:// / file:// / job:// / http(s)://",
            },
            "long_running": {
                "optimize": "single call may take 4-8 h; use /api/tasks/optimize "
                "(async task mode) rather than submit/poll",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/generate": [
                EndpointExample(
                    title="minimal de novo generation",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F target=@pocket.pdb "
                        "-F ligand=@ref_ligand.sdf "
                        '-F n_samples=10 -F pdb_id=5d3h'
                    ),
                    notes="zbetas defaults to [0,0,0,0] (prior); "
                    "increase toward 1 to move toward posterior.",
                ),
            ],
            "/api/generate_spatial": [
                EndpointExample(
                    title="scaffold hopping via SMARTS pattern",
                    curl=(
                        "curl -X POST $URL/api/generate_spatial "
                        "-F target=@pocket.pdb "
                        "-F ligand=@ref_ligand.sdf "
                        '-F substruct_modify_pattern="[CH2]C1:C:C:C:C:C:1" '
                        "-F n_samples=10 -F pdb_id=4w9f"
                    ),
                    notes="Preserves the SMARTS-matched substructure, "
                    "regenerates the rest.",
                ),
                EndpointExample(
                    title="scaffold hopping via SDF fragment",
                    curl=(
                        "curl -X POST $URL/api/generate_spatial "
                        "-F target=@pocket.pdb "
                        "-F ligand=@ref_ligand.sdf "
                        "-F substruct_modify=@fragment.sdf "
                        "-F n_samples=10 -F pdb_id=4w9f"
                    ),
                    notes="Alternative: upload the preserved fragment "
                    "as SDF instead of SMARTS pattern.",
                ),
            ],
            "/api/optimize": [
                EndpointExample(
                    title="quick QVina2 affinity optimization (test-friendly)",
                    curl=(
                        "curl -X POST $URL/api/tasks/optimize "
                        '-H "X-Fc-Invocation-Type: Async" '
                        "-F target=@pocket.pdb "
                        "-F ligand=@ref_ligand.sdf "
                        "-F target_pdbqt=@pocket.pdbqt "
                        "-F key_opt=affinity_qvina "
                        "-F n_cycles=2 -F n_samples_initial=20 -F n_samples=4"
                    ),
                    notes="Use /api/tasks/optimize (async mode) — default "
                    "params (8 cycles × 1000 initial × 20) take 4-8 h.",
                ),
            ],
        }
