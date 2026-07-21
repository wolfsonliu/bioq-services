"""ReinventAdapter — output detection + manifest metadata."""
from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .config_builder import PRIOR_FILES
from .settings import ReinventSettings

# Files reinvent_cli copies to output/ for diagnosis even on FAILURE — must NOT
# count as a successful result. Everything else non-empty in output/ is a real
# artifact (sampling.csv / score_results.csv / peptide_enumeration.csv / *.model
# / *.chkpt / <prefix>_<n>.csv). `_*.json` are reinvent's config echoes.
_AUDIT_FILES = {"config.toml", "reinvent.log"}


class ReinventAdapter(JobAdapter):
    name = "reinvent"

    settings: ReinventSettings

    def __init__(self, settings: ReinventSettings) -> None:
        super().__init__(settings)

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def detect_outputs(self, job_dir: Path) -> bool:
        """Label-agnostic: any non-empty result file in output/ means success.

        The framework calls this with only `job_dir` (no label; it never writes a
        per-job manifest), so detection cannot depend on the run mode. Excludes
        the audit files reinvent_cli copies on failure so a crashed job isn't
        mistaken for a successful one.
        """
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for p in out.iterdir():
            if (p.is_file() and p.stat().st_size > 0
                    and p.name not in _AUDIT_FILES
                    and not p.name.startswith("_")):
                return True
        return False

    def manifest_extras(self) -> dict:
        return {
            "run_modes": ["sampling", "scoring", "enumeration",
                          "transfer_learning", "staged_learning"],
            "generators": ["reinvent", "libinvent", "linkinvent", "mol2mol", "pepinvent"],
            "prior_registry": dict(PRIOR_FILES),
            "prior_base": str(self.settings.prior_base),
            "scoring_backends": ["rdkit", "pumas", "chemprop2", "mmpdb", "apted", "isim"],
            "tool_outputs": {
                "sampling": "output/sampling.csv",
                "scoring": "output/score_results.csv",
                "enumeration": "output/peptide_enumeration.csv",
                "transfer_learning": "output/*.model",
                "staged_learning": "output/*_1.csv",
            },
            "input_uri_schemes": {"upload": "multipart/form-data"},
            "task_endpoints": ("enabled (FC async for sampling/scoring/enumeration; "
                               "long RL/TL should use HPC sbatch)"),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/sampling": [EndpointExample(
                title="De novo sampling from the Reinvent prior",
                curl=("curl -X POST $URL/api/sampling "
                      "-F generator=reinvent -F num_smiles=100"),
                notes="No seed SMILES for Reinvent generator; produces sampling.csv.",
            )],
            "/api/scoring": [EndpointExample(
                title="Score SMILES with a scoring function",
                curl=("curl -X POST $URL/api/scoring "
                      "-F 'scoring={\"type\":\"geometric_mean\",\"component\":[]}' "
                      "-F smiles_file=@compounds.smi"),
                notes="scoring is a JSON-encoded form field; see configs/SCORING.md.",
            )],
            "/api/enumeration": [EndpointExample(
                title="Peptide enumeration",
                curl=("curl -X POST $URL/api/enumeration "
                      "-F 'scoring={\"type\":\"geometric_mean\",\"component\":[]}' "
                      "-F peptide_smiles=@peptide.smi -F amino_acid_library=@library.csv"),
                notes="",
            )],
            "/api/transfer-learning": [EndpointExample(
                title="Fine-tune a prior on target SMILES",
                curl=("curl -X POST $URL/api/transfer-learning "
                      "-F generator=reinvent -F num_epochs=5 -F smiles_file=@target.smi"),
                notes="Long-running; prefer CLI/sbatch for large datasets.",
            )],
            "/api/staged-learning": [EndpointExample(
                title="Reinforcement learning (curriculum stages)",
                curl=("curl -X POST $URL/api/staged-learning -F generator=reinvent "
                      "-F 'stages=[{\"chkpt_name\":\"s1.chkpt\",\"scoring\":"
                      "{\"type\":\"geometric_mean\",\"component\":[]}}]'"),
                notes="stages + diversity_filter + inception are JSON-encoded fields. "
                      "Long RL should run via HPC sbatch with checkpoint chaining.",
            )],
        }
