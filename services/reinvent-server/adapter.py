"""ReinventAdapter — per-label detect_outputs + manifest metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from bioagent_service import EndpointExample, JobAdapter

from .config_builder import PRIOR_FILES
from .settings import ReinventSettings


class ReinventAdapter(JobAdapter):
    name = "reinvent"

    settings: ReinventSettings

    def __init__(self, settings: ReinventSettings) -> None:
        super().__init__(settings)

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def _read_label(self, job_dir: Path) -> str | None:
        try:
            return json.loads((job_dir / "manifest.json").read_text()).get("label")
        except (FileNotFoundError, ValueError):
            return None

    def detect_outputs(self, job_dir: Path) -> bool:
        label = self._read_label(job_dir)
        checkers: dict[str, Callable[[Path], bool]] = {
            "sampling":          self._csv("sampling.csv"),
            "scoring":           self._csv("score_results.csv"),
            "enumeration":       self._csv("peptide_enumeration.csv"),
            "transfer_learning": self._detect_tl,
            "staged_learning":   self._detect_rl,
        }
        checker = checkers.get(label or "")
        return checker(self.output_dir(job_dir)) if checker else False

    def _csv(self, name: str) -> Callable[[Path], bool]:
        def _check(out: Path) -> bool:
            p = out / name
            return p.exists() and p.stat().st_size > 0
        return _check

    def _detect_tl(self, out: Path) -> bool:
        models = [p for p in out.glob("*.model") if p.stat().st_size > 0]
        return bool(models)

    def _detect_rl(self, out: Path) -> bool:
        csvs = [p for p in out.glob("*_*.csv") if p.stat().st_size > 0]
        chkpts = list(out.glob("*.chkpt"))
        return bool(csvs) and bool(chkpts)

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
