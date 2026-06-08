"""AlphaFold service adapter.

Detects ranked PDB output files, supplies the service-specific manifest section,
and provides the subprocess environment.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import AlphaFoldSettings


class AlphaFoldAdapter(JobAdapter):
    name = "alphafold"

    settings: AlphaFoldSettings

    def __init__(self, settings: AlphaFoldSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for pdb in out.rglob("ranked_0.pdb"):
            if pdb.stat().st_size > 0:
                return True
        return False

    def subprocess_env(self) -> dict[str, str]:
        return {
            "TF_FORCE_UNIFIED_MEMORY": "1",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0",
            "OPENMM_CPU_THREADS": str(self.settings.n_cpu),
        }

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "fold": (
                    "output/input/ranked_{0..4}.pdb — ranked structures (0=best). "
                    "output/input/ranking_debug.json — model scores. "
                    "output/input/relaxed_model_*.pdb — Amber-relaxed structures. "
                    "output/input/confidence_model_*.json — per-residue pLDDT. "
                    "output/input/pae_model_*.json — predicted aligned error (pTM/multimer presets)."
                ),
            },
            "model": {
                "name": "AlphaFold v2.3.2",
                "base": "Evoformer + IPA + FAPE, JAX/Haiku",
                "weights": "DeepMind AlphaFold parameters (NAS-mounted)",
                "supports": ["protein (monomer)", "protein complex (multimer)"],
                "output_format": "PDB",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data — FASTA file as `input_fasta`.",
                "job://<id>/<file>": "Re-use a file from a prior job.",
                "file:///abs/path": "Direct NAS path.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL.",
            },
            "config_tips": {
                "model_preset": (
                    "Default monomer_ptm (pLDDT + pTM + PAE). "
                    "Use 'multimer' for protein complexes."
                ),
                "db_preset": (
                    "Default reduced_dbs (~300GB, MSA ~10-30 min). "
                    "full_dbs (~2.6TB, MSA ~30-120 min) for maximum accuracy."
                ),
                "models_to_relax": (
                    "Default 'best' (relax ranked_0 only). "
                    "'all' relaxes all 5 models; 'none' skips Amber relax."
                ),
                "max_template_date": (
                    "Default 2022-01-01. Templates after this date are excluded. "
                    "Use a past date for fair benchmarking."
                ),
            },
            "endpoints_summary": {
                "/api/fold": "Protein structure prediction (monomer or multimer).",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/fold": [
                EndpointExample(
                    title="single protein (monomer_ptm)",
                    curl=(
                        "curl -X POST $URL/api/fold "
                        "-F 'input_fasta=@protein.fasta' "
                        "-F 'model_preset=monomer_ptm'"
                    ),
                    notes="Single-chain structure prediction with pTM/PAE output.",
                ),
                EndpointExample(
                    title="protein complex (multimer)",
                    curl=(
                        "curl -X POST $URL/api/fold "
                        "-F 'input_fasta=@complex.fasta' "
                        "-F 'model_preset=multimer' "
                        "-F 'num_multimer_predictions_per_model=1'"
                    ),
                    notes=(
                        "Multi-chain complex prediction. FASTA must have one entry "
                        "per chain with unique headers."
                    ),
                ),
            ],
        }
