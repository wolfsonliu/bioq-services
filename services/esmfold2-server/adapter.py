"""ESMFold2 service adapter.

Detects mmCIF output files, supplies the service-specific manifest section,
and provides the subprocess cwd.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import ESMFold2Settings


class ESMFold2Adapter(JobAdapter):
    name = "esmfold2"

    settings: ESMFold2Settings

    def __init__(self, settings: ESMFold2Settings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for cif in out.glob("prediction_*.cif"):
            if cif.stat().st_size > 0:
                return True
        return False

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "fold": (
                    "output/prediction_*.cif (one per diffusion sample, mmCIF format). "
                    "output/metrics.json carries per-sample scores (ptm, iptm, plddt_mean)."
                ),
            },
            "model": {
                "name": "ESMFold2",
                "base": "ESMC 6B + diffusion structure prediction",
                "weights": "biohub/ESMFold2 (HuggingFace)",
                "supports": ["protein", "DNA", "RNA", "ligand (CCD/SMILES)"],
                "output_format": "mmCIF",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data — MSA files as `msa_files` (A3M format).",
                "job://<id>/<file>": "Re-use a file from a prior job.",
                "file:///abs/path": "Direct NAS path.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL.",
            },
            "config_tips": {
                "num_loops": "Default 3. More loops = slightly higher accuracy, linear runtime increase.",
                "num_sampling_steps": (
                    "Default 50. Range 10-400. Higher = more accurate but slower. "
                    "50 is a good default; use 200+ for publication-quality structures."
                ),
                "num_diffusion_samples": (
                    "Default 1 (single structure). Set >1 for ensemble generation — "
                    "useful for conformational sampling or ranking."
                ),
                "msa": (
                    "Optional. Upload per-chain A3M files via msa_files (filename stem = chain id). "
                    "Without MSA, ESMFold2 uses single-sequence mode (fast, still accurate)."
                ),
            },
            "endpoints_summary": {
                "/api/fold": "Structure prediction for protein/DNA/RNA/ligand complexes.",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/fold": [
                EndpointExample(
                    title="single protein, single-sequence mode",
                    curl=(
                        "curl -X POST $URL/api/fold "
                        "-F 'sequences=[{\"type\":\"protein\",\"id\":\"A\","
                        "\"sequence\":\"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\"}]'"
                    ),
                    notes="Ubiquitin (76 aa). Single-sequence mode, no MSA needed.",
                ),
                EndpointExample(
                    title="protein-DNA complex with ligand",
                    curl=(
                        "curl -X POST $URL/api/fold "
                        "-F num_loops=3 "
                        "-F num_sampling_steps=50 "
                        "-F 'sequences=["
                        "{\"type\":\"protein\",\"id\":\"A\",\"sequence\":\"MIEIKDKQLTGLR...\"},"
                        "{\"type\":\"dna\",\"id\":\"B\",\"sequence\":\"GATAGCGCTATC\"},"
                        "{\"type\":\"ligand\",\"id\":\"L\",\"ccd\":[\"SAH\"]}"
                        "]'"
                    ),
                    notes="Multi-chain complex with DNA and small molecule ligand.",
                ),
            ],
        }
