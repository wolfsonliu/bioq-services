"""Promera service adapter.

Detects outputs across both endpoints (cofold and design), supplies the
service-specific manifest section, and sets LIGANDMPNN_DIR for the subprocess.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import PromeraSettings


class PromeraAdapter(JobAdapter):
    name = "promera"

    settings: PromeraSettings

    def __init__(self, settings: PromeraSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for cif in out.rglob("*.cif"):
            if cif.stat().st_size > 0:
                return True
        return False

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def subprocess_env(self) -> dict[str, str]:
        return {
            "PROMERA_WEIGHTS": self.settings.weights,
            "LIGANDMPNN_DIR": self.settings.ligandmpnn_dir,
        }

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "cofold": (
                    "output/<name>/<name>_seed<i>_samp<j>.cif — predicted structures (mmCIF). "
                    "<name>_seed<i>_samp<j>_conf.json — confidence scores (plddt, ptm, iptm, chain scores). "
                    "Optional: *_conf.npz (full matrices), *_distogram.npy, *_traj.cif (trajectory)."
                ),
                "design": (
                    "output/<name>/sample<b>/backbone.cif — diffusion backbone. "
                    "sample<b>_design<j>.fasta — redesigned binder sequence. "
                    "refolds/sample<b>_design<j>_refold<r>.cif — refolded structures. "
                    "refolds/*_confidence.json — iptm, scRMSD, scDockQ per refold."
                ),
            },
            "model": {
                "name": "promera",
                "architecture": "pairformer (48 blocks) + EDM diffusion + confidence/contact modules",
                "built_on": "tinyprot",
                "supports_cofolding": True,
                "supports_binder_design": True,
                "supports_vhh_design": True,
            },
            "input_format": {
                "schema": (
                    "tinyprot JSON schema. Each key is a chain ID, value has 'type' "
                    "(protein/ligand) and 'sequence'. Example: "
                    '{"A1": {"type": "protein", "sequence": "MQIFVKTLT..."}}'
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data — JSON schema / CIF template upload.",
                "job://<id>/<file>": "Re-use a file from a prior job.",
                "file:///abs/path": "Direct NAS path.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL.",
            },
            "design_types": {
                "minibinder": "Free-length protein binder (40-120 residues by default).",
                "vhh": "VHH nanobody with caplacizumab framework + variable CDR loops.",
            },
            "endpoints_summary": {
                "/api/cofold": "Structure prediction of protein complexes, protein-ligand, etc.",
                "/api/design": "De novo binder design (minibinder or VHH nanobody).",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/cofold": [
                EndpointExample(
                    title="single protein structure prediction",
                    curl=(
                        "curl -X POST $URL/api/cofold "
                        '-F input_schema=@ubiquitin.json '
                        "-F num_seeds=1 "
                        "-F diffusion_samples=5"
                    ),
                    notes=(
                        "Input is a tinyprot JSON schema file. No MSA required "
                        "(uses dummy MSA by default). See /api/manifest for full params."
                    ),
                ),
            ],
            "/api/design": [
                EndpointExample(
                    title="minibinder design against IL7Ra",
                    curl=(
                        "curl -X POST $URL/api/design "
                        '-F target_schema=@IL7Ra.json '
                        "-F design_type=minibinder "
                        "-F num_backbones=10 "
                        "-F binder_length_min=60 "
                        "-F binder_length_max=80 "
                        "-F inverse_folder_type=solublempnn"
                    ),
                    notes="Generates 10 backbone samples with SolubleMPNN inverse folding.",
                ),
                EndpointExample(
                    title="VHH nanobody design",
                    curl=(
                        "curl -X POST $URL/api/design "
                        '-F target_schema=@PDL1.json '
                        "-F design_type=vhh "
                        "-F num_backbones=5"
                    ),
                    notes=(
                        "VHH mode uses caplacizumab framework with variable CDR loops. "
                        "CDR lengths are sampled from preset ranges (H1: 5-7, H2: 7-12, H3: 9-15)."
                    ),
                ),
            ],
        }
