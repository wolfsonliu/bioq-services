"""DeepRank-Ab service adapter.

Detects *_predictions.csv output from the inference pipeline and exposes
the PYTHONPATH needed by the upstream scripts.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import DeepRankAbSettings


class DeepRankAbAdapter(JobAdapter):
    name = "deeprank-ab"

    settings: DeepRankAbSettings

    def __init__(self, settings: DeepRankAbSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for csv in out.rglob("*_predictions.csv"):
            if csv.stat().st_size > 0:
                return True
        return False

    def subprocess_env(self) -> dict[str, str]:
        return {
            "PYTHONPATH": str(self.settings.root / "DeepRank-Ab"),
        }

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "score": (
                    "output/<stem>-deeprank_ab_pred_<H><L>_<A>/"
                    "<stem>-deeprank_ab_pred_<H><L>_<A>_predictions.csv "
                    "— predicted DockQ scores sorted descending, with quality_flag column. "
                    "Also: *_graph.hdf5 (atom-level graphs) and *_predictions.hdf5 (raw model outputs)."
                ),
                "note": (
                    "Use GET /api/jobs/{id}/files to enumerate; download a single file "
                    "via /api/jobs/{id}/file/{relpath} or zip the whole dir via /download."
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data field `input_pdb`.",
                "job://<id>/<file>": "Re-use a file from a prior job on the same NAS.",
                "file:///abs/path": "Direct NAS path.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "scoring_legend": {
                "predicted_dockq": "0–1, predicted DockQ score from the EGNN model.",
                "quality_flag": (
                    "'ok' = normal; 'low_HL_contacts' = fewer than 15 CA-CA contacts "
                    "between VH and VL at 8 Å (suspicious); 'not_applicable' = nanobody / "
                    "single-chain (no light chain)."
                ),
            },
            "model_info": {
                "architecture": "EGNN (Equivariant Graph Neural Network), dual-branch with Gaussian RBF.",
                "sequence_encoder": "ESM-2 (esm2_t33_650M_UR50D), layer 33 representations.",
                "annotation": "ANARCI IMGT numbering for CDR/FR/CONST region labels.",
                "pretrained_weights": "top_mse_ep37 (trained on AlphaFold models).",
            },
            "config_tips": {
                "light_chain_id": (
                    "Use '-' for nanobodies / VHH that lack a light chain. "
                    "The pipeline will skip light-chain annotation and contact checks."
                ),
                "chain_ids": (
                    "Chain IDs must match those in the input PDB file. "
                    "Common convention: H=heavy, L=light, A=antigen."
                ),
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/score": [
                EndpointExample(
                    title="score antibody-antigen complex",
                    curl=(
                        "curl -X POST $URL/api/score "
                        "-F input_pdb=@complex.pdb "
                        "-F heavy_chain_id=H "
                        "-F light_chain_id=L "
                        "-F antigen_chain_id=A"
                    ),
                    notes=(
                        "Chains H (heavy), L (light), A (antigen) must exist in the PDB. "
                        "Result is a CSV with predicted DockQ scores per model in the ensemble."
                    ),
                ),
                EndpointExample(
                    title="score nanobody-antigen (no light chain)",
                    curl=(
                        "curl -X POST $URL/api/score "
                        "-F input_pdb=@nanobody_complex.pdb "
                        "-F heavy_chain_id=H "
                        "-F light_chain_id=- "
                        "-F antigen_chain_id=A"
                    ),
                    notes="Use '-' for light_chain_id when scoring VHH / nanobody complexes.",
                ),
            ],
        }
