"""DockQ service adapter.

Detects outputs across both endpoints and contributes the service-specific
manifest section. DockQ runs CPU-only, so `subprocess_cwd()` defaults to None
(inheriting the caller's cwd is fine; the `DockQ` binary resolves all paths
absolutely from argv).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter, JobInfo, JobStatus  # noqa: F401

from .settings import DockQSettings


class DockQAdapter(JobAdapter):
    name = "dockq"

    settings: DockQSettings  # narrow for IDEs

    def __init__(self, settings: DockQSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Recognize either of the two modes' artifacts:
          - score        : output/<name>.json
          - score_batch  : output/scores.csv
        """
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for path in out.glob("*.json"):
            if path.stat().st_size > 0:
                return True
        scores = out / "scores.csv"
        if scores.is_file() and scores.stat().st_size > 0:
            return True
        return False

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "score": "output/<name>.json — DockQ's native JSON output for the single (model, native) pair.",
                "score_batch": (
                    "output/scores.csv (sorted summary, one row per model) + "
                    "output/per_model/<basename>.json (raw DockQ JSON per model). "
                    "output/failed.csv lists models that errored (if any)."
                ),
                "note": (
                    "Use GET /api/jobs/{id}/files to enumerate; download a single file "
                    "via /api/jobs/{id}/file/{relpath} or zip the whole dir via /download."
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data (single endpoint: fields `model` + `native`; batch: `native` + repeated `models`).",
                "model_uri / native_uri": "URI alternatives to the single-endpoint `model` / `native` uploads.",
                "models_zip_uri": (
                    "Batch only: URI to a zip of candidate model structures (.pdb/.cif/.gz), "
                    "extracted flat as the `models` set. Use when `models` can't be uploaded "
                    "(e.g. via the gateway, which dispatches form fields only)."
                ),
                "job://<id>/<file>": "Re-use a file from a prior dockq job (or any other bioagent service that wrote to NAS).",
                "file:///abs/path": "Direct NAS path; works across services on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object (needs OSS_ACCESS_KEY_ID / _SECRET env vars).",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "scoring_legend": {
                "DockQ": "0–1, overall docking quality. <0.23 incorrect / 0.23–0.49 acceptable / 0.49–0.80 medium / ≥0.80 high quality.",
                "iRMSD": "Interface residue RMSD (Å); lower is better.",
                "LRMSD": "Ligand RMSD over the smaller chain (Å); lower is better.",
                "fnat": "Fraction of native contacts recovered; higher is better.",
                "fnonnat": "Fraction of predicted contacts that are non-native; lower is better.",
                "clashes": "Number of clashing interface residues (<2 Å) in the model.",
            },
            "config_tips": {
                "mapping": (
                    "Format MODELCHAINS:NATIVECHAINS. For antibody-antigen scoring with "
                    "model chains H,L,A and native B,C,X, use `--mapping HLA:BCX`. "
                    "Use `:HL` to restrict native interfaces to H–L only."
                ),
                "small_molecule": (
                    "Required when scoring PDB/CIF inputs that contain HEM / cofactor / "
                    "small-molecule chains. Without this flag DockQ ignores those interfaces."
                ),
                "no_align": (
                    "Skip sequence alignment and trust residue numbering directly. "
                    "Faster, but ONLY safe when model + native share identical residue indexing "
                    "(e.g. both come from the same crystal structure or pipeline)."
                ),
                "n_cpu": (
                    "Forwarded to DockQ's --n_cpu (parallelism for chain-mapping enumeration "
                    "within a single call, not across batch models). Default 4 fits a 4 vCPU FC instance."
                ),
                "sort_by": (
                    "scores.csv summary column. Defaults to DockQ (descending). "
                    "For RMSD-style metrics the sort is still descending in the CSV — "
                    "agents should re-sort client-side if ascending order is preferred."
                ),
            },
            "endpoints_summary": {
                "/api/score": "Single (model, native) DockQ scoring; output/<name>.json.",
                "/api/score_batch": "1 reference native vs N candidate models; output/scores.csv + per_model/*.json.",
            },
            "batch_limits": {
                "max_batch_size": self.settings.max_batch_size,
                "expected_runtime": "DockQ runs ~1–10 s per protein-protein pair on 4 vCPUs; batch of 100 ~3–15 min.",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/score": [
                EndpointExample(
                    title="basic protein-protein scoring",
                    curl=(
                        "curl -X POST $URL/api/score "
                        "-F model=@model.pdb "
                        "-F native=@native.pdb "
                        "-F name=demo"
                    ),
                    notes="Result lands in output/demo.json once the job completes.",
                ),
                EndpointExample(
                    title="antibody-antigen with explicit chain mapping",
                    curl=(
                        "curl -X POST $URL/api/score "
                        "-F model=@design.pdb "
                        "-F native=@reference.pdb "
                        "-F mapping=HLA:BCX "
                        "-F name=ab_ag"
                    ),
                    notes="`mapping` follows DockQ's MODELCHAINS:NATIVECHAINS format. Wildcards (`*`) allowed.",
                ),
                EndpointExample(
                    title="small-molecule complex (mmCIF input)",
                    curl=(
                        "curl -X POST $URL/api/score "
                        "-F model=@1HHO_hem.cif "
                        "-F native=@2HHB_hem.cif "
                        "-F small_molecule=true "
                        "-F mapping=:ABEFG "
                        "-F name=hem"
                    ),
                    notes="Required for HEM / cofactor / ligand chains; otherwise their interfaces are ignored.",
                ),
            ],
            "/api/score_batch": [
                EndpointExample(
                    title="rank N candidate designs against one reference",
                    curl=(
                        "curl -X POST $URL/api/score_batch "
                        "-F native=@reference_complex.pdb "
                        "-F models=@design_001.pdb "
                        "-F models=@design_002.pdb "
                        "-F models=@design_003.pdb "
                        "-F sort_by=DockQ "
                        "-F name=batch1"
                    ),
                    notes=(
                        "scores.csv lists models sorted by DockQ (descending); raw per-model "
                        "JSONs under output/per_model/. Models that errored land in failed.csv."
                    ),
                ),
                EndpointExample(
                    title="chain mapping shared across all models",
                    curl=(
                        "curl -X POST $URL/api/score_batch "
                        "-F native=@ab_native.pdb "
                        "-F models=@cand_01.pdb "
                        "-F models=@cand_02.pdb "
                        "-F mapping=HLA:BCX "
                        "-F sort_by=DockQ"
                    ),
                    notes="`mapping` is forwarded to every DockQ invocation; cuts mapping-enumeration time substantially.",
                ),
            ],
        }
