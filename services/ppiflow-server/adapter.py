"""Service-wide policy for ppiflow-server.

Notable overrides:

  * `detect_outputs` recursively scans for any `.pdb` since each PPIFlow
    sampler writes its results in a per-name subdirectory.
  * `subprocess_cwd` returns the PPIFlow tool root because all `sample_*.py`
    scripts are launched relative to that directory (matching upstream's
    README usage).
  * `manifest_extras` + `endpoint_examples` document the 5 endpoints, their
    expected output paths, supported input URI schemes, and a `cdr_length` /
    `length_subset` hint each.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .settings import PPIFlowSettings


class PPIFlowAdapter(JobAdapter):
    name = "ppiflow"

    settings: PPIFlowSettings  # narrow for IDEs

    def __init__(self, settings: PPIFlowSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Any non-empty .pdb under output/ counts as success."""
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for f in out.rglob("*.pdb"):
            try:
                if f.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    def subprocess_cwd(self) -> Path | None:
        """PPIFlow's sample_*.py scripts assume cwd == tool/PPIFlow root."""
        return self.settings.root

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                # Verified end-to-end against the deployed FC service:
                # binder writes flat with a name prefix, the other modes nest.
                "binder": "output/<name>_<idx>.pdb (flat) + output/sample_metrics.csv",
                "antibody": "output/<name>/*.pdb",
                "nanobody": "output/<name>/*.pdb",
                "monomer": "output/<name>/*.pdb",
                "scaffolding": "output/<name>/*.pdb",
                "note": (
                    "Binder outputs are flat under `output/` with files named "
                    "`<name>_<idx>.pdb` plus `sample_metrics.csv`.  The other four "
                    "modes nest under `output/<name>/`.  Use "
                    "GET /api/jobs/{id}/files to enumerate, or /file/{relpath} for one."
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data file fields (target_pdb / antigen_pdb / framework_pdb / motif_csv).",
                "job://<id>/<file>": "Re-use a file from a prior ppiflow job (same NAS).",
                "file:///abs/path": "Direct NAS path; works across services on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object (needs OSS_ACCESS_KEY_ID / _SECRET env vars).",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "weights": {
                "binder": "checkpoint/binder.ckpt",
                "antibody": "checkpoint/antibody.ckpt",
                "nanobody": "checkpoint/nanobody.ckpt",
                "monomer": "checkpoint/monomer.ckpt (used by both monomer + scaffolding endpoints)",
                "note": "Baked into the Docker image. Missing files log a warning; the subprocess will then fail with a clearer message.",
            },
            "config_tips": {
                "cdr_length": (
                    "Comma-separated `CDR<name>,min-max,...`. Antibody must include "
                    "all six CDRs (CDRH1-3 + CDRL1-3); nanobody covers heavy only "
                    "(CDRH1-3). Defaults are upstream's recommendations."
                ),
                "length_subset": (
                    "Monomer endpoint takes a JSON list of target lengths. PPIFlow "
                    "samples `samples_per_target` structures at each length."
                ),
                "specified_hotspots": (
                    "Format `<chain><resnum>,<chain><resnum>,...` (e.g. "
                    "`B119,B141,B200`). Omitting it puts PPIFlow into random "
                    "hotspot sampling via `sample_hotspot_rate_{min,max}`."
                ),
                "framework_pdb": (
                    "Antibody/nanobody only. CDR loops must already be removed and "
                    "IMGT numbering applied. See PPIFlow README for preprocessing."
                ),
            },
            "endpoints_summary": {
                "/api/sample/binder": "PPI binder design against an uploaded target PDB.",
                "/api/sample/antibody": "Full antibody CDR design (heavy + light) over a framework.",
                "/api/sample/nanobody": "VHH CDR design (heavy-only) over a framework.",
                "/api/sample/monomer": "Unconditional monomer generation at requested lengths.",
                "/api/sample/scaffolding": "Motif scaffolding from a CSV + motif PDBs.",
            },
            "not_in_scope_v0_0_1": (
                "Partial-flow redesign (sample_*_partial.py) is not exposed in v0.0.1. "
                "Sequence design (ProteinMPNN/AbMPNN), packing (Flowpacker), scoring "
                "(AF3Score), and Rosetta steps live in their own bioagent services."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/sample/binder": [
                EndpointExample(
                    title="basic binder against IL7Ra target",
                    curl=(
                        "curl -X POST $URL/api/sample/binder "
                        "-F target=@target.pdb "
                        "-F target_chain=B "
                        "-F binder_chain=A "
                        "-F 'specified_hotspots=B119,B141,B200' "
                        "-F samples_min_length=75 "
                        "-F samples_max_length=120 "
                        "-F samples_per_target=5 "
                        "-F name=IL7Ra"
                    ),
                    notes="Mirrors the upstream README example. Output PDBs land in output/IL7Ra/*.pdb.",
                ),
            ],
            "/api/sample/antibody": [
                EndpointExample(
                    title="antibody CDR design against IL-13",
                    curl=(
                        "curl -X POST $URL/api/sample/antibody "
                        "-F antigen=@antigen.pdb "
                        "-F framework=@framework.pdb "
                        "-F antigen_chain=C "
                        "-F heavy_chain=A "
                        "-F light_chain=B "
                        "-F 'specified_hotspots=C11,C14,C15,C101,C107,C108' "
                        "-F 'cdr_length=CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11' "
                        "-F samples_per_target=5 "
                        "-F name=1IJZ_IL13"
                    ),
                    notes="Framework PDB needs IMGT numbering and CDRs removed (see upstream README).",
                ),
            ],
            "/api/sample/nanobody": [
                EndpointExample(
                    title="VHH design against FGFR1",
                    curl=(
                        "curl -X POST $URL/api/sample/nanobody "
                        "-F antigen=@antigen.pdb "
                        "-F framework=@nanobody_framework.pdb "
                        "-F antigen_chain=C "
                        "-F heavy_chain=A "
                        "-F 'specified_hotspots=C101,C135,C171,C198' "
                        "-F 'cdr_length=CDRH1,8-8,CDRH2,8-8,CDRH3,9-21' "
                        "-F samples_per_target=5 "
                        "-F name=1CVS_FGFR1"
                    ),
                    notes=(
                        "Same script as antibody but with nanobody.ckpt and no light "
                        "chain. Output: output/1CVS_FGFR1/*.pdb."
                    ),
                ),
            ],
            "/api/sample/monomer": [
                EndpointExample(
                    title="unconditional monomer at two lengths",
                    curl=(
                        "curl -X POST $URL/api/sample/monomer "
                        "-F 'length_subset=[50, 100]' "
                        "-F samples_per_target=5 "
                        "-F name=monomer_test"
                    ),
                    notes="length_subset is a JSON list. Produces 5 samples at each length.",
                ),
            ],
            "/api/sample/scaffolding": [
                EndpointExample(
                    title="motif scaffolding from a CSV",
                    curl=(
                        "curl -X POST $URL/api/sample/scaffolding "
                        "-F motif_csv=@motif_metadata.csv "
                        "-F 'motif_names=[\"01_1LDB\"]' "
                        "-F samples_per_target=5 "
                        "-F name=motif_test"
                    ),
                    notes=(
                        "CSV columns: target,length,contig,motif_path. motif_path is a "
                        "PDB filename relative to the CSV's directory; upload the PDB(s) "
                        "alongside in a sibling location reachable from the CSV's parent "
                        "(typically include them in the same upload zip)."
                    ),
                ),
            ],
        }

    def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
        out = self.output_dir(job_dir)
        pdbs = [
            f for f in out.rglob("*.pdb")
            if f.is_file() and f.stat().st_size > 0
        ] if out.is_dir() else []
        if pdbs:
            return JobInfo(
                job_id=job_dir.name,
                status=JobStatus.COMPLETED,
                message=f"Recovered from disk ({len(pdbs)} PDB outputs)",
            )
        return JobInfo(
            job_id=job_dir.name,
            status=JobStatus.FAILED,
            message="Recovered from disk (no PDB outputs)",
        )
