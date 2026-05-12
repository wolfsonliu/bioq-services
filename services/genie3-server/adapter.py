"""Service-wide policy for genie3-server.

Three overrides vs. the framework defaults:

  * `detect_outputs` walks the whole `output/` tree looking for `*.pdb` because
    genie3 nests results under `output/<experiment_name>/pdbs/`.
  * `subprocess_cwd` returns the genie3 source root so its relative imports +
    hydra config dirs resolve.
  * `manifest_extras` documents the four endpoints' output conventions and the
    `dataset` zip layout an agent must produce for motif / binder modes.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .settings import Genie3Settings


class Genie3Adapter(JobAdapter):
    name = "genie3"

    settings: Genie3Settings  # narrow the type for IDEs

    def __init__(self, settings: Genie3Settings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """genie3 writes PDBs under `output/<exp>/pdbs/`. Recurse for any .pdb."""
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
        """genie3 CLI assumes cwd == project root (hydra config_dir, package data)."""
        return self.settings.root

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        """Examples for all four endpoints, including the `cond_strategy` workaround."""
        return {
            "/api/generate/unconditional": [
                EndpointExample(
                    title="short protein, default direction_scale",
                    curl=(
                        "curl -X POST $URL/api/generate/unconditional "
                        "-F min_length=100 "
                        "-F max_length=100 "
                        "-F n_sample=8"
                    ),
                    notes="No dataset needed. Use direction_scale=0.0 for length > 300.",
                ),
            ],
            "/api/generate/motif": [
                EndpointExample(
                    title="motif scaffolding from a benchmark zip",
                    curl=(
                        "curl -X POST $URL/api/generate/motif "
                        "-F dataset=@motifbench.zip "
                        "-F selections=22_1BCF "
                        "-F n_sample=8"
                    ),
                    notes=(
                        "The zip must contain problems/*.json + motifs/*.pdb. The server "
                        "rewrites motif_filepaths inside problem JSONs to absolute paths."
                    ),
                ),
            ],
            "/api/generate/binder": [
                EndpointExample(
                    title="binder design against a target",
                    curl=(
                        "curl -X POST $URL/api/generate/binder "
                        "-F dataset=@binderbench.zip "
                        "-F selections=01_bhrf1 "
                        "-F n_sample=8 "
                        "-F direction_scale=0.0"
                    ),
                    notes="Zip layout: problems/ + targets/{pdb,fasta,msa}/.",
                ),
            ],
            "/api/generate": [
                EndpointExample(
                    title="custom YAML with cond_strategy override (the common gotcha)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F 'config_yaml=experiment:\n"
                        "  name: trem1_vhh\n"
                        "generation:\n"
                        "  dataset:\n"
                        "    source: target\n"
                        "    selections: trem1_vhh\n"
                        "    n_sample: 4\n"
                        "    cond_strategy: hotspot' "
                        "-F dataset=@trem1.zip"
                    ),
                    notes=(
                        "Default cond_strategy='extended' fails with "
                        "ValueError: Interface mode 'extended' not found when the "
                        "problem JSON only defines 'hotspot'. Override here to use "
                        "the hotspot interface."
                    ),
                ),
            ],
        }

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "all_modes": "output/<experiment_name>/pdbs/*.pdb",
                "note": (
                    "Each `genie3 generate` run nests its PDBs under the experiment "
                    "name from the YAML (`experiment.name`). Use `/api/jobs/{id}/files` "
                    "to list, or `/api/jobs/{id}/file/{relpath}` to fetch one."
                ),
            },
            "input_uri_schemes": {
                "upload": (
                    "multipart/form-data zip with problems/ + targets/ (binder) "
                    "or problems/ + motifs/ (motif). The server rewrites filepath "
                    "fields in problem JSONs to absolute paths after extraction."
                ),
                "unconditional": "No dataset required.",
            },
            "endpoints_summary": {
                "/api/generate/unconditional": "No dataset; configurable min/max length + step.",
                "/api/generate/motif": "Dataset zip with motifs/; scaffold around motif residues.",
                "/api/generate/binder": "Dataset zip with targets/; design binder against target.",
                "/api/generate": (
                    "Freeform YAML body via `config_yaml` Form field. The server "
                    "auto-fills `paths.rootdir` and (if a dataset zip is attached) "
                    "`paths.dataset`. Use this for `cond_strategy` overrides or "
                    "iterative refinement configs."
                ),
            },
            "config_tips": {
                "cond_strategy": (
                    "Default upstream `cond_strategy: extended` requires an `extended` "
                    "interface in the problem JSON. If your problem only defines "
                    "`hotspot`, override via `generation.dataset.cond_strategy: hotspot`."
                ),
                "direction_scale": (
                    "Quality/diversity knob. Recommended 0.8 for length ≤ 300, 0.0 "
                    "for longer (unconditional) or for binder mode (default 0.0)."
                ),
            },
        }

    def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
        """Recovered jobs surface the number of PDBs already on disk as a hint."""
        out = self.output_dir(job_dir)
        pdbs = [f for f in out.rglob("*.pdb") if f.is_file() and f.stat().st_size > 0] if out.is_dir() else []
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
