"""LightDock service adapter.

Detects ranked docking outputs and contributes the service-specific manifest
section. LightDock is CPU-only with no NN weights, so `subprocess_env()` just
inherits the environment; the driver resolves all paths absolutely / via chdir.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter, JobInfo, JobStatus  # noqa: F401

from .settings import LightdockSettings


class LightdockAdapter(JobAdapter):
    name = "lightdock"

    settings: LightdockSettings  # narrow for IDEs

    def __init__(self, settings: LightdockSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Success = at least one ranked pose or a non-empty ranking file.

        Label-agnostic (the framework writes no per-job manifest): look for
        output/top/*.pdb, else output/rank_by_scoring.list.
        """
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        top_dir = out / "top"
        if top_dir.is_dir():
            for pdb in top_dir.glob("top_*.pdb"):
                if pdb.stat().st_size > 0:
                    return True
        ranking = out / "rank_by_scoring.list"
        return ranking.is_file() and ranking.stat().st_size > 0

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "dock": (
                    "output/top/top_<n>.pdb — ranked docked complexes (receptor + "
                    "ligand), top_1 = best score. output/rank_by_scoring.list — "
                    "global ranking table. output/setup.json — resolved LightDock "
                    "config (swarm count, ANM, seeds)."
                ),
                "note": (
                    "Use GET /api/jobs/{id}/files to enumerate; download a single "
                    "file via /api/jobs/{id}/file/{relpath} or zip the whole dir "
                    "via /download."
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data fields `receptor` + `ligand` (+ optional `restraints`).",
                "receptor_uri / ligand_uri / restraints_uri": "URI alternatives to the uploads.",
                "job://<id>/<file>": "Re-use a file from a prior job on the shared NAS.",
                "file:///abs/path": "Direct NAS path; works across services on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object (needs OSS_ACCESS_KEY_ID / _SECRET env vars).",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "config_tips": {
                "runtime": (
                    "Cost scales with swarms x glowworms x steps. The upstream "
                    "production preset (400 swarms, 200 glowworms, 100 steps) takes "
                    "hours. For interactive runs set a small `swarms` (10-40) and "
                    "supply `restraints` to focus sampling."
                ),
                "swarms": (
                    "0 = auto-estimate from the receptor surface (can be hundreds). "
                    "Set explicitly to bound runtime."
                ),
                "restraints": (
                    "LightDock restraints file: lines like `R A.35` (receptor active) "
                    "or `L B.10` (ligand). Focuses swarms near known interface residues."
                ),
                "scoring_function": (
                    "fastdfire (default, fast DFIRE) suits most protein-protein cases; "
                    "`dna` for protein-DNA; `cpydock` (pyDock energy) and `pisa` are "
                    "alternatives. See /healthz/detail for the installed set."
                ),
                "use_anm": (
                    "Enable ANM backbone flexibility to sample conformational changes "
                    "(slower). Off by default (rigid-body)."
                ),
            },
            "endpoints_summary": {
                "/api/dock": (
                    "Full LightDock GSO docking protocol (setup -> run -> "
                    "conformations -> cluster -> rank -> top); output/top/top_N.pdb."
                ),
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/dock": [
                EndpointExample(
                    title="quick interactive protein-protein docking (small sampling)",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F receptor=@receptor.pdb "
                        "-F ligand=@ligand.pdb "
                        "-F swarms=20 -F glowworms=100 -F steps=50 -F top=10"
                    ),
                    notes=(
                        "Ranked poses land in output/top/top_1.pdb .. top_10.pdb. "
                        "Keep swarms small for interactive latency."
                    ),
                ),
                EndpointExample(
                    title="restraint-driven docking",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F receptor=@receptor.pdb "
                        "-F ligand=@ligand.pdb "
                        "-F restraints=@restraints.list "
                        "-F swarms=20 -F steps=100"
                    ),
                    notes="restraints.list focuses swarms near the specified interface residues.",
                ),
                EndpointExample(
                    title="protein-DNA docking",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F receptor=@protein.pdb "
                        "-F ligand=@dna.pdb "
                        "-F scoring_function=dna -F swarms=20"
                    ),
                    notes="Use the `dna` scoring function for protein-nucleic-acid complexes.",
                ),
            ],
        }
