"""Service-wide policy for bindflow-server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bioagent_service import EndpointExample, JobAdapter

from .settings import BindFlowSettings


class BindFlowAdapter(JobAdapter):
    name = "bindflow"

    settings: BindFlowSettings  # narrow for IDEs

    def __init__(self, settings: BindFlowSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Completed iff BindFlow wrote at least one of its partial-result CSVs.

        NOTE: BindFlow's snakemake pipeline runs with `--keep-going`; the CSV
        is written even when only some ligand × replica combinations
        succeeded.  A green `detect_outputs` therefore means "some results
        are available", NOT "all requested ligand × replica cells passed".
        Clients must check row count == len(ligands) × replicas to gate on
        full completion (documented in manifest_extras.tool_outputs).
        """
        out = self.output_dir(job_dir)
        for name in ("fep_partial_results.csv", "mmxbsa_partial_results.csv"):
            p = out / name
            if p.exists() and p.stat().st_size > 0:
                return True
        return False

    def subprocess_cwd(self) -> Optional[Path]:
        # cwd is decided per-invocation by the argv builder (needs job_dir),
        # so return None here and let tools.py set it via a wrapper `cd` in
        # argv.  In practice we prefer cwd=<job_dir>/output so snakemake's
        # `.snakemake/` cache lands in the job dir, not the algorithm root.
        return None

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "BindFlow",
                "method": "snakemake-orchestrated GROMACS free-energy workflow (Boresch-restrained FEP + MMPBSA)",
                "task": "MD-based absolute binding free energy prediction",
                "long_running": True,
                "typical_duration": {
                    "fep": "hours to days per project (30-40 lambda windows × 5-20 ns each)",
                    "mmpbsa": "minutes to hours per project (single-trajectory approach)",
                },
                "license": "GPL-3.0 (upstream); fork of ABFE_workflow (Biggin Lab, Oxford).",
            },
            "tool_outputs": {
                "fep": (
                    "output/fep_partial_results.csv — MBAR / BAR ΔG per ligand × replica. "
                    "PARTIAL until all snakemake rules complete: check row count == "
                    "len(ligands) × replicas before treating as final."
                ),
                "mmpbsa": (
                    "output/mmxbsa_partial_results.csv — MM(P/G)BSA ΔG per ligand × replica × sample. "
                    "Same partial-completion caveat as fep."
                ),
                "artifacts": (
                    "output/<ligand>/<replica>/ — per-ligand per-replica MD run directories "
                    "(TPR, XTC, log, xvg intermediates).  Preserved for post-hoc analysis "
                    "via alchemlyb / pymbar."
                ),
            },
            "input_uri_schemes": {
                "protein": "multipart UploadFile OR one of: oss:// / job:// / file:// / http(s)://",
                "ligands": "multipart repeat (List[UploadFile]) OR zip via ligands_zip_uri",
                "cofactor": "optional; single SDF/MOL",
                "membrane": "optional PDB (requires CRYST1)",
                "custom_ff": "optional zip of *.ff directories",
                "topology": "optional zip of per-ligand .gro + .top",
            },
            "config_tips": {
                "hmr_factor": (
                    "Default 2.5 with dt_max=0.004.  Set to null / omit for topologies "
                    "that already have HMR applied.  hmr_factor < 2.0 requires dt_max <= 0.002."
                ),
                "num_jobs_frontend": (
                    "For scheduler='frontend' (default), cap at floor(n_cpu / threads) — "
                    "otherwise the node overheats."
                ),
                "scheduler_slurm_caveat": (
                    "scheduler='slurm' requires apptainer --bind of sbatch/squeue/scancel + slurm socket. "
                    "See engineering/guides/apptainer-compatibility.md."
                ),
                "espaloma_note": (
                    "Espaloma FF is NOT bundled in v0.0.1 (would add ~3-6 GB TF+DGL). "
                    "ligand_ff_type accepts 'openff' or 'gaff' only."
                ),
                "mmpbsa_requires_extra": (
                    "MMPBSA endpoint requires the gmx_MMPBSA fork; healthz/detail "
                    "reports its availability."
                ),
            },
            "task_comparison": {
                "vs_boltz_server_affinity": (
                    "boltz-server predicts ΔG in seconds via a ML head; bindflow-server "
                    "is MD-based (hours-days) and considered more rigorous.  Use boltz "
                    "for large screens, bindflow for top-N refinement of the shortlist."
                ),
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/calculate/fep": [
                EndpointExample(
                    title="minimal FEP job (3 ligands, 3 replicas)",
                    curl=(
                        "curl -X POST $URL/api/calculate/fep "
                        "-F protein=@receptor.pdb "
                        "-F ligands=@lig_a.sdf -F ligands=@lig_b.sdf -F ligands=@lig_c.sdf "
                        "-F water_model=amber/tip3p -F replicas=3 -F threads=8 -F num_jobs=6"
                    ),
                    notes=(
                        "Long-running (hours+).  Poll /api/jobs/<id> for status.  "
                        "In production, prefer the CLI batch mode on HPC (design doc §8.1)."
                    ),
                ),
                EndpointExample(
                    title="FEP with cofactor + custom lambda windows",
                    curl=(
                        "curl -X POST $URL/api/calculate/fep "
                        "-F protein=@receptor.pdb -F ligands=@lig_a.sdf "
                        "-F cofactor=@gdp.sdf -F cofactor_on_protein=true "
                        "-F nwindows_ligand_vdw=15 -F nwindows_complex_vdw=25 "
                        "-F replicas=3"
                    ),
                    notes="More lambda windows increase precision at proportional cost.",
                ),
            ],
            "/api/calculate/mmpbsa": [
                EndpointExample(
                    title="minimal MMPBSA job",
                    curl=(
                        "curl -X POST $URL/api/calculate/mmpbsa "
                        "-F protein=@receptor.pdb -F ligands=@lig_a.sdf "
                        "-F samples=20 -F replicas=3"
                    ),
                    notes=(
                        "Faster than FEP (minutes-hours).  Requires gmx_MMPBSA "
                        "(bundled in image); healthz/detail reports its presence."
                    ),
                ),
                EndpointExample(
                    title="MMPBSA with membrane system",
                    curl=(
                        "curl -X POST $URL/api/calculate/mmpbsa "
                        "-F protein=@gpcr.pdb -F ligands=@lig.sdf "
                        "-F membrane=@popc.pdb -F cofactor=@na.sdf "
                        "-F replicas=3 -F samples=40"
                    ),
                    notes="Membrane PDB must have a correct CRYST1 line (from CHARMM-GUI etc.).",
                ),
            ],
        }
