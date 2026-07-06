"""Argv assembly for bindflow-server.

Every request → argv list for our own `inference.py` wrapper (see
`inference.py`), which in turn calls `bindflow.runners.calculate(...)`.
We do NOT subprocess into the upstream `bindflow` CLI; that CLI only
supports post-hoc `dag / check_* / clean` commands, not the actual
workflow submission.

The wrapper is the extension point where we translate our pydantic
fields into BindFlow's `global_config` dict and its many keyword
arguments.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import BaseCalculateRequest, FepCalculateRequest, MmpbsaCalculateRequest
from .settings import BindFlowSettings


_LIGAND_SUFFIXES = {".sdf", ".mol", ".mol2"}


def calculate_argv(
    req: BaseCalculateRequest,
    *,
    calculation_type: str,
    job_dir: Path,
    protein_path: Path,
    ligands_dir: Path,
    settings: BindFlowSettings,
    cofactor_path: Path | None = None,
    membrane_path: Path | None = None,
    custom_ff_dir: Path | None = None,
    topology_dir: Path | None = None,
) -> list[str]:
    """Compose argv for our `inference.py` wrapper.

    Output goes to `<job_dir>/output/`.  All paths are absolute so the
    subprocess `cwd` (set to the output dir for snakemake sanity) does not
    affect resolution.
    """
    if calculation_type not in ("fep", "mmpbsa"):
        raise ValueError(f"calculation_type must be 'fep' or 'mmpbsa'; got {calculation_type!r}")

    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [
        settings.python,
        str(settings.inference_script),
        "--calculation-type", calculation_type,
        "--protein", str(protein_path),
        "--ligands-dir", str(ligands_dir),
        "--output-dir", str(output_dir),
        # Shared base fields
        "--water-model", req.water_model,
        "--ligand-ff-type", req.ligand_ff_type,
        "--protein-ff-code", req.protein_ff_code,
        "--dt-max", str(req.dt_max),
        "--solv-d", str(req.solv_d),
        "--solv-bt", req.solv_bt,
        "--solv-rmin", str(req.solv_rmin),
        "--solv-ion-conc", str(req.solv_ion_conc),
        "--host-name", req.host_name,
        "--host-selection", req.host_selection,
        "--cofactor-selection", req.cofactor_selection,
        "--threads", str(req.threads),
        "--num-jobs", str(req.num_jobs),
        "--replicas", str(req.replicas),
        "--scheduler", req.scheduler,
    ]

    # Boolean flags
    argv.append("--fix-protein" if req.fix_protein else "--no-fix-protein")
    argv.append("--cofactor-on-protein" if req.cofactor_on_protein else "--no-cofactor-on-protein")
    argv.append("--submit" if req.submit else "--no-submit")

    # Optional fields
    if req.ligand_ff_code is not None:
        argv += ["--ligand-ff-code", req.ligand_ff_code]
    if req.hmr_factor is not None:
        argv += ["--hmr-factor", str(req.hmr_factor)]
    if req.job_prefix:
        argv += ["--job-prefix", req.job_prefix]
    if req.global_config_yaml:
        yaml_path = job_dir / "input" / "global_config.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(req.global_config_yaml, encoding="utf-8")
        argv += ["--global-config-yaml", str(yaml_path)]

    # Optional structural inputs
    if cofactor_path is not None:
        argv += ["--cofactor", str(cofactor_path)]
    if membrane_path is not None:
        argv += ["--membrane", str(membrane_path)]
    if custom_ff_dir is not None:
        argv += ["--custom-ff-dir", str(custom_ff_dir)]
    if topology_dir is not None:
        argv += ["--topology-dir", str(topology_dir)]

    # Calculation-type specific fields
    if isinstance(req, FepCalculateRequest):
        argv += [
            "--nwindows-ligand-vdw", str(req.nwindows_ligand_vdw),
            "--nwindows-ligand-coul", str(req.nwindows_ligand_coul),
            "--nwindows-complex-vdw", str(req.nwindows_complex_vdw),
            "--nwindows-complex-coul", str(req.nwindows_complex_coul),
            "--nwindows-complex-bonded", str(req.nwindows_complex_bonded),
        ]
    if isinstance(req, MmpbsaCalculateRequest):
        argv += ["--samples", str(req.samples)]
        if req.mmpbsa_yaml:
            mmpbsa_yaml_path = job_dir / "input" / "mmpbsa.yaml"
            mmpbsa_yaml_path.parent.mkdir(parents=True, exist_ok=True)
            mmpbsa_yaml_path.write_text(req.mmpbsa_yaml, encoding="utf-8")
            argv += ["--mmpbsa-yaml", str(mmpbsa_yaml_path)]

    return argv


def list_ligands(ligands_dir: Path) -> list[Path]:
    """Return sorted list of ligand files under `ligands_dir`.

    Accepts `.sdf`, `.mol`, `.mol2`.  Non-recursive by design — one flat dir.
    """
    if not ligands_dir.is_dir():
        raise FileNotFoundError(f"ligands directory not found: {ligands_dir}")
    hits = sorted(
        p for p in ligands_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _LIGAND_SUFFIXES
    )
    if not hits:
        raise ValueError(f"no ligand SDF/MOL/MOL2 files under {ligands_dir}")
    return hits


_SAFE_LIGAND_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def sanitize_ligand_filename(name: str) -> str:
    """Reject filenames that are unsafe for use as snakemake wildcards.

    BindFlow uses `ligand_file.stem` as a subdirectory name; if the stem
    contains slashes, whitespace, or shell meta-chars, downstream rules
    misbehave.  Reject aggressively.
    """
    if not _SAFE_LIGAND_NAME.match(name):
        raise ValueError(
            f"ligand filename {name!r} must match [A-Za-z0-9._-]+; "
            f"rename before uploading."
        )
    return name


__all__ = [
    "calculate_argv",
    "list_ligands",
    "sanitize_ligand_filename",
]
