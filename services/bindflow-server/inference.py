"""Wrapper script — the subprocess entry point for bindflow-server.

Called as::

    python -m server ...        (via inference.py's main)
    /opt/conda/envs/bindflow/bin/python /opt/bindflow/server/inference.py ...
        (from tools.calculate_argv → SubprocessRunner)

Responsibilities:

1. Parse a flat CLI mirroring `BaseCalculateRequest` + calc-type extras.
2. Validate inputs (files exist, ff codes look sane) — cheap; fail fast.
3. `os.chdir(output_dir)` so snakemake writes .snakemake / lockfiles there.
4. Import `bindflow.runners.calculate` (deferred — heavy import).
5. Build `global_config` dict + `ligands` list; call `calculate(...)`.

Upstream source is NOT modified.  If the wrapper needs to grow, keep it
here — patches to upstream are last resort (design doc §6.1).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

try:  # optional (only needed when --global-config-yaml / --mmpbsa-yaml supplied)
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — yaml ships with the conda env
    yaml = None  # type: ignore[assignment]


LOG = logging.getLogger("bindflow-server.inference")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _bool_flag(parser: argparse.ArgumentParser, name: str, *, default: bool) -> None:
    """Add mutually exclusive --name / --no-name flags."""
    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=default)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BindFlow wrapper: run FEP / MMPBSA workflow via bindflow.runners.calculate."
    )
    p.add_argument("--calculation-type", required=True, choices=("fep", "mmpbsa"))
    p.add_argument("--protein", required=True, type=Path)
    p.add_argument("--ligands-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)

    # Shared base fields
    p.add_argument("--water-model", required=True)
    p.add_argument("--ligand-ff-type", required=True, choices=("openff", "gaff"))
    p.add_argument("--ligand-ff-code", default=None)
    p.add_argument("--protein-ff-code", default="amber99sb-ildn")
    p.add_argument("--hmr-factor", type=float, default=None)
    p.add_argument("--dt-max", type=float, default=0.004)
    p.add_argument("--solv-d", type=float, default=1.5)
    p.add_argument("--solv-bt", default="dodecahedron")
    p.add_argument("--solv-rmin", type=float, default=1.0)
    p.add_argument("--solv-ion-conc", type=float, default=0.15)
    p.add_argument("--host-name", default="Protein")
    p.add_argument("--host-selection", default="protein and name CA")
    _bool_flag(p, "fix-protein", default=True)
    _bool_flag(p, "cofactor-on-protein", default=True)
    p.add_argument("--cofactor-selection", default="resname COF")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--num-jobs", type=int, default=10)
    p.add_argument("--replicas", type=int, default=3)
    p.add_argument("--job-prefix", default=None)
    p.add_argument("--scheduler", default="frontend", choices=("frontend", "slurm"))
    _bool_flag(p, "submit", default=True)
    p.add_argument("--global-config-yaml", type=Path, default=None)

    # Optional structural inputs
    p.add_argument("--cofactor", type=Path, default=None)
    p.add_argument("--membrane", type=Path, default=None)
    p.add_argument("--custom-ff-dir", type=Path, default=None)
    p.add_argument("--topology-dir", type=Path, default=None)

    # FEP-specific
    p.add_argument("--nwindows-ligand-vdw", type=int, default=11)
    p.add_argument("--nwindows-ligand-coul", type=int, default=11)
    p.add_argument("--nwindows-complex-vdw", type=int, default=21)
    p.add_argument("--nwindows-complex-coul", type=int, default=11)
    p.add_argument("--nwindows-complex-bonded", type=int, default=11)

    # MMPBSA-specific
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--mmpbsa-yaml", type=Path, default=None)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation (cheap; runs before heavy imports)
# ---------------------------------------------------------------------------


def validate(args: argparse.Namespace) -> None:
    if not args.protein.exists():
        raise SystemExit(f"protein file not found: {args.protein}")
    if not args.ligands_dir.is_dir():
        raise SystemExit(f"ligands dir not found: {args.ligands_dir}")
    _LIGAND_SUFFIXES = {".sdf", ".mol", ".mol2"}
    lig_files = sorted(
        p for p in args.ligands_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _LIGAND_SUFFIXES
    )
    if not lig_files:
        raise SystemExit(f"no SDF/MOL/MOL2 files under {args.ligands_dir}")
    args._ligand_files = lig_files  # stashed for build_ligands_list

    if args.cofactor is not None and not args.cofactor.exists():
        raise SystemExit(f"cofactor file not found: {args.cofactor}")
    if args.membrane is not None and not args.membrane.exists():
        raise SystemExit(f"membrane file not found: {args.membrane}")
    if args.custom_ff_dir is not None and not args.custom_ff_dir.is_dir():
        raise SystemExit(f"custom-ff-dir not found: {args.custom_ff_dir}")
    if args.global_config_yaml is not None and not args.global_config_yaml.exists():
        raise SystemExit(f"global-config-yaml not found: {args.global_config_yaml}")
    if args.mmpbsa_yaml is not None and not args.mmpbsa_yaml.exists():
        raise SystemExit(f"mmpbsa-yaml not found: {args.mmpbsa_yaml}")

    if args.hmr_factor is not None and args.hmr_factor < 2.0 and args.dt_max > 0.002:
        raise SystemExit(
            f"hmr_factor={args.hmr_factor} < 2.0 requires dt_max <= 0.002 "
            f"(got {args.dt_max}); adjust one of them."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise SystemExit(
            f"pyyaml not available; cannot load {path}.  Ensure the conda env includes pyyaml."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML at {path} must parse to a mapping, got {type(data).__name__}")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` into `base` recursively; override wins on leaves."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_ligands_list(args: argparse.Namespace) -> list[dict]:
    """One `{conf, ff}` dict per ligand file."""
    ff_block: dict[str, Any] = {"type": args.ligand_ff_type}
    if args.ligand_ff_code:
        ff_block["code"] = args.ligand_ff_code
    return [{"conf": str(p), "ff": ff_block} for p in args._ligand_files]


def build_global_config(args: argparse.Namespace) -> dict:
    """Build the BindFlow `global_config` dict.

    Fields set here layer under the YAML escape hatch (yaml overrides win on
    same keys) — that's the opposite of the design doc's initial phrasing,
    but the practical convention we adopt: YAML is for "power users" who
    want to control cluster/mdrun/mdp/extra_directives; pydantic fields cover
    the common path.  Users who set both should read the merge order.
    """
    cfg: dict[str, Any] = {}
    if args.calculation_type == "mmpbsa":
        cfg["samples"] = args.samples
        if args.mmpbsa_yaml is not None:
            cfg["mmpbsa"] = _load_yaml(args.mmpbsa_yaml)

    if args.calculation_type == "fep":
        cfg["nwindows"] = {
            "ligand": {
                "vdw": args.nwindows_ligand_vdw,
                "coul": args.nwindows_ligand_coul,
            },
            "complex": {
                "vdw": args.nwindows_complex_vdw,
                "coul": args.nwindows_complex_coul,
                "bonded": args.nwindows_complex_bonded,
            },
        }

    # A minimal cluster/options/calculation block for FrontEnd scheduler.  BindFlow's
    # `approach_flow` reads `cluster.options.calculation` unconditionally; if we
    # leave it empty, upstream raises when SlurmScheduler asks for `partition`.
    # For FrontEnd this block is inert (frontend scheduler ignores it).
    cfg["cluster"] = {"options": {"calculation": {}}}
    cfg["extra_directives"] = {}

    if args.global_config_yaml is not None:
        cfg = _deep_merge(cfg, _load_yaml(args.global_config_yaml))
    return cfg


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    args = parse_args(argv)
    validate(args)

    # Chdir into output dir so snakemake writes cache alongside results.
    os.chdir(args.output_dir)

    # Heavy imports deferred to here — validation errors surface without paying
    # 15-30s for pytorch / openff / mdanalysis import.
    from bindflow.orchestration.generate_scheduler import (  # noqa: E402
        FrontEnd,
        SlurmScheduler,
    )
    from bindflow.runners import calculate  # noqa: E402

    scheduler_cls = SlurmScheduler if args.scheduler == "slurm" else FrontEnd

    ligands = build_ligands_list(args)
    global_config = build_global_config(args)

    kwargs: dict[str, Any] = dict(
        calculation_type=args.calculation_type,
        protein=str(args.protein),
        ligands=ligands,
        water_model=args.water_model,
        custom_ff_path=str(args.custom_ff_dir) if args.custom_ff_dir else None,
        host_name=args.host_name,
        host_selection=args.host_selection,
        fix_protein=args.fix_protein,
        solv_d=args.solv_d,
        solv_bt=args.solv_bt,
        solv_rmin=args.solv_rmin,
        solv_ion_conc=args.solv_ion_conc,
        cofactor_on_protein=args.cofactor_on_protein,
        cofactor_selection=args.cofactor_selection,
        hmr_factor=args.hmr_factor,
        dt_max=args.dt_max,
        threads=args.threads,
        num_jobs=args.num_jobs,
        replicas=args.replicas,
        scheduler_class=scheduler_cls,
        debug=False,
        job_prefix=args.job_prefix,
        out_root_folder_path=str(args.output_dir),
        submit=args.submit,
        global_config=global_config,
    )
    if args.cofactor:
        kwargs["cofactor"] = str(args.cofactor)
    if args.membrane:
        kwargs["membrane"] = str(args.membrane)

    LOG.info(
        "Invoking bindflow.runners.calculate: %s",
        json.dumps(
            {k: v for k, v in kwargs.items() if k not in {"ligands", "global_config"}},
            default=str, indent=2,
        ),
    )
    calculate(**kwargs)
    LOG.info("bindflow.runners.calculate returned; output dir: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
