"""Thin wrapper around the vendored `openbpmd` for openbpmd-server.

Why a wrapper instead of calling upstream `openbpmd.py` directly?

  1. Upstream does `from simtk.openmm import *` at module top.  The `simtk`
     namespace was REMOVED in OpenMM 8.x, so a modern conda openmm can't
     import upstream unchanged.  We inject a `simtk` -> `openmm` alias shim
     (exactly what OpenMM's own 7.x compat layer did) BEFORE importing the
     vendored module — upstream source stays 0-modified for this.  Pinning
     the ancient OpenMM 7.7 (which keeps `simtk`) is rejected: it lacks CUDA
     kernels for sm_86/89/90 GPUs.  See design doc §6.1.
  2. Upstream hardcodes `Platform.getPlatformByName('CUDA')` in 3 places.
     We monkeypatch it to honour `--platform` so offline smoke can run CPU.
     See design doc §6.2.
  3. Upstream routes output via `args.output`; we route to `<job_dir>/output`
     and additionally emit `scoring_stats.json` for the adapter / agent.

The advanced `--sim-ns` / `--equil-steps` knobs set the vendored module's
`PROD_SIM_NS` / `EQUIL_STEPS` globals (introduced by patches/0001-*.patch,
defaults preserve upstream behaviour).  They exist for fast integration
tests — non-standard values break BPMD score comparability.

Exit codes:
  0 = results.csv written (at least one full rep + aggregation)
  1 = ran but no results.csv (adapter treats as FAILED)
  2 = pre-flight validation failed (missing input / bad args)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

_AMBER_STRUCT = {".rst7", ".inpcrd", ".ncrst"}
_AMBER_PARM = {".prm7", ".parm7", ".prmtop", ".top7"}
_GROMACS_STRUCT = {".gro"}
_GROMACS_PARM = {".top"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenBPMD binding-pose metadynamics stability scoring "
        "(single pre-solvated complex).",
    )
    p.add_argument("--structure", type=Path, required=True,
                   help="Coordinate file (.rst7 Amber or .gro Gromacs).")
    p.add_argument("--parameters", type=Path, required=True,
                   help="Topology/parameter file (.prm7 Amber or .top Gromacs).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory (results.csv + rep_*/ written here).")
    p.add_argument("--lig-resname", type=str, default="MOL")
    p.add_argument("--nreps", type=int, default=10)
    p.add_argument("--hill-height", type=float, default=0.3)
    p.add_argument("--platform", type=str, default="CUDA",
                   help="OpenMM platform (CUDA/CPU/OpenCL).")
    p.add_argument("--system-format", choices=["amber", "gromacs"], default=None,
                   help="Force format; default auto-detect by extension.")
    # Advanced / testing (default None -> upstream standard values).
    p.add_argument("--sim-ns", type=float, default=None,
                   help="ADVANCED: production length in ns (default 10).")
    p.add_argument("--equil-steps", type=int, default=None,
                   help="ADVANCED: equilibration steps (default 250000).")
    return p.parse_args(argv)


def _detect_format(args: argparse.Namespace) -> str:
    if args.system_format:
        return args.system_format
    return "gromacs" if args.structure.suffix.lower() in _GROMACS_STRUCT else "amber"


def validate(args: argparse.Namespace) -> None:
    """Fail fast (rc=2) before paying the ~20 s OpenMM import cost."""
    for path, label in ((args.structure, "structure"),
                        (args.parameters, "parameters")):
        if not path.exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(2)

    fmt = _detect_format(args)
    s_ext = args.structure.suffix.lower()
    p_ext = args.parameters.suffix.lower()
    if fmt == "gromacs":
        ok = s_ext in _GROMACS_STRUCT and p_ext in _GROMACS_PARM
    else:
        ok = s_ext in _AMBER_STRUCT and p_ext in _AMBER_PARM
    if not ok:
        print(
            f"ERROR: structure/parameters extensions ({s_ext}, {p_ext}) do not "
            f"form a valid {fmt} pair. Expected Amber (.rst7/.prm7) or "
            f"Gromacs (.gro/.top).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.nreps < 1:
        print(f"ERROR: --nreps must be >= 1; got {args.nreps}", file=sys.stderr)
        sys.exit(2)
    if not (0.0 < args.hill_height <= 5.0):
        print(f"ERROR: --hill-height out of range: {args.hill_height}",
              file=sys.stderr)
        sys.exit(2)

    args.output_dir.mkdir(parents=True, exist_ok=True)


def _install_simtk_shim() -> None:
    """Alias `simtk.*` to `openmm.*` so upstream's star-imports work on 8.x."""
    import openmm
    import openmm.app
    import openmm.app.metadynamics
    import openmm.unit

    simtk = types.ModuleType("simtk")
    simtk.openmm = openmm
    simtk.unit = openmm.unit
    sys.modules["simtk"] = simtk
    sys.modules["simtk.openmm"] = openmm
    sys.modules["simtk.openmm.app"] = openmm.app
    sys.modules["simtk.unit"] = openmm.unit
    sys.modules["simtk.openmm.app.metadynamics"] = openmm.app.metadynamics


def _install_platform_override(platform_name: str) -> None:
    """Make upstream's hardcoded getPlatformByName('CUDA') honour our choice."""
    from openmm import Platform

    original = Platform.getPlatformByName

    def _get(_name: str):
        return original(os.environ.get("OPENBPMD_PLATFORM", platform_name))

    Platform.getPlatformByName = staticmethod(_get)


def _import_upstream():
    upstream_dir = os.environ.get("OPENBPMD_UPSTREAM_DIR")
    if upstream_dir:
        root = Path(upstream_dir)
    else:
        root = Path(__file__).resolve().parent.parent / "upstream"
    sys.path.insert(0, str(root))
    import openbpmd  # noqa: E402  (vendored upstream module)
    return openbpmd


def _write_stats(output_dir: Path, args: argparse.Namespace,
                 wall_time_s: float, platform_name: str) -> bool:
    """Emit scoring_stats.json; return True iff results.csv exists."""
    import pandas as pd

    results_csv = output_dir / "results.csv"
    reps_done = sum(
        1 for d in output_dir.glob("rep_*")
        if (d / "bpm_results.csv").exists()
    )
    stats: dict = {
        "nreps_requested": args.nreps,
        "nreps_done": reps_done,
        "lig_resname": args.lig_resname,
        "hill_height": args.hill_height,
        "sim_ns": args.sim_ns if args.sim_ns is not None else 10.0,
        "platform": platform_name,
        "wall_time_s": round(wall_time_s, 2),
        "results_written": results_csv.exists(),
    }
    if results_csv.exists():
        row = pd.read_csv(results_csv).iloc[0].to_dict()
        stats.update({
            "comp_score": row.get("CompScore"),
            "comp_score_sd": row.get("CompScoreSD"),
            "pose_score": row.get("PoseScore"),
            "pose_score_sd": row.get("PoseScoreSD"),
            "contact_score": row.get("ContactScore"),
            "contact_score_sd": row.get("ContactScoreSD"),
        })
    (output_dir / "scoring_stats.json").write_text(json.dumps(stats, indent=2))
    return results_csv.exists()


def main() -> int:
    args = parse_args()
    validate(args)

    # Heavy imports deferred until after validation.
    _install_simtk_shim()
    _install_platform_override(args.platform)
    openbpmd = _import_upstream()

    # Apply advanced length overrides onto the (patched) module globals.
    if args.sim_ns is not None:
        openbpmd.PROD_SIM_NS = args.sim_ns
    if args.equil_steps is not None:
        openbpmd.EQUIL_STEPS = args.equil_steps

    upstream_args = types.SimpleNamespace(
        structure=str(args.structure),
        parameters=str(args.parameters),
        output=str(args.output_dir),
        lig_resname=args.lig_resname,
        nreps=args.nreps,
        hill_height=args.hill_height,
    )

    t0 = time.time()
    openbpmd.main(upstream_args)
    wall = time.time() - t0

    ok = _write_stats(args.output_dir, args, wall, args.platform)
    if not ok:
        print(
            "ERROR: no results.csv produced — metadynamics may have been "
            "interrupted before all reps completed.",
            file=sys.stderr,
        )
        return 1

    print(f"OpenBPMD done: {args.nreps} reps in {wall:.1f}s "
          f"-> {args.output_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
