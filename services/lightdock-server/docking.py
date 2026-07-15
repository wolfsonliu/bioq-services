"""Subprocess wrapper for lightdock-server (approach A -- no upstream edits).

Orchestrates the multi-step LightDock protocol by driving the upstream `lgd_*`
console scripts (installed with their `.py` suffix by LightDock's setup.py):

    lgd_setup.py                 -> setup.json + swarm_*/ + lightdock_*.pdb
    lgd_run.py                   -> swarm_*/gso_<steps>.out
    lgd_generate_conformations.py-> swarm_*/lightdock_<i>.pdb   (per swarm)
    lgd_cluster_bsas.py          -> swarm_*/cluster.repr        (per swarm)
    lgd_rank.py                  -> rank_by_scoring.list
    lgd_top.py                   -> top_<n>.pdb

Every lgd_* script is heavily CWD-dependent (relative swarm_*/ paths, the
`lightdock_` filename prefix, setup.json references), so the driver chdir's into
the staged workdir and refers to receptor/ligand by bare filename. Ranked
results are then copied into <output-dir>.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def list_scoring_functions() -> list[str]:
    """Best-effort discovery of installed LightDock scoring functions.

    Returns [] when lightdock is not importable (e.g. offline unit tests), in
    which case callers should skip validation rather than fail.
    """
    try:
        spec = importlib.util.find_spec("lightdock.scoring")
    except (ImportError, ValueError, ModuleNotFoundError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    scoring_dir = Path(list(spec.submodule_search_locations)[0])
    return sorted(
        p.name
        for p in scoring_dir.iterdir()
        if p.is_dir() and not p.name.startswith("__")
    )


def lightdock_version() -> str | None:
    try:
        from lightdock.version import CURRENT_VERSION

        return CURRENT_VERSION
    except Exception:
        return None


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a subprocess, teeing its command line into our log."""
    print(f"[lightdock-wrapper] $ {' '.join(argv)}", flush=True)
    return subprocess.run(argv, text=True, **kw)


def _script(bin_dir: Path, name: str) -> str:
    """Absolute path to an lgd_* console script (installed with .py suffix)."""
    return str(bin_dir / name)


def _cmd_dock(args: argparse.Namespace) -> int:
    bin_dir = Path(args.bin_dir)
    work = Path(args.workdir).resolve()
    out = Path(args.output_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # Validate scoring against the installed set (skip if undiscoverable).
    available = list_scoring_functions()
    if available and args.scoring not in available:
        print(
            f"[lightdock-wrapper] ERROR: unknown scoring function '{args.scoring}'. "
            f"Available: {', '.join(available)}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    # Stage inputs into workdir under clean, bare filenames.
    receptor = work / "receptor.pdb"
    ligand = work / "ligand.pdb"
    shutil.copy2(args.receptor, receptor)
    shutil.copy2(args.ligand, ligand)
    restraints = None
    if args.restraints:
        restraints = work / "restraints.list"
        shutil.copy2(args.restraints, restraints)

    py = sys.executable
    prev_cwd = os.getcwd()
    os.chdir(work)
    try:
        # 1. setup
        setup_argv = [
            py, _script(bin_dir, "lgd_setup.py"),
            "receptor.pdb", "ligand.pdb",
            "-g", str(args.glowworms),
            "--seed_points", str(args.swarm_seed),
        ]
        if args.swarms > 0:
            setup_argv += ["-s", str(args.swarms)]
        if restraints is not None:
            setup_argv += ["-r", "restraints.list"]
        if args.anm:
            setup_argv.append("-anm")
        if args.noxt:
            setup_argv.append("--noxt")
        if args.noh:
            setup_argv.append("--noh")
        if args.now:
            setup_argv.append("--now")
        if _run(setup_argv).returncode != 0:
            return 1

        setup_file = work / "setup.json"
        if not setup_file.is_file():
            print("[lightdock-wrapper] ERROR: setup.json not produced", file=sys.stderr)
            return 1
        setup = json.loads(setup_file.read_text())
        num_swarms = int(setup.get("swarms", args.swarms))
        if num_swarms <= 0:
            print("[lightdock-wrapper] ERROR: setup produced 0 swarms", file=sys.stderr)
            return 1
        print(f"[lightdock-wrapper] setup produced {num_swarms} swarms", flush=True)

        # 2. run GSO (all swarms)
        run_argv = [
            py, _script(bin_dir, "lgd_run.py"),
            "setup.json", str(args.steps),
            "-c", str(args.cores),
            "-s", args.scoring,
            "-sg", str(args.gso_seed),
        ]
        if _run(run_argv).returncode != 0:
            return 1

        gso_name = f"gso_{args.steps}.out"

        # 3. per-swarm conformation generation + clustering
        for swarm_id in range(num_swarms):
            swarm_dir = work / f"swarm_{swarm_id}"
            gso_out = swarm_dir / gso_name
            if not gso_out.is_file():
                print(
                    f"[lightdock-wrapper] WARN: {gso_out} missing, skipping swarm",
                    flush=True,
                )
                continue
            conf_argv = [
                py, _script(bin_dir, "lgd_generate_conformations.py"),
                "receptor.pdb", "ligand.pdb",
                str(gso_out), str(args.glowworms),
                "--setup", "setup.json",
            ]
            _run(conf_argv)
            cluster_argv = [
                py, _script(bin_dir, "lgd_cluster_bsas.py"),
                str(gso_out), "-c", str(args.cluster_cutoff),
            ]
            _run(cluster_argv)

        # 4. global ranking
        rank_argv = [
            py, _script(bin_dir, "lgd_rank.py"),
            str(num_swarms), str(args.steps),
        ]
        if _run(rank_argv).returncode != 0:
            return 1
        ranking = work / "rank_by_scoring.list"
        if not ranking.is_file():
            print("[lightdock-wrapper] ERROR: ranking not produced", file=sys.stderr)
            return 1

        # 5. top-N generation
        top_argv = [
            py, _script(bin_dir, "lgd_top.py"),
            "receptor.pdb", "ligand.pdb",
            "rank_by_scoring.list", str(args.top),
            "--setup", "setup.json",
        ]
        if _run(top_argv).returncode != 0:
            return 1

        # 6. collect results into output/
        top_out = out / "top"
        top_out.mkdir(parents=True, exist_ok=True)
        top_pdbs = sorted(work.glob("top_*.pdb"))
        for pdb in top_pdbs:
            shutil.copy2(pdb, top_out / pdb.name)
        for name in ("rank_by_scoring.list", "rank_by_luciferin.list",
                     "rank_by_rmsd.list", "setup.json"):
            src = work / name
            if src.is_file():
                shutil.copy2(src, out / name)

        if not top_pdbs:
            print("[lightdock-wrapper] ERROR: no top_*.pdb generated", file=sys.stderr)
            return 1
        print(
            f"[lightdock-wrapper] wrote {len(top_pdbs)} ranked poses to {top_out}",
            flush=True,
        )
        return 0
    finally:
        os.chdir(prev_cwd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lightdock-wrapper")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dock")
    d.add_argument("--receptor", required=True)
    d.add_argument("--ligand", required=True)
    d.add_argument("--restraints", default=None)
    d.add_argument("--workdir", required=True)
    d.add_argument("--output-dir", required=True)
    d.add_argument("--bin-dir", required=True)
    d.add_argument("--swarms", type=int, default=0)
    d.add_argument("--glowworms", type=int, default=200)
    d.add_argument("--steps", type=int, default=100)
    d.add_argument("--scoring", default="fastdfire")
    d.add_argument("--cores", type=int, default=8)
    d.add_argument("--top", type=int, default=10)
    d.add_argument("--swarm-seed", type=int, default=324324)
    d.add_argument("--gso-seed", type=int, default=324324)
    d.add_argument("--cluster-cutoff", type=float, default=4.0)
    d.add_argument("--anm", action="store_true")
    d.add_argument("--noxt", action="store_true")
    d.add_argument("--noh", action="store_true")
    d.add_argument("--now", action="store_true")
    d.set_defaults(func=_cmd_dock)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
