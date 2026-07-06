"""Wrapper for qligfep protprep.py — spherical boundary prep."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import vendor_shim


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protein-pdb", type=Path, required=True)
    p.add_argument("--sphere-radius", type=float, required=True)
    p.add_argument("--sphere-center", required=True)
    p.add_argument("--forcefield", required=True)
    p.add_argument("--mutchain", default=None)
    p.add_argument("--nowater", action="store_true")
    p.add_argument("--noclean", action="store_true")
    p.add_argument("--preplocation", default="LOCAL")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def run(protein_pdb: Path, sphere_radius: float, sphere_center: str,
        forcefield: str, mutchain: str | None,
        nowater: bool, noclean: bool, preplocation: str,
        work_dir: Path, output_dir: Path) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = work_dir / protein_pdb.name
    shutil.copy2(protein_pdb, staged)

    script = vendor_shim.script("protprep.py")
    argv = [
        sys.executable, str(script),
        "-p", str(staged.name),
        "-r", str(sphere_radius),
        "-c", sphere_center,
        "-f", forcefield,
        "-P", preplocation,
    ]
    if mutchain:
        argv += ["-mc", mutchain]
    if nowater:
        argv += ["-w"]
    if noclean:
        argv += ["--noclean"]

    r = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    (output_dir / "protprep.log").write_text(
        r.stdout + "\n---STDERR---\n" + r.stderr
    )
    if r.returncode != 0:
        return r.returncode

    for name in ("protein.pdb", "water.pdb"):
        src = work_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    (output_dir / "system.json").write_text(json.dumps({
        "sphere_center": sphere_center,
        "sphere_radius": sphere_radius,
        "forcefield": forcefield,
        "preplocation": preplocation,
        "mutchain": mutchain,
        "nowater": nowater,
    }, indent=2))
    return 0


def main() -> int:
    a = parse_args()
    return run(
        a.protein_pdb, a.sphere_radius, a.sphere_center,
        a.forcefield, a.mutchain, a.nowater, a.noclean, a.preplocation,
        a.work_dir, a.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
