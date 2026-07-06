"""Wrapper for qligfep QLIE.py — Linear Interaction Energy setup."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import vendor_shim


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ligand-name", required=True)
    p.add_argument("--forcefield", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--cofactor", action="append", default=[], dest="cofactors")
    p.add_argument("--radius", type=float, required=True)
    p.add_argument("--time-ns", type=float, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--replicates", type=int, required=True)
    p.add_argument("--cluster", required=True)
    p.add_argument("--preplocation", required=True)
    p.add_argument("--ligprep-dir", type=Path, required=True)
    p.add_argument("--protprep-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def run(*, ligand_name, forcefield, system, cofactors, radius, time_ns,
        temperature, replicates, cluster, preplocation,
        ligprep_dir, protprep_dir, work_dir, output_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("lib", "prm", "pdb"):
        src = ligprep_dir / f"{ligand_name}.{ext}"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, work_dir / f"{ligand_name}.{ext}")
    for name in ("protein.pdb", "water.pdb"):
        src = protprep_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, work_dir / name)

    script = vendor_shim.script("QLIE.py")
    argv = [
        sys.executable, str(script),
        "-l", ligand_name, "-f", forcefield, "-S", system,
        "-R", str(radius), "-t", str(time_ns),
        "-T", str(temperature), "-r", str(replicates),
        "-C", cluster, "-P", preplocation,
    ]
    for c in cofactors or []:
        argv += ["-c", c]

    r = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    (output_dir / "setup.log").write_text(
        r.stdout + "\n---STDERR---\n" + r.stderr
    )
    if r.returncode != 0:
        return r.returncode

    for name in ("inputfiles", "md_LIE_bound", "md_LIE_free"):
        src = work_dir / name
        if src.exists():
            shutil.copytree(src, output_dir / name, dirs_exist_ok=True)

    (output_dir / "setup.json").write_text(json.dumps({
        "ligand": ligand_name, "forcefield": forcefield, "system": system,
        "cofactors": cofactors, "radius": radius, "time_ns": time_ns,
        "temperature": temperature, "replicates": replicates,
        "cluster": cluster, "preplocation": preplocation,
    }, indent=2))
    return 0


def main():
    a = parse_args()
    return run(
        ligand_name=a.ligand_name, forcefield=a.forcefield, system=a.system,
        cofactors=a.cofactors, radius=a.radius, time_ns=a.time_ns,
        temperature=a.temperature, replicates=a.replicates,
        cluster=a.cluster, preplocation=a.preplocation,
        ligprep_dir=a.ligprep_dir, protprep_dir=a.protprep_dir,
        work_dir=a.work_dir, output_dir=a.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
