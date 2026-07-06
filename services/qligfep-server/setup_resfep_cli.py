"""Wrapper for qligfep QresFEP.py — residue mutation FEP setup."""
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
    p.add_argument("--mutation", required=True)
    p.add_argument("--mutchain", required=True)
    p.add_argument("--protprep-dir", type=Path, required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--dual", action="store_true")
    p.add_argument("--shell-rest", type=float, required=True)
    p.add_argument("--tripeptide", action="store_true")
    p.add_argument("--cofactor", action="append", default=[], dest="cofactors")
    p.add_argument("--forcefield", required=True)
    p.add_argument("--windows", type=int, required=True)
    p.add_argument("--sampling", required=True)
    p.add_argument("--timestep", required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--replicates", type=int, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--cluster", required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def run(*, mutation, mutchain, protprep_dir, system, dual, shell_rest,
        tripeptide, cofactors, forcefield, windows, sampling, timestep,
        temperature, replicates, start, cluster, work_dir, output_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("protein.pdb", "water.pdb"):
        src = protprep_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, work_dir / name)

    script = vendor_shim.script("QresFEP.py")
    argv = [
        sys.executable, str(script),
        "-m", mutation, "-mc", mutchain, "-S", system,
        "-sh", str(shell_rest),
        "-f", forcefield, "-w", str(windows),
        "-s", sampling, "-l", start, "-ts", timestep,
        "-T", str(temperature), "-r", str(replicates), "-C", cluster,
    ]
    if dual:
        argv += ["-d"]
    if tripeptide:
        argv += ["-t"]
    for c in cofactors or []:
        argv += ["-c", c]

    r = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    (output_dir / "setup.log").write_text(
        r.stdout + "\n---STDERR---\n" + r.stderr
    )
    if r.returncode != 0:
        return r.returncode

    setup_root = work_dir / mutation
    if not setup_root.exists():
        setup_root = work_dir
    for leg in ("1.protein", "2.water"):
        src = setup_root / leg
        if src.exists():
            shutil.copytree(src, output_dir / leg, dirs_exist_ok=True)

    (output_dir / "setup.json").write_text(json.dumps({
        "mutation": mutation, "mutchain": mutchain, "system": system,
        "dual": dual, "shell_rest": shell_rest, "tripeptide": tripeptide,
        "cofactors": cofactors, "forcefield": forcefield, "windows": windows,
        "sampling": sampling, "timestep": timestep, "temperature": temperature,
        "replicates": replicates, "start": start, "cluster": cluster,
    }, indent=2))
    return 0


def main():
    a = parse_args()
    return run(
        mutation=a.mutation, mutchain=a.mutchain, protprep_dir=a.protprep_dir,
        system=a.system, dual=a.dual, shell_rest=a.shell_rest,
        tripeptide=a.tripeptide, cofactors=a.cofactors,
        forcefield=a.forcefield, windows=a.windows,
        sampling=a.sampling, timestep=a.timestep, temperature=a.temperature,
        replicates=a.replicates, start=a.start, cluster=a.cluster,
        work_dir=a.work_dir, output_dir=a.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
