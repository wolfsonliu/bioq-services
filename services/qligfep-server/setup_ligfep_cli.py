"""Wrapper for qligfep QligFEP.py — dual-topology ligand FEP setup."""
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
    p.add_argument("--lig1-name", required=True)
    p.add_argument("--lig2-name", required=True)
    p.add_argument("--ligprep-dir", type=Path, required=True)
    p.add_argument("--protprep-dir", type=Path, required=True)
    p.add_argument("--forcefield", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--replicates", type=int, required=True)
    p.add_argument("--windows", type=int, required=True)
    p.add_argument("--sampling", required=True)
    p.add_argument("--timestep", required=True)
    p.add_argument("--cluster", required=True)
    p.add_argument("--to-clean", action="store_true")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def stage_workdir(work_dir: Path, ligprep_dir: Path, protprep_dir: Path,
                  lig1: str, lig2: str) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("lib", "prm", "pdb"):
        for name in (lig1, lig2):
            src = ligprep_dir / f"{name}.{ext}"
            if not src.exists():
                raise FileNotFoundError(f"missing ligand param file: {src}")
            shutil.copy2(src, work_dir / f"{name}.{ext}")
    for name in ("protein.pdb", "water.pdb"):
        src = protprep_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing protprep file: {src}")
        shutil.copy2(src, work_dir / name)


def run(*, lig1_name: str, lig2_name: str,
        ligprep_dir: Path, protprep_dir: Path,
        forcefield: str, system: str, start: str,
        temperature: float, replicates: int, windows: int,
        sampling: str, timestep: str, cluster: str,
        to_clean: bool,
        work_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_workdir(work_dir, ligprep_dir, protprep_dir, lig1_name, lig2_name)

    script = vendor_shim.script("QligFEP.py")
    argv = [
        sys.executable, str(script),
        "-l1", lig1_name, "-l2", lig2_name,
        "-f", forcefield, "-S", system, "-C", cluster,
        "-l", start, "-T", str(temperature),
        "-r", str(replicates), "-s", sampling,
        "-w", str(windows), "-ts", timestep,
    ]
    if to_clean:
        argv += ["-clean", "true"]

    r = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    (output_dir / "setup.log").write_text(
        r.stdout + "\n---STDERR---\n" + r.stderr
    )
    if r.returncode != 0:
        return r.returncode

    setup_root = work_dir / f"{lig1_name}-{lig2_name}"
    if not setup_root.exists():
        # some upstream versions dump directly to CWD; fall back
        setup_root = work_dir
    for leg in ("1.protein", "2.water"):
        src = setup_root / leg
        if src.exists():
            shutil.copytree(src, output_dir / leg, dirs_exist_ok=True)

    (output_dir / "setup.json").write_text(json.dumps({
        "windows": windows, "replicates": replicates, "timestep": timestep,
        "lig1": lig1_name, "lig2": lig2_name, "forcefield": forcefield,
        "system": system, "temperature": temperature, "sampling": sampling,
        "start": start, "cluster": cluster,
    }, indent=2))
    return 0


def main() -> int:
    a = parse_args()
    return run(
        lig1_name=a.lig1_name, lig2_name=a.lig2_name,
        ligprep_dir=a.ligprep_dir, protprep_dir=a.protprep_dir,
        forcefield=a.forcefield, system=a.system, start=a.start,
        temperature=a.temperature, replicates=a.replicates, windows=a.windows,
        sampling=a.sampling, timestep=a.timestep, cluster=a.cluster,
        to_clean=a.to_clean,
        work_dir=a.work_dir, output_dir=a.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
