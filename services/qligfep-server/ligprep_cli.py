"""Wrapper: OpenFF ligand parameterization → qligfep openff2Q.py.

The OpenFF step generates a per-molecule .offxml + topology which upstream
``openff2Q.py`` consumes to produce Q ``.lib`` / ``.prm`` / ``.pdb`` files.

Heavy openff-toolkit imports are deferred to `_generate_offxml` so argument
validation failures return quickly instead of paying a ~20 s startup cost.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from . import vendor_shim


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ligand", type=Path, required=True)
    p.add_argument("--ligand-name", required=True)
    p.add_argument("--forcefield", default="openff-2.1.0")
    p.add_argument("--net-charge", type=int, default=None)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def _generate_offxml(ligand: Path, ligand_name: str, forcefield: str,
                     net_charge: int | None, work_dir: Path) -> Path:
    """Load ligand via OpenFF toolkit → per-molecule offxml + top."""
    from openff.toolkit import ForceField, Molecule  # deferred

    mol = Molecule.from_file(str(ligand))
    if net_charge is not None:
        mol.total_charge = net_charge
    ff = ForceField(f"{forcefield}.offxml")
    interchange = ff.create_interchange(mol.to_topology())
    offxml_path = work_dir / f"{ligand_name}.offxml"
    ff.to_file(offxml_path)
    top_path = work_dir / f"{ligand_name}.top"
    interchange.to_top(top_path)
    return offxml_path


def run(ligand: Path, ligand_name: str, forcefield: str, net_charge: int | None,
        work_dir: Path, output_dir: Path) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = work_dir / ligand.name
    shutil.copy2(ligand, staged)

    try:
        _generate_offxml(staged, ligand_name, forcefield, net_charge, work_dir)
    except Exception as e:
        print(f"OpenFF parameterization failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    script = vendor_shim.script("openff2Q.py")
    argv = [
        sys.executable, str(script),
        "-l", ligand_name,
        "-f", f"{ligand_name}.offxml",
    ]
    r = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    (output_dir / f"{ligand_name}.log").write_text(
        r.stdout + "\n---STDERR---\n" + r.stderr
    )
    if r.returncode != 0:
        return r.returncode

    for ext in ("lib", "prm", "pdb"):
        src = work_dir / f"{ligand_name}.{ext}"
        if src.exists():
            shutil.copy2(src, output_dir / f"{ligand_name}.{ext}")
    return 0


def main() -> int:
    a = parse_args()
    return run(a.ligand, a.ligand_name, a.forcefield, a.net_charge,
               a.work_dir, a.output_dir)


if __name__ == "__main__":
    sys.exit(main())
