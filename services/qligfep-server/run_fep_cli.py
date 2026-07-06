"""Wrapper: run one lambda window (qprep + eq*.inp + md_XXXX_YYYY.inp) with
qdyn / qdynp / qdyn_cuda.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--setup-dir", type=Path, required=True)
    p.add_argument("--window-idx", type=int, required=True)
    p.add_argument("--leg", choices=["protein", "water"], required=True)
    p.add_argument("--replicate-idx", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "mpi", "gpu"], required=True)
    p.add_argument("--nprocs", type=int, default=1)
    p.add_argument("--stage", choices=["eq", "md", "both"], required=True)
    p.add_argument("--keep-dcd", action="store_true")
    p.add_argument("--q-bin-dir", type=Path, default=Path("/opt/Q6/bin"))
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def _select_binary(device: str, q_bin_dir: Path) -> Path:
    return q_bin_dir / {"cpu": "qdyn", "mpi": "qdynp", "gpu": "qdyn_cuda"}[device]


def _mpi_prefix(nprocs: int) -> list[str]:
    if nprocs <= 1:
        return []
    return ["mpirun", "-np", str(nprocs)]


def _qprep(q_bin_dir: Path, work_dir: Path) -> int:
    qprep = str(q_bin_dir / "qprep")
    inp = work_dir / "qprep.inp"
    if not inp.exists():
        # some setups keep qprep.inp under inputfiles/
        return 0
    with open(inp) as f_in, open(work_dir / "qprep.log", "w") as f_out:
        r = subprocess.run([qprep], cwd=work_dir, stdin=f_in, stdout=f_out,
                           stderr=subprocess.STDOUT)
    return r.returncode


def _run_qdyn(binary: Path, nprocs: int, inp: Path, work_dir: Path) -> int:
    """Run one qdyn/qdynp/qdyn_cuda invocation on a .inp file, capturing stdout."""
    argv = _mpi_prefix(nprocs) + [str(binary), inp.name]
    log = work_dir / f"{inp.stem}.log"
    with open(log, "w") as f_out:
        r = subprocess.run(argv, cwd=work_dir, stdout=f_out,
                           stderr=subprocess.STDOUT)
    return r.returncode


def _finalize(return_codes, started, window_idx, replicate_idx, leg,
              device, work_dir, output_dir, keep_dcd, final_rc):
    win_out = output_dir / f"window_{window_idx}_rep_{replicate_idx}"
    win_out.mkdir(exist_ok=True)
    for ext in ("en", "log"):
        for p in work_dir.glob(f"*.{ext}"):
            shutil.copy2(p, win_out / p.name)
    if keep_dcd:
        for p in work_dir.glob("*.dcd"):
            shutil.copy2(p, win_out / p.name)
    (output_dir / "run.json").write_text(json.dumps({
        "window_idx": window_idx, "replicate_idx": replicate_idx,
        "leg": leg, "device": device,
        "return_codes": return_codes,
        "duration_seconds": time.monotonic() - started,
    }, indent=2))
    return final_rc


def run(*, setup_dir: Path, window_idx: int, leg: str,
        replicate_idx: int, device: str, nprocs: int,
        stage: str, keep_dcd: bool, q_bin_dir: Path,
        work_dir: Path, output_dir: Path) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    fep_dir = setup_dir / f"FEP{window_idx}"
    if not fep_dir.exists():
        # Fallback: QligFEP historically uses 1-based FEP dirs (FEP1, FEP2, ...);
        # allow callers that pass 0-based window_idx to still resolve.
        alt = setup_dir / f"FEP{window_idx + 1}"
        if alt.exists():
            fep_dir = alt
        else:
            print(f"FEP{window_idx} not found in {setup_dir}", file=sys.stderr)
            return 2

    for item in fep_dir.iterdir():
        shutil.copy2(item, work_dir / item.name)

    binary = _select_binary(device, q_bin_dir)
    return_codes: dict[str, int] = {}
    started = time.monotonic()

    # 1. qprep
    rc = _qprep(q_bin_dir, work_dir)
    return_codes["qprep"] = rc
    if rc != 0:
        return _finalize(return_codes, started, window_idx, replicate_idx, leg,
                         device, work_dir, output_dir, keep_dcd, rc)

    # 2. equilibration
    if stage in ("eq", "both"):
        for eq in sorted(work_dir.glob("eq*.inp")):
            rc = _run_qdyn(binary, nprocs, eq, work_dir)
            return_codes[eq.stem] = rc
            if rc != 0:
                return _finalize(return_codes, started, window_idx, replicate_idx,
                                 leg, device, work_dir, output_dir, keep_dcd, rc)

    # 3. production
    if stage in ("md", "both"):
        for md in sorted(work_dir.glob("md_*.inp")):
            rc = _run_qdyn(binary, nprocs, md, work_dir)
            return_codes[md.stem] = rc
            if rc != 0:
                return _finalize(return_codes, started, window_idx, replicate_idx,
                                 leg, device, work_dir, output_dir, keep_dcd, rc)

    return _finalize(return_codes, started, window_idx, replicate_idx, leg,
                     device, work_dir, output_dir, keep_dcd, 0)


def main():
    a = parse_args()
    return run(
        setup_dir=a.setup_dir, window_idx=a.window_idx, leg=a.leg,
        replicate_idx=a.replicate_idx, device=a.device, nprocs=a.nprocs,
        stage=a.stage, keep_dcd=a.keep_dcd, q_bin_dir=a.q_bin_dir,
        work_dir=a.work_dir, output_dir=a.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
