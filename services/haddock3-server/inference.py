"""Subprocess wrapper for haddock3-server (approach A — no upstream edits).

Dispatches to HADDOCK3's console scripts and normalises the stdout-only tools
into concrete output files so the framework's `detect_outputs` has something to
find. Subcommands mirror the HTTP endpoints:

    dock              --config <cfg> --workdir <dir> --output-dir <dir>
    score             --pdb <p> --output-dir <dir> [--full] [-p K V ...]
    restrain-bodies   --pdb <p> --output-dir <dir> [--exclude CHAINS]
    actpass-to-ambig  --a1 <f> --a2 <f> --output-dir <dir> [--segid1 A --segid2 B]

The console scripts are resolved next to the current interpreter
(`<venv>/bin/haddock3*`), so this works regardless of PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_BIN = Path(sys.executable).parent
_HADDOCK3 = str(_BIN / "haddock3")
_HADDOCK3_SCORE = str(_BIN / "haddock3-score")
_HADDOCK3_RESTRAINTS = str(_BIN / "haddock3-restraints")

_SCORE_RE = re.compile(r"HADDOCK-score \(emscoring\) = ([-\d.eE+]+)")
_COMP_RE = re.compile(
    r"vdw=([-\d.eE+]+),elec=([-\d.eE+]+),desolv=([-\d.eE+]+),"
    r"air=([-\d.eE+]+),bsa=([-\d.eE+]+)"
)


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a subprocess, teeing its output to our own stdout/stderr for the log."""
    print(f"[haddock3-wrapper] $ {' '.join(argv)}", flush=True)
    return subprocess.run(argv, text=True, **kw)


def _cmd_dock(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)  # run_dir (out/<name>) is created by haddock3
    proc = _run([_HADDOCK3, str(Path(args.config).resolve())], cwd=args.workdir)
    return proc.returncode


def _cmd_score(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    argv = [_HADDOCK3_SCORE, str(Path(args.pdb).resolve())]
    argv += ["--run_dir", str(out.parent / "haddock-score-run")]
    if args.full:
        argv += ["--full"]
    if args.param:
        argv += ["-p", *[tok for pair in args.param for tok in pair]]
    proc = _run(argv, capture_output=True)
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if proc.returncode != 0:
        return proc.returncode

    text = proc.stdout or ""
    m = _SCORE_RE.search(text)
    result: dict = {"raw_stdout": text.strip()}
    if m:
        result["haddock_score"] = float(m.group(1))
    c = _COMP_RE.search(text)
    if c:
        result["components"] = {
            "vdw": float(c.group(1)), "elec": float(c.group(2)),
            "desolv": float(c.group(3)), "air": float(c.group(4)),
            "bsa": float(c.group(5)),
        }
    (out / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if "haddock_score" in result else 1


def _capture_to_file(argv: list[str], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(argv, capture_output=True)
    sys.stderr.write(proc.stderr or "")
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout or "")
        return proc.returncode
    dest.write_text(proc.stdout or "", encoding="utf-8")
    print(f"[haddock3-wrapper] wrote {dest} ({dest.stat().st_size} bytes)", flush=True)
    return 0 if dest.stat().st_size > 0 else 1


def _cmd_restrain_bodies(args: argparse.Namespace) -> int:
    argv = [_HADDOCK3_RESTRAINTS, "restrain_bodies", str(Path(args.pdb).resolve())]
    if args.exclude:
        argv += ["--exclude", args.exclude]
    return _capture_to_file(argv, Path(args.output_dir) / "restraints.tbl")


def _cmd_actpass_to_ambig(args: argparse.Namespace) -> int:
    argv = [
        _HADDOCK3_RESTRAINTS, "active_passive_to_ambig",
        str(Path(args.a1).resolve()), str(Path(args.a2).resolve()),
        "--segid-one", args.segid1, "--segid-two", args.segid2,
    ]
    return _capture_to_file(argv, Path(args.output_dir) / "ambig.tbl")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="haddock3-wrapper")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dock")
    d.add_argument("--config", required=True)
    d.add_argument("--workdir", required=True)
    d.add_argument("--output-dir", required=True)
    d.set_defaults(func=_cmd_dock)

    s = sub.add_parser("score")
    s.add_argument("--pdb", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--full", action="store_true")
    s.add_argument("-p", "--param", nargs=2, action="append", metavar=("KEY", "VALUE"))
    s.set_defaults(func=_cmd_score)

    rb = sub.add_parser("restrain-bodies")
    rb.add_argument("--pdb", required=True)
    rb.add_argument("--output-dir", required=True)
    rb.add_argument("--exclude", default=None)
    rb.set_defaults(func=_cmd_restrain_bodies)

    ap = sub.add_parser("actpass-to-ambig")
    ap.add_argument("--a1", required=True)
    ap.add_argument("--a2", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--segid1", default="A")
    ap.add_argument("--segid2", default="B")
    ap.set_defaults(func=_cmd_actpass_to_ambig)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
