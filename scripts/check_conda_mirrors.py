#!/usr/bin/env python3
"""Invariant guard for the shared conda mirror mapping.

Every service Dockerfile that writes /root/.condarc must source its mirror
mapping from deploy/conda/mirrors.condarc (PKU), and no TUNA mirror URL may
remain. Exit 0 iff all invariants hold; 1 otherwise (with a report).

Usage: python3 scripts/check_conda_mirrors.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "deploy" / "conda" / "mirrors.condarc"

PKU = "https://mirrors.pku.edu.cn/anaconda"
TUNA = "mirrors.tuna.tsinghua.edu.cn"


def conda_dockerfiles():
    return sorted(
        p for p in (ROOT / "services").glob("*/Dockerfile")
        if "cat > /root/.condarc" in p.read_text()
    )


def main() -> int:
    errors = []

    # Invariant 1: shared file exists, is PKU-only, has both mapping keys.
    if not SHARED.exists():
        errors.append(f"missing {SHARED}")
    else:
        text = SHARED.read_text()
        if TUNA in text:
            errors.append(f"{SHARED} still references TUNA")
        for key in ("default_channels", "custom_channels"):
            if key not in text:
                errors.append(f"{SHARED} missing `{key}`")
        for channel in ("pkgs/main", "pkgs/r", "pkgs/msys2", "cloud"):
            if f"{PKU}/{channel}" not in text:
                errors.append(f"{SHARED} missing {PKU}/{channel}")

    # Invariant 2 + 3: every conda-using Dockerfile is TUNA-free and consumes
    # the shared file via COPY + a STANDALONE `RUN cat ... >>` append. Requiring
    # the standalone RUN (not a substring) also catches the invalid `&& cat ...`
    # glued after a heredoc EOF, which is a shell syntax error.
    for df in conda_dockerfiles():
        text = df.read_text()
        if TUNA in text:
            errors.append(f"{df}: still references TUNA mirror")
        if "COPY deploy/conda/mirrors.condarc" not in text:
            errors.append(f"{df}: missing COPY of shared mirrors.condarc")
        if not any(
            ln.strip() == "RUN cat /tmp/mirrors.condarc >> /root/.condarc"
            for ln in text.splitlines()
        ):
            errors.append(
                f"{df}: missing standalone 'RUN cat /tmp/mirrors.condarc >> /root/.condarc'"
                " (append must be its own RUN, not glued with '&&' after heredoc EOF)"
            )

    if errors:
        print(f"FAILED ({len(errors)} issues):")
        for e in errors:
            print(f"  - {e}")
        return 1

    n = len(conda_dockerfiles())
    print(f"OK: {n} conda Dockerfiles use the shared PKU mirror mapping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())