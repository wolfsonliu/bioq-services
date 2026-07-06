#!/usr/bin/env bash
# Vendor qligfep + Q6 sources into services/qligfep-server/upstream/ at pinned SHAs,
# and rewrite qligfep/settings.py into an env-driven shim (see
# engineering/decisions/2026-07-06-qligfep-server-design.md §6.1).
#
#   ./services/qligfep-server/scripts/vendor.sh
#
# Mirror override (CN networks):
#
#   QLIGFEP_REPO=https://ghproxy.cn/https://github.com/qusers/qligfep.git \
#   Q6_REPO=https://ghproxy.cn/https://github.com/esguerra/Q6.git \
#       ./services/qligfep-server/scripts/vendor.sh
#
# To bump pins, edit *_SHA below.

set -euo pipefail

QLIGFEP_REPO="${QLIGFEP_REPO:-https://github.com/qusers/qligfep.git}"
QLIGFEP_SHA="${QLIGFEP_SHA:-5751f83726529070af2eab5b776706d6a858f8fe}"
Q6_REPO="${Q6_REPO:-https://github.com/esguerra/Q6.git}"
# Q6_SHA is left empty until first successful clone — vendor.sh prints the
# resolved HEAD SHA and instructs the operator to pin it in-place.  Once
# pinned, checkout that SHA and verify.
Q6_SHA="${Q6_SHA:-202d90cff0f841b6be9bc1d57fbf26a41c002b8f}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/qligfep-server/upstream"
TMP="$(mktemp -d -t qligfep-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT
mkdir -p "$DST"

vendor_one() {
    local name="$1" repo="$2" sha="$3" dst_sub="$4"
    echo "=== Vendoring ${name} @ ${sha:-HEAD} ==="
    for i in 1 2 3 4 5; do
        rm -rf "$TMP/$name"
        if git clone --filter=blob:none --no-checkout "$repo" "$TMP/$name"; then
            break
        fi
        [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
        echo "  clone failed, retrying in $((i*10))s ..."
        sleep $((i*10))
    done
    cd "$TMP/$name"
    if [ -z "$sha" ]; then
        git checkout HEAD >/dev/null 2>&1 || true
        sha="$(git rev-parse HEAD)"
        echo "  ${name} pin resolved to: ${sha}"
        echo "  --> Edit vendor.sh to set ${name^^}_SHA=${sha} before committing."
    else
        git checkout "$sha"
        actual="$(git rev-parse HEAD)"
        [[ "$actual" = "$sha" ]] || {
            echo "ERROR: ${name} HEAD mismatch (got $actual, expected $sha)" >&2
            exit 1
        }
    fi
    rm -rf .git
    rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
        "$TMP/$name/" "$DST/$dst_sub/"
    cd - >/dev/null
    echo "  -> $DST/$dst_sub"
    du -sh "$DST/$dst_sub"
}

vendor_one qligfep "$QLIGFEP_REPO" "$QLIGFEP_SHA" qligfep
vendor_one Q6      "$Q6_REPO"      "$Q6_SHA"      Q6

# --- shim: rewrite qligfep/settings.py to read env ---
python3 - <<'PY' "$DST/qligfep/settings.py"
import sys, pathlib
p = pathlib.Path(sys.argv[1])
new = '''\
"""Env-driven qligfep settings shim (vendored by services/qligfep-server/scripts/vendor.sh)."""
import os
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
FF_DIR = os.path.join(ROOT_DIR, "FF")
INPUT_DIR = os.path.join(ROOT_DIR, "INPUTS")
_Q = os.environ.get("QLIGFEP_Q_BIN_DIR", "/opt/Q6/bin").rstrip("/") + "/"
Q_DIR = {"LOCAL": _Q, "CSB": _Q, "SLURM": _Q}
BIN = os.path.join(ROOT_DIR, "bin")
SCHROD_DIR = os.environ.get("QLIGFEP_SCHROD_DIR", "")
DEFAULT = os.environ.get("QLIGFEP_DEFAULT_CLUSTER", "LOCAL")
LOCAL = {
    "NODES": "1", "NTASKS": "8", "TIME": "1-00:00:00",
    "PARTITION": "", "EXCLUDE": "", "MODULES": "\\n",
    "QDYN":  "qdyn=" + Q_DIR["LOCAL"] + "qdynp",
    "QPREP": Q_DIR["LOCAL"] + "qprep",
    "QFEP":  Q_DIR["LOCAL"] + "qfep",
    "QCALC": Q_DIR["LOCAL"] + "qcalc",
}
CSB = LOCAL
SLURM = LOCAL
'''
p.write_text(new)
print(f"Rewrote {p} into env-driven shim.")
PY

echo "=== Vendor complete ==="
