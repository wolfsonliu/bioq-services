#!/usr/bin/env bash
# Stage a compiled CNS executable for haddock3-server.
#
# CNS is license-gated (academic users request it free at http://cns-online.org)
# and CANNOT be auto-downloaded — you must obtain + compile it yourself, patched
# with the HADDOCK files shipped under upstream/varia/cns1.3/. See
# services/haddock3-server/README.md (## CNS) and upstream docs/pages/CNS.md.
#
# This helper just copies YOUR already-compiled `cns` binary into the layout the
# service expects (a `cns/` dir containing the executable named `cns`).
#
# Local stage (default -> services/haddock3-server/weights/cns/cns):
#   CNS_SRC=/path/to/cns_solve_1.3/.../source/cns_solve-XXXX.exe \
#       ./services/haddock3-server/scripts/stage_cns.sh
#
# Stage straight to NAS (FC deployment):
#   CNS_SRC=/path/to/cns.exe \
#   WEIGHTS_DST=/mnt/nas/data/models/haddock3 \
#       ./services/haddock3-server/scripts/stage_cns.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"

if [[ -z "${CNS_SRC:-}" ]]; then
    echo "ERROR: set CNS_SRC to your compiled CNS executable." >&2
    echo "       CNS is license-gated; see README.md (## CNS)." >&2
    exit 1
fi
if [[ ! -f "$CNS_SRC" ]]; then
    echo "ERROR: CNS_SRC not found: $CNS_SRC" >&2
    exit 1
fi

mkdir -p "$DST/cns"
install -m 0755 "$CNS_SRC" "$DST/cns/cns"

echo "Staged CNS -> $DST/cns/cns"
"$DST/cns/cns" <<<'stop' >/dev/null 2>&1 && echo "  (binary runs)" || \
    echo "  WARNING: binary did not run here (arch/libgfortran mismatch?) — verify on the target."
