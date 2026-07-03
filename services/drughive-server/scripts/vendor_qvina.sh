#!/usr/bin/env bash
# Vendor the QuickVina 2 static binary into
# services/drughive-server/qvina/qvina2.1 at a pinned SHA, so the Docker
# build does no network access.
#
#   ./services/drughive-server/scripts/vendor_qvina.sh
#
# Rationale
# ---------
# Upstream DrugHIVE's generate_optimize.py / dock.py shells out to `qvina2.1`
# (or a `docking_cmd` override) for QVina2 molecular docking.  The `bioconda`
# channel has a *landing page* for `qvina` but no actual package files
# (verified 2026-07-02 on linux-64 repodata).  We therefore vendor the
# statically-linked binary shipped in the QVina github repo's `bin/`
# directory (Apache-2.0, ~4.3 MB, no runtime deps).
#
# Github mirror override (CN networks):
#
#   QVINA_URL=https://ghproxy.cn/https://raw.githubusercontent.com/QVina/qvina/f4bb3b1073a0d50bb2f1fdd14d38594f937602ee/bin/qvina2.1 \
#       ./services/drughive-server/scripts/vendor_qvina.sh
#
# License: Apache-2.0 (see QVina github repo LICENSE file).
# Repo: https://github.com/QVina/qvina

set -euo pipefail

QVINA_SHA="${QVINA_SHA:-f4bb3b1073a0d50bb2f1fdd14d38594f937602ee}"
QVINA_URL="${QVINA_URL:-https://raw.githubusercontent.com/QVina/qvina/${QVINA_SHA}/bin/qvina2.1}"
# sha256 of the qvina2.1 binary at the pinned SHA above (verified 2026-07-02).
QVINA_SHA256="${QVINA_SHA256:-bc1f908869181d17b48c9a9b4c2a28cce5264ccf3053e15bce32bb5baf83b43d}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST_DIR="$PROJECT_ROOT/services/drughive-server/qvina"
DST="$DST_DIR/qvina2.1"

mkdir -p "$DST_DIR"

# Retry: CN networks sometimes drop raw.githubusercontent.com connections.
for i in 1 2 3 4 5; do
    if curl -fsSL -o "$DST" "$QVINA_URL"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: qvina2.1 download failed after 5 attempts" >&2; exit 1; }
    echo "  download failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

# Verify sha256 — pin against binary corruption / URL swap.
actual="$(sha256sum "$DST" | awk '{print $1}')"
if [[ "$actual" != "$QVINA_SHA256" ]]; then
    echo "ERROR: sha256 mismatch for qvina2.1" >&2
    echo "  expected: $QVINA_SHA256" >&2
    echo "  actual:   $actual" >&2
    echo "  Delete $DST and re-run, or bump QVINA_SHA256 if the pin is stale." >&2
    exit 1
fi

chmod +x "$DST"

echo "Vendored QVina2 @ ${QVINA_SHA}"
echo "  -> $DST"
ls -la "$DST"
