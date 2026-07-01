#!/usr/bin/env bash
# Vendor the MMseqs2 GPU binary into
# services/mmseqs2-server/upstream/mmseqs-linux-gpu.tar.gz, so `docker build`
# does no network access.  Run once before each build (and again to upgrade):
#
#   ./services/mmseqs2-server/scripts/vendor.sh
#
# To pin a different release tag (only tags that publish a
# `mmseqs-linux-gpu.tar.gz` asset are valid — check the Releases page):
#
#   MMSEQS_VERSION=18-8cc5c ./services/mmseqs2-server/scripts/vendor.sh
#
# NOTE: unlike the other services' vendor.sh which git-clones source, this
# service vendors a *prebuilt binary tarball* — the mmseqs2 GPU build is
# non-trivial (CUDA + AVX2 dispatcher) so we deliberately ship the upstream
# release artifact rather than building from source.  The ColabFold Python
# orchestration is separately vendored (manual copy — see VENDOR_INFO.md).

set -euo pipefail

MMSEQS_VERSION="${MMSEQS_VERSION:-18-8cc5c}"
MMSEQS_URL="${MMSEQS_URL:-https://github.com/soedinglab/MMseqs2/releases/download/${MMSEQS_VERSION}/mmseqs-linux-gpu.tar.gz}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/mmseqs2-server/upstream"
OUT="$DST/mmseqs-linux-gpu.tar.gz"
VERSION_FILE="$DST/VERSION"

mkdir -p "$DST"

# Retry: GitHub connections from CN hosts sometimes reset mid-transfer.
for i in 1 2 3 4 5; do
    echo "fetching mmseqs ${MMSEQS_VERSION} from $MMSEQS_URL (attempt $i)"
    if curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
            --continue-at - -o "$OUT" "$MMSEQS_URL"; then
        break
    fi
    [ "$i" = "5" ] && {
        echo "ERROR: download failed after 5 attempts" >&2
        # Only wipe a partial download — keep any prior successful vendor intact.
        [ -s "$OUT" ] || rm -f "$OUT"
        exit 1
    }
    echo "  fetch failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

# Basic sanity: gzip-decompressible + contains an `mmseqs/bin/mmseqs` entry.
if ! tar -tzf "$OUT" | grep -q '^mmseqs/bin/mmseqs$'; then
    echo "ERROR: downloaded tarball missing mmseqs/bin/mmseqs — corrupt?" >&2
    rm -f "$OUT"
    exit 1
fi

echo "$MMSEQS_VERSION" > "$VERSION_FILE"

echo "Vendored mmseqs-linux-gpu ${MMSEQS_VERSION}"
echo "  -> $OUT"
du -sh "$OUT"
