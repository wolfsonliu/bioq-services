#!/usr/bin/env bash
# Vendor the prebuilt DIAMOND binary tarball into
# services/diamond-server/upstream/diamond-linux64.tar.gz, so `docker build`
# does no network access. Run once before each build (and again to upgrade):
#
#   ./services/diamond-server/scripts/vendor.sh
#
# To pin a different release (only tags publishing a `diamond-linux64.tar.gz`
# asset are valid — check the Releases page):
#
#   DIAMOND_VERSION=v2.2.1 ./services/diamond-server/scripts/vendor.sh
#
# NOTE: like mmseqs2-server, this vendors the upstream *prebuilt binary* release
# artifact rather than building from source — DIAMOND is CPU-only and the
# official static linux64 build covers makedb/blastp/blastx/cluster/deepclust.
# Source pin for provenance: v2.2.1 == SHA
# 2a390b526f7e99b6bfdb21b69e008140e10ad88c (github.com/bbuchfink/diamond).

set -euo pipefail

DIAMOND_VERSION="${DIAMOND_VERSION:-v2.2.1}"
DIAMOND_URL="${DIAMOND_URL:-https://github.com/bbuchfink/diamond/releases/download/${DIAMOND_VERSION}/diamond-linux64.tar.gz}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/diamond-server/upstream"
OUT="$DST/diamond-linux64.tar.gz"
VERSION_FILE="$DST/VERSION"

mkdir -p "$DST"

# Retry: GitHub connections from CN hosts sometimes reset mid-transfer.
for i in 1 2 3 4 5; do
    echo "fetching diamond ${DIAMOND_VERSION} from $DIAMOND_URL (attempt $i)"
    if curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
            --continue-at - -o "$OUT" "$DIAMOND_URL"; then
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

# Basic sanity: gzip-decompressible + contains a `diamond` binary entry.
if ! tar -tzf "$OUT" | grep -qx 'diamond'; then
    echo "ERROR: downloaded tarball missing top-level 'diamond' binary — corrupt?" >&2
    rm -f "$OUT"
    exit 1
fi

echo "$DIAMOND_VERSION" > "$VERSION_FILE"

echo "Vendored diamond-linux64 ${DIAMOND_VERSION}"
echo "  -> $OUT"
du -sh "$OUT"
