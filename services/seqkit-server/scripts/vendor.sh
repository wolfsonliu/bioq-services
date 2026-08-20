#!/usr/bin/env bash
# Vendor the upstream SeqKit binary into services/seqkit-server/upstream/ at a
# pinned release, so `docker build` does no network access. Run once before
# each build (and again to upgrade):
#
#   ./services/seqkit-server/scripts/vendor.sh
#
# To use a mirror (CN networks, flaky TLS), prefix the release download base:
#
#   SEQKIT_MIRROR=https://ghproxy.cn/https://github.com/shenwei356/seqkit/releases/download \
#       ./services/seqkit-server/scripts/vendor.sh
#
# To bump the upstream pin, edit SEQKIT_VERSION + SEQKIT_SHA256 + SEQKIT_MD5.

set -euo pipefail

# v2.13.0 (2026). Provenance: github.com/shenwei356/seqkit releases.
SEQKIT_VERSION="${SEQKIT_VERSION:-v2.13.0}"
SEQKIT_SHA256="${SEQKIT_SHA256:-7d686de448464fada1b1988e2e07d693bec68768312da62846bc0e2b502bfc46}"
SEQKIT_MD5="${SEQKIT_MD5:-872368c1e24706dbd1f931d26b38d7d1}"
ASSET="seqkit_linux_amd64.tar.gz"
BASE="${SEQKIT_MIRROR:-https://github.com/shenwei356/seqkit/releases/download}"
URL="$BASE/$SEQKIT_VERSION/$ASSET"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/seqkit-server/upstream"
TMP="$(mktemp -d -t seqkit-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -f "$TMP/$ASSET"
    if curl -fSL --retry 3 --connect-timeout 20 -o "$TMP/$ASSET" "$URL"; then
        break
    fi
    [ "$i" = "5" ] && {
        echo "ERROR: download failed after 5 attempts: $URL" >&2
        exit 1
    }
    echo "  download failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

# Integrity: sha256 (primary) + upstream-published md5 (secondary).
echo "$SEQKIT_SHA256  $TMP/$ASSET" | sha256sum -c -
echo "$SEQKIT_MD5  $TMP/$ASSET" | md5sum -c -

tar -xzf "$TMP/$ASSET" -C "$TMP"
[ -f "$TMP/seqkit" ] || { echo "ERROR: $ASSET did not contain ./seqkit" >&2; exit 1; }
mv "$TMP/seqkit" "$DST/seqkit"
chmod +x "$DST/seqkit"

# Sanity: run it when the host can execute linux/amd64 binaries (skip on
# other arches — the Docker build targets linux/amd64 regardless).
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    "$DST/seqkit" version
fi

echo "vendored seqkit $SEQKIT_VERSION -> $DST/seqkit"
