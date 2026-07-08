#!/usr/bin/env bash
# Vendor the upstream REINVENT4 source into services/reinvent-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/reinvent-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#   REINVENT_REPO=https://ghproxy.cn/https://github.com/MolecularAI/REINVENT4.git \
#       ./services/reinvent-server/scripts/vendor.sh
#
# To bump the pin, edit REINVENT_SHA below.
set -euo pipefail

REINVENT_REPO="${REINVENT_REPO:-https://github.com/MolecularAI/REINVENT4.git}"
REINVENT_SHA="${REINVENT_SHA:-04de385d33f95e97f3960b5c4184a0c0bd3ad7f8}"  # v4.8.24

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${HERE}/upstream/REINVENT4"

mkdir -p "${HERE}/upstream"
rm -rf "${DST}"

# Retry 5x: CN build hosts occasionally hit GnuTLS TLS reset.
for i in 1 2 3 4 5; do
    if git clone "${REINVENT_REPO}" "${DST}"; then break; fi
    echo "clone attempt ${i} failed; retrying..." >&2
    sleep 3
done

git -C "${DST}" checkout "${REINVENT_SHA}"
rm -rf "${DST}/.git"
echo "Vendored REINVENT4 @ ${REINVENT_SHA} → ${DST}"
