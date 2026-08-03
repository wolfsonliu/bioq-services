#!/usr/bin/env bash
# local-down.sh — tear down the local kind + OpenFaaS bioq-services deployment
# created by local-up.sh.
#
# Usage:
#   ./local-down.sh            # stop port-forward + delete the kind cluster
#   ./local-down.sh --purge    # also remove the work dir (kubeconfig, shared vol, tools)
set -euo pipefail

CLUSTER="${BIOQ_CLUSTER:-bioq}"
WORKDIR="${BIOQ_WORKDIR:-$HOME/.cache/bioq-local}"
export PATH="$WORKDIR/bin:$PATH"

log() { printf '\033[1;34m[local-down]\033[0m %s\n' "$*"; }

for pf in port-forward keycloak-port-forward; do
  PF_PID_FILE="$WORKDIR/$pf.pid"
  if [ -f "$PF_PID_FILE" ]; then
    kill "$(cat "$PF_PID_FILE")" 2>/dev/null && log "stopped $pf" || true
    rm -f "$PF_PID_FILE"
  fi
done

if command -v kind >/dev/null 2>&1 && kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  log "deleting kind cluster '$CLUSTER'"
  kind delete cluster --name "$CLUSTER"
else
  log "kind cluster '$CLUSTER' not found (nothing to delete)"
fi

if [ "${1:-}" = "--purge" ]; then
  log "purging work dir $WORKDIR"
  rm -rf "$WORKDIR"
fi

log "done"
