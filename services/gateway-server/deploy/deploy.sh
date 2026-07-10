#!/usr/bin/env bash
# Deploy gateway-server on an ECS host via docker compose.
#
# Usage (run in this directory on the ECS host):
#   ./deploy.sh            # (pull if remote) + up -d + health wait
#   ./deploy.sh --build    # build the image from the repo root first, then up -d
#
# Prereqs: docker + docker compose plugin; a `.env` here (copy .env.example).
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "error: .env not found — copy .env.example to .env and fill it in." >&2
  exit 1
fi

# Load config (GATEWAY_DATA_DIR) so we can pre-create the persistent dirs the
# SQLite DB + job scratch live in (bind-mounted at /data).
set -a
# shellcheck disable=SC1091
source .env
set +a
DATA_DIR="${GATEWAY_DATA_DIR:-./data}"
mkdir -p "${DATA_DIR}/gateway" "${DATA_DIR}/gateway_jobs"

if [[ "${1:-}" == "--build" ]]; then
  echo ">> building image from repo root (make build-gateway-server)..."
  ( cd ../../.. && make build-gateway-server )
fi

# Best-effort pull for registry images; harmless no-op / ignored for local tags.
docker compose pull 2>/dev/null || true

echo ">> starting gateway-server..."
docker compose up -d

echo ">> waiting for health..."
for _ in $(seq 1 30); do
  if docker compose exec -T gateway python -c \
      "import urllib.request; urllib.request.urlopen('http://localhost:9000/healthz')" 2>/dev/null; then
    echo "gateway-server is healthy."
    docker compose ps
    exit 0
  fi
  sleep 2
done

echo "warning: health check did not pass within ~60s; recent logs:" >&2
docker compose logs --tail 50 gateway || true
exit 1
