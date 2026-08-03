#!/usr/bin/env bash
# OIDC spike: prove the gateway's existing JWT/JWKS auth layer verifies a token
# issued by a real third-party IdP (Keycloak), with no gateway code changes.
#
# Steps:
#   1. bring up Keycloak (realm `bioq`, client `bioq-gateway`, user alice/alice)
#   2. obtain an access token via the OAuth2 password grant (direct access grant)
#   3. Step A — call the gateway's own verify_jwt() against Keycloak's JWKS
#   4. Step B — run the gateway locally and hit auth-gated /v1/services:
#                no token -> 401, valid Bearer -> 200
#
# Usage:  ./verify.sh          # up + verify (leaves Keycloak running)
#         ./verify.sh --down   # tear Keycloak down
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
GATEWAY_DIR="$REPO_ROOT/gateway"
PYBIN="$GATEWAY_DIR/.venv/bin/python"

KC="http://localhost:8080"
REALM="bioq"
CLIENT="bioq-gateway"
USER="alice"
PASS="alice"
JWKS="$KC/realms/$REALM/protocol/openid-connect/certs"
GW_PORT="9001"
GW="http://localhost:$GW_PORT"

if [ "${1:-}" = "--down" ]; then
    (cd "$HERE" && docker compose down)
    exit 0
fi

log() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mFAIL: %s\033[0m\n' "$*"; exit 1; }

[ -x "$PYBIN" ] || fail "gateway venv not found ($PYBIN); run 'cd gateway && uv sync' first"

# --- 1. Keycloak up ---
log "starting Keycloak (docker compose up -d)"
(cd "$HERE" && docker compose up -d)

log "waiting for realm '$REALM' to be ready"
for i in $(seq 1 60); do
    if curl -fsS "$KC/realms/$REALM/.well-known/openid-configuration" >/dev/null 2>&1; then
        echo "  ready after ${i}0s"; break
    fi
    [ "$i" = 60 ] && fail "Keycloak realm not ready after 600s"
    sleep 10
done

# --- 2. get an access token (password grant) ---
log "requesting access token for $USER via password grant"
TOKEN="$(curl -fsS -X POST "$KC/realms/$REALM/protocol/openid-connect/token" \
    -d "client_id=$CLIENT" -d "grant_type=password" \
    -d "username=$USER" -d "password=$PASS" \
    | "$PYBIN" -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')"
[ -n "$TOKEN" ] || fail "no access_token returned"
echo "  token acquired (${#TOKEN} chars)"

# --- 3. Step A: gateway verify_jwt() against Keycloak JWKS ---
log "Step A — gateway verify_jwt() against Keycloak JWKS"
GW_TOKEN="$TOKEN" GW_JWKS="$JWKS" "$PYBIN" - <<'PY'
import os
from server.auth.jwt_verifier import verify_jwt
claims = verify_jwt(os.environ["GW_TOKEN"], jwks_url=os.environ["GW_JWKS"],
                    audience="gateway-server")
print("  verified OK")
print("  sub   :", claims.get("sub"))
print("  aud   :", claims.get("aud"))
print("  iss   :", claims.get("iss"))
print("  groups:", claims.get("groups"))
PY

# --- 4. Step B: end-to-end through a locally-run gateway ---
log "Step B — running gateway on :$GW_PORT (BYPASS_VPC=false, JWKS=Keycloak)"
TMP="$(mktemp -d)"
echo "services: {}" > "$TMP/services.yaml"
GW_PID=""
cleanup() { [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT

( cd "$GATEWAY_DIR" && \
  GATEWAY_AUTH__BYPASS_VPC=false \
  GATEWAY_AUTH__JWT_JWKS_URL="$JWKS" \
  GATEWAY_AUTH__JWT_AUDIENCE="gateway-server" \
  GATEWAY_DB_URL="sqlite:///$TMP/gw.db" \
  GATEWAY_JOBS_BASE_DIR="$TMP/jobs" \
  GATEWAY_REGISTRY_PATH="$TMP/services.yaml" \
  "$PYBIN" -m uvicorn server.app:app --port "$GW_PORT" --log-level warning ) &
GW_PID=$!

for i in $(seq 1 30); do
    curl -fsS "$GW/healthz" >/dev/null 2>&1 && break
    [ "$i" = 30 ] && fail "gateway did not come up on :$GW_PORT"
    sleep 1
done

code_noauth="$(curl -s -o /dev/null -w '%{http_code}' "$GW/v1/services")"
code_auth="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "$GW/v1/services")"
echo "  GET /v1/services  (no token)      -> $code_noauth  (expect 401)"
echo "  GET /v1/services  (Bearer token)  -> $code_auth  (expect 200)"

[ "$code_noauth" = "401" ] || fail "expected 401 without token, got $code_noauth"
[ "$code_auth" = "200" ]  || fail "expected 200 with Keycloak token, got $code_auth"

log "PASS — existing JWT/JWKS layer verifies Keycloak tokens end-to-end"
echo "Keycloak left running at $KC (admin/admin). Tear down: ./verify.sh --down"
