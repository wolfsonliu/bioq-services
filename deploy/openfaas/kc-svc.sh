#!/usr/bin/env bash
# Create/rotate a Keycloak confidential client with a Service Account
# (OAuth2 client-credentials) in realm `bioq`. Runs INSIDE the keycloak pod:
#   kubectl -n bioq exec -i deploy/keycloak -- bash -s -- <client_id> <secret> [admin] < kc-svc.sh
# ADMIN (3rd arg, non-empty) puts the service-account user in `bioq-admins` → admin role.
set -euo pipefail
CID="$1"; SECRET="$2"; ADMIN="${3:-}"
K=/opt/keycloak/bin/kcadm.sh

"$K" config credentials --server http://localhost:8080 --realm master \
  --user admin --password admin >/dev/null

id="$("$K" get clients -r bioq -q "clientId=$CID" --fields id --format csv --noquotes | head -1)"
if [ -z "$id" ]; then
  "$K" create clients -r bioq -s "clientId=$CID" -s enabled=true -s publicClient=false \
    -s serviceAccountsEnabled=true -s standardFlowEnabled=false \
    -s directAccessGrantsEnabled=false -s "secret=$SECRET" >/dev/null
  id="$("$K" get clients -r bioq -q "clientId=$CID" --fields id --format csv --noquotes | head -1)"
  # audience + groups mappers (so the token carries aud=gateway-server + groups)
  "$K" create "clients/$id/protocol-mappers/models" -r bioq \
    -s name=aud-gateway-server -s protocol=openid-connect \
    -s protocolMapper=oidc-audience-mapper \
    -s 'config."included.custom.audience"=gateway-server' \
    -s 'config."access.token.claim"=true' -s 'config."id.token.claim"=false' >/dev/null
  "$K" create "clients/$id/protocol-mappers/models" -r bioq \
    -s name=groups -s protocol=openid-connect \
    -s protocolMapper=oidc-group-membership-mapper \
    -s 'config."claim.name"=groups' -s 'config."full.path"=false' \
    -s 'config."access.token.claim"=true' -s 'config."id.token.claim"=false' >/dev/null
else
  "$K" update "clients/$id" -r bioq -s "secret=$SECRET" >/dev/null   # rotate secret
fi

if [ -n "$ADMIN" ]; then
  uid="$("$K" get "clients/$id/service-account-user" -r bioq \
        --fields id --format csv --noquotes | head -1)"
  gid="$("$K" get groups -r bioq -q search=bioq-admins \
        --fields id --format csv --noquotes | head -1)"
  "$K" update "users/$uid/groups/$gid" -r bioq \
    -s realm=bioq -s "userId=$uid" -s "groupId=$gid" -n
  echo "service client '$CID' ready (admin — service account in bioq-admins)"
else
  echo "service client '$CID' ready (normal)"
fi
echo "  client_id     : $CID"
echo "  client_secret : $SECRET"
